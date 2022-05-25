from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend("agg")

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as TorchDDP
from hfai.nn.parallel import DistributedDataParallel as HfaiDDP

from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tensorboardX import SummaryWriter

import hfai
import hfai.distributed as dist
from hfai.datasets import ImageNet, GoogleConceptualCaption
import clip

from data import (
    tokenize,
    imagenet_classnames,
    openai_imagenet_template,
)
from utils import *


parser = ArgumentParser(description="Train CLIP")
parser.add_argument("--ddp", type=str, default='hfai', choices=['torch', 'hfai'])
parser.add_argument("--to_hfai", action='store_true')
args = parser.parse_args()


# follow openclip
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
writer = None


class PairwiseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

    def forward(self, img_embeds, text_embeds, logit_scale):
        logit_scale = logit_scale.mean()

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        all_img_embeds = [torch.zeros_like(img_embeds) for _ in range(world_size)]
        all_text_embeds = [torch.zeros_like(text_embeds) for _ in range(world_size)]
        dist.all_gather(all_img_embeds, img_embeds)
        dist.all_gather(all_text_embeds, text_embeds)

        # for backward
        all_img_embeds[rank] = img_embeds
        all_text_embeds[rank] = text_embeds
        all_img_embeds = torch.cat(all_img_embeds, 0)
        all_text_embeds = torch.cat(all_text_embeds, 0)

        scores = all_img_embeds @ all_text_embeds.t()
        logits_per_image = logit_scale * scores
        logits_per_text = logits_per_image.t()

        ground_truth = torch.arange(len(logits_per_image)).long().cuda()
        loss1 = self.loss_img(logits_per_image, ground_truth)
        loss2 = self.loss_txt(logits_per_text, ground_truth)
        total_loss = (loss1 + loss2) / 2

        return total_loss, scores


def train(dataloader, model, optimizer, criterion, scaler, scheduler, use_amp, epoch, local_rank, start_step, acc):

    model.train()
    rank = dist.get_rank()
    steps_per_epoch = len(dataloader) + start_step

    for step, batch in enumerate(dataloader):
        step += start_step
        lr = scheduler.step(epoch + step / steps_per_epoch)

        images, texts = [x.cuda(non_blocking=True) for x in batch]

        with torch.cuda.amp.autocast(enabled=use_amp):
            image_embeds, text_embeds, logit_scale = model(images, texts)
            loss, logits = criterion(image_embeds, text_embeds, logit_scale)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m = model.module
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        # save checkpoint if going to suspend
        model.try_save(epoch, step, others=acc)

        dist.all_reduce(loss)
        loss = loss.item() / dist.get_world_size()
        if local_rank == 0 and step % 10 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss:.3f}, lr: {lr:.6f}", flush=True)

        global_steps = epoch * steps_per_epoch + step
        if rank == 0 and step % 10 == 0:
            writer.add_scalar("lr", lr, global_steps)
            writer.add_scalar("loss", loss, global_steps)
            writer.add_scalar("scale", m.logit_scale.data, global_steps)

        if rank == 0 and step % 100 == 0:
            fig = plt.figure()
            scores = logits.float().cpu().detach().numpy()
            plt.imshow(scores, interpolation="nearest")
            writer.add_figure("scores", fig, global_steps)


@torch.no_grad()
def validate(model, dataloader, epoch, local_rank):
    model.eval()
    zeroshot_weights = []
    for classname in imagenet_classnames:
        texts = [template(classname) for template in openai_imagenet_template]
        texts = tokenize(texts).cuda()
        class_embeds = model.module.encode_text(texts)
        class_embeds /= class_embeds.norm(dim=-1, keepdim=True)
        class_embed = class_embeds.mean(dim=0)
        class_embed /= class_embed.norm()
        zeroshot_weights.append(class_embed)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    correct1, correct5, total = torch.zeros(3).cuda()
    for step, batch in enumerate(dataloader):
        images, labels = [x.cuda(non_blocking=True) for x in batch]
        img_embeds = model.module.encode_image(images)

        img_embeds /= img_embeds.norm(dim=-1, keepdim=True)
        logits = img_embeds @ zeroshot_weights

        _, preds = logits.topk(5, -1, True, True)
        correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
        correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
        total += images.size(0)

    for x in [correct1, correct5, total]:
        dist.reduce(x, 0)

    if local_rank == 0:
        acc1 = 100 * correct1.item() / total.item()
        acc5 = 100 * correct5.item() / total.item()
        print(f"Epoch: {epoch}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%", flush=True)

        if dist.get_rank() == 0:
            writer.add_scalar("Acc@1", acc1, epoch)
            writer.add_scalar("Acc@5", acc5, epoch)

    return correct1.item() / total.item()


def main(local_rank):
    # hyper parameters
    warmup_epochs = 3
    epochs = 30
    batch_size = 64  # 8 nodes, batch size 4096
    base_lr = 1.5e-4
    lr = None
    min_lr = 0

    use_amp = True  # mixed precision
    model = "clip_rn50"

    save_path = Path("output/RN50")
    log_path = save_path / "runs"
    save_path.mkdir(exist_ok=True, parents=True)

    init_dist(local_rank)

    # fix the seed for reproducibility
    seed = dist.get_rank()
    torch.manual_seed(seed)

    if dist.get_rank() == 0:
        global writer
        writer = SummaryWriter(log_path)

    total_batch_size = batch_size * dist.get_world_size()
    lr = lr if lr else base_lr * total_batch_size / 256

    # model, dataloader, optimizer
    model = clip.__dict__[model]()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.to_hfai:
        model = hfai.nn.to_hfai(model, verbose=True)

    DDP = HfaiDDP if args.ddp == 'hfai' else TorchDDP
    model = DDP(model.cuda(), device_ids=[local_rank])

    # train dataloader
    img_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        lambda x: x.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])  # train transform
    def train_transfrom(img, text):
        img = img_transform(img)
        text = tokenize(text)
        return img, text
    train_dataset = GoogleConceptualCaption("train", transform=train_transfrom)
    train_datasampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = train_dataset.loader(batch_size, sampler=train_datasampler, num_workers=4, pin_memory=True)

    # val dataloader
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])  # val transform
    val_dataset = ImageNet("val", transform=val_transform)
    val_datasampler = DistributedSampler(val_dataset)
    val_dataloader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=4, pin_memory=True)

    optimizer = configure_optimizer(model, lr)
    criterion = PairwiseLoss().cuda()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = CosineLRWarmUp(optimizer, warmup_epochs=warmup_epochs, epochs=epochs, lr=lr, min_lr=min_lr)

    # 自动断点训练
    ckpt_path = str(save_path / "latest.pt")
    start_epoch, start_step, acc = hfai.checkpoint.init(model, optimizer, amp_scaler=scaler, ckpt_path=ckpt_path)
    best_acc = acc or 0

    # train, validate
    for epoch in range(start_epoch, epochs):
        # resume from epoch and step
        train_datasampler.set_epoch(epoch)
        train_dataloader.set_step(start_step)

        train(
            train_dataloader,
            model,
            optimizer,
            criterion,
            scaler,
            scheduler,
            use_amp,
            epoch,
            local_rank,
            start_step,
            best_acc,
        )
        start_step = 0  # reset

        acc = validate(model, val_dataloader, epoch, local_rank)

        # save
        if dist.get_rank() == 0:
            state = {"model": model.module.state_dict(), "epoch": epoch, "acc": acc}
            if acc > best_acc:
                best_acc = acc
                print(f"New Best Acc: {100 * acc:.2f}%!")
                torch.save(state, save_path / "best.pt")

            if epoch % 5 == 0 or epoch == epochs - 1:
                torch.save(state, save_path / f"{epoch:04d}.pt")


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)

import math
import os
from pathlib import Path
import numpy as np
import torch
from hfai.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tensorboardX import SummaryWriter

from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

import hfai
import hfai.distributed as dist

from hfai.datasets import ImageNet
from ffrecord.torch import DataLoader

import lr_decay as lrd
import mae
from utils import init_dist


writer = None

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class CosineLRWarmUp:
    def __init__(self, optimizer, warmup_epochs, epochs, lr, min_lr):
        self.optimizer = optimizer
        self.wepochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr

    def step(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.wepochs:
            lr = self.lr * epoch / self.wepochs
        else:
            angle = math.pi * (epoch - self.wepochs) / (self.epochs - self.wepochs)
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (1.0 + math.cos(angle))

        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr


def train(
    dataloader,
    model,
    optimizer,
    criterion,
    scheduler,
    mixup,
    epoch,
    local_rank,
    start_step,
    best_acc,
):
    model.train()
    steps_per_epoch = len(dataloader) + start_step

    for step, batch in enumerate(dataloader):
        step += start_step
        lr = scheduler.step(epoch + step / steps_per_epoch)

        imgs, labels = [x.cuda(non_blocking=True) for x in batch]
        imgs, labels = mixup(imgs, labels)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save checkpoint if going to suspend
        rank = torch.distributed.get_rank()
        model.try_save(epoch, step + 1, others=best_acc)

        # log
        dist.all_reduce(loss)
        loss = loss.item() / dist.get_world_size()
        if local_rank == 0 and step % 10 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss:.3f}, lr: {lr:.6f}", flush=True)

        global_steps = epoch * steps_per_epoch + step
        if rank == 0 and step % 10 == 0:
            writer.add_scalar("lr", lr, global_steps)
            writer.add_scalar("loss", loss, global_steps)


def validate(dataloader, model, epoch, local_rank):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            imgs, labels = [x.cuda(non_blocking=True) for x in batch]
            outputs = model(imgs)
            _, preds = outputs.topk(5, -1, True, True)
            correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
            correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
            total += imgs.size(0)

    for x in [correct1, correct5, total]:
        dist.all_reduce(x)

    if local_rank == 0:
        acc1 = 100 * correct1.item() / total.item()
        acc5 = 100 * correct5.item() / total.item()
        print(f"Epoch: {epoch}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%", flush=True)

    return correct1.item() / total.item()


def main(local_rank):
    # hyper parameters
    warmup_epochs = 5
    batch_size = 16  # use 8 nodes, batch size 1024
    num_workers = 4
    lr = None
    min_lr = 1e-6
    linear_probe = False

    epochs = 50
    base_lr = 1e-3
    weight_decay = 0.05
    layer_decay = 0.75
    droppath = 0.3
    model = "vit_huge_patch14"
    mae_ckpt = "output/pretrain/mae_vit_huge_patch14/0799.pt"
    save_path = Path("output/finetune") / model

    log_path = save_path / "runs"
    save_path.mkdir(exist_ok=True, parents=True)

    # init dist
    rank, world_size = init_dist(local_rank)

    # fix the seed for reproducibility
    torch.manual_seed(12345)
    np.random.seed(12345)
    torch.backends.cudnn.benchmark = True

    if rank == 0 and local_rank == 0:
        global writer
        writer = SummaryWriter(log_path)

    total_batch_size = batch_size * world_size
    lr = lr if lr else base_lr * total_batch_size / 256

    model = mae.__dict__[model](num_classes=1000, droppath=droppath, global_pool=True)
    model = hfai.nn.to_hfai(model)
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])

    # mixup two images and labels
    mixup = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=1000,
    )
    criterion = SoftTargetCrossEntropy()

    train_transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=None,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=mean,
        std=std,
    )
    train_dataset = ImageNet(split="train", transform=train_transform)
    train_datasampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = train_dataset.loader(batch_size, sampler=train_datasampler, num_workers=8, pin_memory=True, drop_last=True)

    mode = transforms.InterpolationMode.BICUBIC
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=mode),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_dataset = ImageNet(split="val", transform=val_transform)
    val_datasampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=8, pin_memory=True)

    param_groups = lrd.param_groups_lrd(model.module, weight_decay, layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    scheduler = CosineLRWarmUp(optimizer, warmup_epochs=warmup_epochs, epochs=epochs, lr=lr, min_lr=min_lr)

    # auto reload model
    ckpt_path = str(save_path / "latest.pt")
    start_epoch, start_step, best_acc = hfai.checkpoint.init(model, optimizer, ckpt_path=ckpt_path)
    best_acc = best_acc or 0.

    # load from pretrained model
    if start_epoch == 0 and start_step == 0:
        state = torch.load(mae_ckpt, map_location="cpu")
        missing_keys = model.module.load_state_dict(state["model"], strict=False)[0]
        print(missing_keys)
        assert len(missing_keys) <= 4, missing_keys
        if linear_probe:
            for param in model.module.encoder.parameters():
                param.requires_grad = False
        print(f"loaded model from pretrained mae model {mae_ckpt}")

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
            scheduler,
            mixup,
            epoch,
            local_rank,
            start_step,
            best_acc,
        )
        start_step = 0  # reset

        acc = validate(val_dataloader, model, epoch, local_rank)

        # save
        if rank == 0 and local_rank == 0:
            writer.add_scalar("Acc@1", acc, epoch)
            state = {"model": model.module.state_dict(), "acc": acc}
            if epoch % 20 == 0:
                torch.save(state, save_path / f"{epoch:04d}.pt")

            if acc > best_acc:
                best_acc = acc
                print(f"New Best Acc: {100*acc:.2f}%!")
                torch.save(state, save_path / "best.pt")


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus)

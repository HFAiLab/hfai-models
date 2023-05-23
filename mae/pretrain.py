from argparse import ArgumentParser
import math
from pathlib import Path
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import timm.optim

from torch.nn.parallel import DistributedDataParallel as TorchDDP
from haiscale.ddp import DistributedDataParallel as HfaiDDP

import hfai
import hfai.distributed as dist
from hfai.datasets import ImageNet
from utils import init_dist

import mae


parser = ArgumentParser(description="Pre-train Masked Autoencoder")
parser.add_argument("--ddp", type=str, default='hfai', choices=['torch', 'hfai'])
parser.add_argument("--no_to_hfai", action='store_true')
args = parser.parse_args()


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


def train(dataloader, model, optimizer, scheduler, epoch, local_rank, start_step):
    model.train()
    steps_per_epoch = len(dataloader) + start_step

    for step, batch in enumerate(dataloader):
        step += start_step
        lr = scheduler.step(epoch + step / steps_per_epoch)

        imgs = batch[0].cuda()
        pred, mask, loss = model(imgs, mask_ratio=0.75)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save checkpoint if going to suspend
        model.try_save(epoch, step + 1)

        # log
        dist.all_reduce(loss)
        loss = loss.item() / dist.get_world_size()
        if local_rank == 0 and step % 10 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss:.3f}, lr: {lr:.6f}", flush=True)

        rank = dist.get_rank()
        global_steps = epoch * steps_per_epoch + step
        if rank == 0 and step % 10 == 0:
            writer.add_scalar("lr", lr, global_steps)
            writer.add_scalar("loss", loss, global_steps)

        if rank == 0 and step % 100 == 0:
            std1 = torch.tensor(std).view(-1, 1, 1).to(pred.device)
            mean1 = torch.tensor(mean).view(-1, 1, 1).to(pred.device)
            pred = pred[0] * std1 + mean1
            img = imgs[0] * std1 + mean1
            img = torch.cat([img, pred], axis=2).clamp(0, 1)
            writer.add_image("img-pred", img, global_steps)


def main(local_rank):
    # hyper parameters
    warmup_epochs = 40
    epochs = 800
    batch_size = 64  # 8 nodes, batch size 4096
    base_lr = 1.5e-4
    lr = None
    weight_decay = 0.05
    min_lr = 0
    model = "mae_vit_huge_patch14"
    save_path = Path("output/pretrain") / model

    log_path = save_path / "runs"
    save_path.mkdir(exist_ok=True, parents=True)

    rank, world_size = init_dist(local_rank)

    # fix the seed for reproducibility
    torch.manual_seed(12345)

    if rank == 0:
        global writer
        writer = SummaryWriter(log_path)

    total_batch_size = batch_size * world_size
    lr = lr if lr else base_lr * total_batch_size / 256

    # model, dataloader, optimizer
    model = mae.__dict__[model]()
    if not args.no_to_hfai:
        model = hfai.nn.to_hfai(model, verbose=True)
    DDP = HfaiDDP if args.ddp == 'hfai' else TorchDDP
    model = DDP(model.cuda(), device_ids=[local_rank])
    print(DDP)

    mode = transforms.InterpolationMode.BICUBIC
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=mode),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])  # train transform
    dataset = ImageNet(split="train", transform=train_transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = dataset.loader(batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    param_groups = timm.optim.optim_factory.add_weight_decay(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    scheduler = CosineLRWarmUp(optimizer, warmup_epochs=warmup_epochs, epochs=epochs, lr=lr, min_lr=min_lr)

    ckpt_path = str(save_path / "latest.pt")
    start_epoch, start_step, _ = hfai.checkpoint.init(model, optimizer, ckpt_path=ckpt_path)

    # train, validate
    for epoch in range(start_epoch, epochs):
        # resume from epoch and step
        sampler.set_epoch(epoch)
        dataloader.set_step(start_step)

        train(dataloader, model, optimizer, scheduler, epoch, local_rank, start_step)
        start_step = 0  # reset
        # save
        if rank == 0 and (epoch % 20 == 0 or epoch == epochs - 1):
            state = {"model": model.module.state_dict(), "epoch": epoch}
            torch.save(state, save_path / f"{epoch:04d}.pt")

    if writer:
        writer.close()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)

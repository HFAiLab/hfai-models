import os
import math
import torch
import hfai.distributed as dist


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


def configure_optimizer(model, lr):
    # optimizer
    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    include = lambda n: not exclude(n)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    param_groups = [
        {"params": gain_or_bias_params, "weight_decay": 0.0},
        {"params": rest_params, "weight_decay": 0.1},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.99), eps=1e-8)

    return optimizer


def init_dist(local_rank):
    # init dist
    ip = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    hosts = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)

    return dist.get_rank()

import os
import math
from pathlib import Path
import torch
import torch.distributed as dist
from haiscale.ddp import DistributedDataParallel
from haiscale.pipeline import PipeDream, make_subgroups, partition
from data import ChunkDataset, RandomSampleDataset
from gpt import gpt2_medium, GPTLoss, configure_optimizer

try:
    import hfai
    hfai.distributed.set_nccl_opt_level(hfai.distributed.HFAI_NCCL_OPT_LEVEL.AUTO)
except:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CosineLRWarmUp:
    def __init__(self, optimizer, warmup_epochs, epochs, lr, min_lr=1e-6):
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


def init_dist(local_rank):
    ip = os.getenv('MASTER_ADDR', "127.0.0.1")
    port = os.getenv('MASTER_PORT', '12345')
    nodes = int(os.getenv('WORLD_SIZE', 1))  # number of nodes
    node_rank = int(os.getenv('RANK', 0))    # node rank
    gpus = torch.cuda.device_count()         # number of gpus per node

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=nodes * gpus,
        rank=node_rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size()


@torch.no_grad()
def validate(epoch, dataset, gpt, dp_group, pp_group):
    gpt.eval()
    rank = dist.get_rank()
    is_last_pipeline_stage = (pp_group.rank() == pp_group.size() - 1)

    criterion = GPTLoss()
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(0., device="cuda")
    for i in range(len(dataset)):
        tokens = dataset[i].cuda()
        logits = gpt(tokens)  # [B, L, V]

        if is_last_pipeline_stage:
            loss += criterion(logits, tokens)
            count += 1

    # all-reduce in data paralel group
    if is_last_pipeline_stage:
        dist.all_reduce(loss, group=dp_group)
        dist.all_reduce(count, group=dp_group)
        loss = loss / count
        perp = torch.exp(loss)
    else:
        loss, perp = torch.zeros(2, device="cuda")

    # broadcast from the last stage to other pipeline stages
    dist.all_reduce(loss, group=pp_group)
    dist.all_reduce(perp, group=pp_group)

    if pp_group.rank() == 0:
        print(f"RANK {rank}, epoch {epoch}, validate loss: {loss:.3f}, perplexity {perp:.3f}", flush=True)

    return perp.item()


def main(local_rank):
    rank, world_size = init_dist(local_rank)
    torch.manual_seed(12345)

    # hyperparameters
    pp_size = 8         # pipeline parallel size
    batch_size = 256
    valid_batch_size = 32
    context_size = 256
    vocab_size = 50257  # gpt2-medium
    lr = 1e-3           # for 64 GPUs, global batch size 2048
    max_norm = 1.0
    weight_decay = 0.1
    epochs = 20
    warmup_epochs = 2
    chunks = 32

    # data parallel + pipeline parallel
    dp_group, pp_group = make_subgroups(pp_size=pp_size)
    dp_rank, dp_size = dp_group.rank(), dp_group.size()
    pp_rank, pp_size = pp_group.rank(), pp_group.size()
    is_last_pipeline_stage = (pp_group.rank() == pp_group.size() - 1)
    print(f"RANK {rank}: data parallel {dp_rank}/{dp_size}, pipeline parallel {pp_rank}/{pp_size}", flush=True)

    # save path
    output_dir = Path(f"output/ddp_and_pipeline_8x8")
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"latest.RANK{pp_rank:03d}.pt"

    # model & criterion & optimizer
    gpt = gpt2_medium(vocab_size, context_size).core
    balance = [4, 4, 4, 3, 3, 3, 3, 2]
    gpt = partition(gpt, pp_group.rank(), pp_group.size(), balance=balance)

    gpt = DistributedDataParallel(gpt.cuda(), process_group=dp_group)
    gpt = PipeDream(gpt, chunks, process_group=pp_group)

    criterion = GPTLoss(chunks)
    optimizer = configure_optimizer(gpt, lr=lr, wd=weight_decay)
    scheduler = CosineLRWarmUp(optimizer, warmup_epochs, epochs, lr)

    # dataset
    train_data = "data/wikitext-103/train.npy"
    valid_data = "data/wikitext-103/valid.npy"
    g = torch.Generator()
    g.manual_seed(66229)
    train_dataset = RandomSampleDataset(train_data, context_size, batch_size, g, dp_rank, dp_size)
    valid_dataset = ChunkDataset(valid_data, context_size, valid_batch_size, dp_rank, dp_size)

    # resume from saved checkpoint
    start_epoch, start_step, best_perp = 0, 0, 1e9
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        gpt.module.module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        g.set_state(state["generator"])
        start_epoch, start_step = state["epoch"], state["step"]
        best_perp = state["perp"]

        if local_rank == 0:
            print(f"resume from epoch {start_epoch}, step {start_step}", flush=True)

    # start training
    steps_per_epoch = len(train_dataset)
    validate(start_epoch - 1, valid_dataset, gpt, dp_group, pp_group)

    for epoch in range(start_epoch, epochs):
        gpt.train()
        for step in range(start_step, steps_per_epoch):
            tokens = train_dataset.sample_batch().cuda()
            loss, _ = gpt.forward_backward(tokens, criterion=criterion, labels=(tokens,))
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm)

            lr = scheduler.step(epoch + step / steps_per_epoch)
            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 0 and is_last_pipeline_stage:
                loss = loss.sum()
                mem = torch.cuda.max_memory_allocated() / (1 << 20)
                print(f"RANK {rank}, epoch {epoch}, step {step}/{steps_per_epoch}, lr {lr:.6f} "
                      f"loss {loss:.3f}, peak mem {int(mem)} MiB", flush=True)

        perp = validate(epoch, valid_dataset, gpt, dp_group, pp_group)
        best_perp = min(best_perp, perp)
        start_step = 0

        # save checkpoint
        if dp_rank == 0:
            state = build_state(gpt, optimizer, g, epoch + 1, 0, best_perp)
            safe_save(state, output_dir / f"latest.RANK{pp_rank:03d}.pt")
            if perp == best_perp:
                print(f"New best perplexity: {perp:.3f}", flush=True)
                safe_save(state, output_dir / f"best.RANK{pp_rank:03d}.pt")

    # synchronize all processes
    gpt.module.reducer.stop()
    dist.barrier()


def safe_save(obj, path):
    path = str(path)
    if not Path(path).exists():
        torch.save(obj, path)
        return

    torch.save(obj, path + ".tmp")
    Path(path).rename(path + ".old")
    Path(path + ".tmp").rename(path)
    Path(path + ".old").unlink()


def build_state(gpt, optimizer, g, epoch, step, perp):
    state = {
        "model": gpt.module.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": g.get_state(),
        "epoch": epoch,
        "step": step,
        "perp": perp,
    }

    return state


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, nprocs=ngpus)

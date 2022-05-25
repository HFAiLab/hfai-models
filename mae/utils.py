import os
import torch
import hfai.distributed as dist


def init_dist(local_rank):
    # init dist
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = os.getenv("MASTER_PORT", 1024)
    hosts = int(os.getenv("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.getenv("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size()

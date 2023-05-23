import os
import time
from argparse import ArgumentParser
import torch
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

import hfai
import hfai.distributed as dist
from haiscale.ddp import DistributedDataParallel as HfaiDDP
from torch.nn.parallel import DistributedDataParallel as TorchDDP

from bert import bert_base_MLM


parser = ArgumentParser(description="Pretrain BERT")
parser.add_argument("--ddp", type=str, default='hfai', choices=['torch', 'hfai'])
parser.add_argument("--no_to_hfai", action='store_true')
args = parser.parse_args()


class Trainer:
    def __init__(self, model, lr, warmup, batch_size, epochs, vocab_size, save_path, log_path):
        super().__init__()
        self.steps = 1
        self.model = model
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.save_path = save_path
        self.log_path = log_path
        os.makedirs(save_path, exist_ok=True)

        self.start_step = 0
        self.start_epoch = 0

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr)
        self.scheduler = LambdaLR(self.optimizer, lambda epoch: lr * min(pow(self.steps, -0.5), pow(warmup, -1.5) * self.steps))
        self.rank = dist.get_rank()
        if self.rank == 0:
            self.logger = SummaryWriter(self.log_path)
        self.gpus = dist.get_world_size()

    def pretrain(self, dataset):
        if os.path.exists(os.path.join(self.save_path, "model.pt")):
            self.load()

        self.model.train()
        datasampler = DistributedSampler(dataset)
        dataloader = dataset.loader(self.batch_size, sampler=datasampler, pin_memory=True)
        for epoch in range(self.start_epoch, self.epochs):
            datasampler.set_epoch(epoch)
            dataloader.set_step(self.start_step)

            for step, batch in enumerate(dataloader):
                step += self.start_step

                seq, mask, label = [data.cuda() for data in batch]
                self.optimizer.zero_grad()
                output = self.model(seq, mask.bool())
                loss = self.criterion(output.view(-1, self.vocab_size), label.view(-1))
                loss.backward()
                self.optimizer.step()
                if self.rank % torch.cuda.device_count() == 0 and step % 20 == 0:
                    print(f"Epoch: {epoch}/{self.epochs}, Step: {step}/{len(dataloader)}, Loss: {loss.item()}, Total: {self.steps}", flush=True)
                if self.rank == 0 and self.steps % 10 == 0:
                    self.logger.add_scalar("Loss", loss.item(), self.steps * self.batch_size * self.gpus)
                self.steps += 1
                self.scheduler.step()
                if self.rank == 0 and hfai.client.receive_suspend_command():
                    self.save(epoch, step + 1)
                    hfai.client.go_suspend()

            self.start_step = 0
            if self.rank == 0:
                self.save(epoch + 1, 0)

    def load(self):
        state = torch.load(os.path.join(self.save_path, "model.pt"), map_location="cpu")
        self.model.module.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.steps = state['steps']
        self.start_step = state['step']
        self.start_epoch = state['epoch']
        if dist.get_rank() == 0:
            print(f'Load from epoch {self.start_epoch}, step {self.start_step}', flush=True)

    def save(self, epoch, step):
        state = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'steps': self.steps,
            'epoch': epoch,
            'step': step
        }
        torch.save(state, os.path.join(self.save_path, "model.pt"))
        if dist.get_rank() == 0:
            print(f'save: epoch {epoch}, step {step}', flush=True)


def pretrain(local_rank):
    seed = 12
    vocab_size = 8021
    max_length = 128
    batch_size = 200
    epochs = 10
    lr = 2e-1
    warmup = 10000
    save_path = "output/bert"
    log_path = save_path + "/log"

    torch.manual_seed(seed)

    ip = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    hosts = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    gpus = torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)

    model = bert_base_MLM(vocab_size=vocab_size, max_length=max_length).cuda()

    # accelerate by hfai.nn.to_hfai and hfai DDP
    if not args.no_to_hfai:
        model = hfai.nn.to_hfai(model, verbose=True)
    DDP = HfaiDDP if args.ddp == 'hfai' else TorchDDP
    model = DDP(model, device_ids=[local_rank])

    trainer = Trainer(
        model,
        lr=lr,
        warmup=warmup,
        batch_size=batch_size,
        epochs=epochs,
        vocab_size=vocab_size,
        save_path=save_path,
        log_path=log_path,
    )
    dataset = hfai.datasets.CLUEForMLM()
    trainer.pretrain(dataset)


if __name__ == "__main__":
    hfai.multiprocessing.spawn(pretrain, args=(), nprocs=torch.cuda.device_count(), bind_numa=True)

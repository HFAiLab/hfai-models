import torch
import numpy as np


class RandomSampleDataset():

    def __init__(self, tokens_file, context_size, batch_size, generator, rank=0, world_size=1):
        self.tokens = torch.from_numpy(np.load(tokens_file))
        self.size = len(self.tokens)
        self.context_size = context_size
        self.batch_size = batch_size
        self.g = generator
        self.rank = rank
        self.world_size = world_size

    def __len__(self):
        return self.size // (self.context_size * self.batch_size * self.world_size)

    def sample_batch(self):
        high = self.size - self.context_size
        num_batches = self.batch_size * self.world_size
        start = torch.randint(0, high, size=(num_batches,), generator=self.g)
        start = start.chunk(self.world_size)[self.rank]

        tokens = [self.tokens[s:(s + self.context_size)] for s in start]
        tokens = torch.stack(tokens, dim=0)  # [B, L]

        return tokens.to(torch.int64)


class ChunkDataset():

    def __init__(self, tokens_file, context_size, batch_size, rank=0, world_size=1):
        self.context_size = context_size
        self.batch_size = batch_size

        tokens = torch.from_numpy(np.load(tokens_file))
        chunk_size = context_size * batch_size
        n = tokens.size(0) // chunk_size * chunk_size
        tokens = tokens[:n].view(-1, batch_size, context_size)

        tokens = tokens.chunk(world_size)
        if rank < len(tokens):
            self.tokens = tokens[rank].clone()
        else:
            self.tokens = torch.zeros(0, context_size, batch_size, dtype=torch.int32)

    def __getitem__(self, index):
        return self.tokens[index].to(torch.int64)

    def __len__(self):
        return self.tokens.size(0)

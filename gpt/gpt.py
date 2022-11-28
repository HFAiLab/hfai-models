from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTEmbedding(nn.Module):

    def __init__(self, vocab_size, n_embed, context_size, dropout):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, context_size, n_embed))
        self.drop = nn.Dropout(dropout)
        self.context_size = context_size

        torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.tok_emb(idx)

        assert x.size(1) <= self.context_size
        x = x + self.pos_emb[:, :x.size(1)]  # (N, L+1, n_embed)
        x = self.drop(x)

        return x


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout, context_size) -> None:
        super().__init__()

        self.encoder = nn.TransformerEncoderLayer(
            n_embd,
            n_head,
            n_embd * 4,
            dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        mask = torch.ones(context_size, context_size, dtype=torch.bool)
        mask = ~torch.tril(mask)  # top-left is False, up-right is True
        self.register_buffer("mask", mask)

    def forward(self, x):
        L = x.size(1)
        assert L <= self.mask.size(0)
        mask = self.mask[:L, :L]
        x = self.encoder(x, mask)
        return x


class GPTHead(nn.Module):

    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, x):
        x = self.norm(x)
        logits = self.head(x)
        return logits


class GPTLoss(nn.Module):

    def __init__(self, chunks=1):
        super().__init__()
        self.chunks = chunks
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits, idx):
        V = logits.size(2)
        logits = logits[:, :-1].reshape(-1, V)
        idx = idx[:, 1:].reshape(-1)
        loss = self.criterion(logits, idx)

        loss = loss / self.chunks

        return loss


class GPT(nn.Module):

    def __init__(self, vocab_size, n_layer, n_embed, n_head, context_size=256, dropout=0.1):
        super().__init__()

        embed = GPTEmbedding(vocab_size, n_embed, context_size, dropout)
        blocks = [Block(n_embed, n_head, dropout, context_size) for _ in range(n_layer)]
        head = GPTHead(vocab_size, n_embed)
        layers = [embed] + blocks + [head]

        self.core = nn.Sequential(*layers)

        self.core.context_size = context_size
        self.context_size = context_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        loss = self.core(idx)
        return loss


def gpt2_medium(vocab_size, context_size=256, dropout=0.1):
    return GPT(
        vocab_size=vocab_size,
        n_layer=24,
        n_embed=1024,
        n_head=16,
        context_size=context_size,
        dropout=dropout,
    )


def gpt2_large(vocab_size, context_size=256, dropout=0.1):
    return GPT(
        vocab_size=vocab_size,
        n_layer=36,
        n_embed=1280,
        n_head=20,
        context_size=context_size,
        dropout=dropout,
    )


def configure_optimizer(gpt, lr, wd=0.01, beta1=0.9, beta2=0.95):
    decay = set()
    no_decay = set()
    whitelist = (torch.nn.Linear, torch.nn.MultiheadAttention)
    blacklist = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in gpt.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('pos_emb'):
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in gpt.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    outer_params = param_dict.keys() - union_params
    assert len(inter_params) == 0, inter_params
    assert len(outer_params) == 0, outer_params

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(beta1, beta2))

    return optimizer

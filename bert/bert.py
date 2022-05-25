import torch
import torch.nn as nn
from collections import OrderedDict


class BertForMLM(nn.Module):

    def __init__(
        self,
        vocab_size=21128,
        hidden_size=768,
        feedforward_size=3072,
        max_length=256,
        attn_heads=12,
        hidden_layers=12,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.bert = Bert(vocab_size, hidden_size, feedforward_size, max_length, attn_heads, hidden_layers, dropout_prob)
        self.mlm = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(hidden_size, hidden_size)),
            ("gelu", nn.GELU()),
            ("norm", nn.LayerNorm(hidden_size, eps=1e-12)),
            ("fc2", nn.Linear(hidden_size, vocab_size)),
        ]))

    def forward(self, seq, mask, seg=None):
        return self.mlm(self.bert(seq, mask, seg))


class BertForCLS(nn.Module):

    def __init__(
        self,
        vocab_size=21128,
        hidden_size=768,
        feedforward_size=3072,
        max_length=256,
        attn_heads=12,
        hidden_layers=12,
        dropout_prob=0.1,
        classes=10,
    ):
        super().__init__()
        self.bert = Bert(vocab_size, hidden_size, feedforward_size, max_length, attn_heads, hidden_layers, dropout_prob)
        self.cls = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(hidden_size, hidden_size)),
            ("gelu", nn.GELU()),
            ("norm", nn.LayerNorm(hidden_size, eps=1e-12)),
            ("fc2", nn.Linear(hidden_size, classes)),
        ]))

    def forward(self, seq, mask, seg=None):
        return self.cls(self.bert(seq, mask, seg)[:, 0])


class Bert(nn.Module):
    def __init__(
        self,
        vocab_size=21128,
        hidden_size=768,
        feedforward_size=3072,
        max_length=256,
        attn_heads=12,
        hidden_layers=12,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_length)
        self.encoder = BertEncoder(hidden_size, attn_heads, feedforward_size, hidden_layers, dropout_prob)

    def forward(self, seq, mask=None, seg=None):
        # seq, mask, seg: (batch, seq)
        embed = self.embeddings(seq.T, seg.T if seg is not None else None)  # embed: (seq, batch, feature)
        encode = self.encoder(embed, mask)  # encode: (seq, batch, feature)
        return encode.transpose(0, 1).contiguous()  # output: (batch, seq, feature)


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=21128, hidden_size=768, max_length=256):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position = nn.Embedding(max_length, hidden_size)
        self.segment = nn.Embedding(2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.drop = nn.Dropout()

    def forward(self, seq, seg=None):
        seq_length = seq.size(0)
        pos = torch.arange(seq_length, dtype=torch.long, device=seq.device).unsqueeze(1)
        embed = self.token(seq) + self.position(pos)
        if seg is None:
            seg = torch.zeros((seq_length, 1), dtype=torch.long, device=seq.device)
        embed += self.segment(seg)
        return self.drop(self.norm(embed))


class BertEncoder(nn.Module):
    def __init__(self, hidden_size=768, attn_heads=12, feedforward_size=3072, hidden_layers=12, dropout_prob=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(hidden_size, attn_heads, feedforward_size, dropout_prob, "gelu")
        self.layers = nn.TransformerEncoder(layer, hidden_layers)

    def forward(self, seq, mask=None):
        return self.layers(seq, src_key_padding_mask=mask)


def bert_base_MLM(**kwargs):
    model = BertForMLM(hidden_layers=12, hidden_size=768, attn_heads=12, **kwargs)
    return model


def bert_large_MLM(**kwargs):
    model = BertForMLM(hidden_layers=24, hidden_size=1024, attn_heads=16, **kwargs)
    return model


def bert_base_CLS(**kwargs):
    model = BertForCLS(hidden_layers=12, hidden_size=768, attn_heads=12, **kwargs)
    return model


def bert_large_CLS(**kwargs):
    model = BertForCLS(hidden_layers=24, hidden_size=1024, attn_heads=16, **kwargs)
    return model

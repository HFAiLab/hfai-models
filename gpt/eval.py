import torch
from data import ChunkDataset
from gpt import gpt2_medium, GPTLoss


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def eval(dataset, gpt, tag):
    gpt.eval()
    criterion = GPTLoss()

    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(0., device="cuda")
    for i in range(len(dataset)):
        tokens = dataset[i].cuda()
        with torch.no_grad():
            logits = gpt(tokens)  # [B, L, V]
            loss += criterion(logits, tokens)
            count += 1

    loss = loss / count
    perp = torch.exp(loss).item()

    wordcnt = {"valid": 216347, "test": 244102}
    bpecnt = {"valid": 257382, "test": 291716}
    word_perp = loss * bpecnt[tag] / wordcnt[tag]
    word_perp = torch.exp(word_perp).item()

    print(f"[{tag}] loss: {loss:.3f}, bpe ppl {perp:.3f}, word ppl {word_perp:.3f}", flush=True)

    return perp


def main():
    context_size = 256
    vocab_size = 50257
    batch_size = 32
    gpt = gpt2_medium(vocab_size, context_size).cuda()

    ckpt_path = "output/ddp-node8-cos-lr1e-3/best.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    gpt.load_state_dict(state["model"])

    valid_data = "data/wikitext-103/valid.npy"
    test_data = "data/wikitext-103/test.npy"
    valid_dataset = ChunkDataset(valid_data, context_size, batch_size, 0, 1)
    test_dataset = ChunkDataset(test_data, context_size, batch_size, 0, 1)

    eval(valid_dataset, gpt, "valid")
    eval(test_dataset, gpt, "test")


if __name__ == "__main__":
    main()


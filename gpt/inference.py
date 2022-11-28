from pathlib import Path
from transformers import GPT2TokenizerFast
import torch
import torch.nn.functional as F
from gpt import gpt2_medium


@torch.no_grad()
def sample(gpt, idx, max_new_tokens, temperature=1.0, do_sample=False, topk=None):
    gpt.eval()

    for i in range(max_new_tokens):
        logits = gpt(idx)[:, -1]  # get the last output
        logits /= temperature

        if topk is not None:
            v, _ = torch.topk(logits, topk)
            logits[logits < v[:, [-1]]] = -torch.inf

        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        idx = torch.cat([idx, idx_next], dim=1)

    return idx


def main():
    context_size = 256
    vocab_size = 50257
    gpt = gpt2_medium(vocab_size, context_size).cuda()

    ckpt_path = "output/ddp-node8-cos-lr1e-3/best.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    gpt.load_state_dict(state["model"])

    if Path("data/GPT2Tokenizer/vocab.json").exists():
        tokenizer = GPT2TokenizerFast(
            vocab_file="data/GPT2Tokenizer/vocab.json",
            merges_file="data/GPT2Tokenizer/merges.txt",
            tokenizer_file="data/GPT2Tokenizer/tokenizer.json",
        )
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")

    # sentence = "At the beginning of the film, "
    sentence = "= = = baseball = = ="
    idx = tokenizer.encode(sentence.lower())
    idx = torch.tensor(idx, device="cuda", dtype=torch.int64)
    idx = idx.unsqueeze(0)
    print(idx)

    idx = sample(gpt, idx, 128, do_sample=True, topk=50)
    idx = idx[0].cpu().numpy().tolist()
    words = tokenizer.decode(idx)
    print(words)


if __name__ == "__main__":
    main()

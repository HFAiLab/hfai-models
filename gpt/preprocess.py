from pathlib import Path
import numpy as np
from transformers import GPT2TokenizerFast


"""
Output:

vocab size: 50257
data/wikitext-103/wiki.test.tokens: number of words 244102, number of subwords: 291716
data/wikitext-103/wiki.valid.tokens: number of words 216347, number of subwords: 257382
data/wikitext-103/wiki.train.tokens: number of words 102590700, number of subwords: 121798554
"""


def preprocess():
    if Path("data/GPT2Tokenizer/vocab.json").exists():
        tokenizer = GPT2TokenizerFast(
            vocab_file="data/GPT2Tokenizer/vocab.json",
            merges_file="data/GPT2Tokenizer/merges.txt",
            tokenizer_file="data/GPT2Tokenizer/tokenizer.json",
        )
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")

    print("vocab size:", tokenizer.vocab_size)
    for split in ["test", "valid", "train"]:
        tokens = []
        num_words = 0
        num_subwords = 0
        fname = f"data/wikitext-103/wiki.{split}.tokens"
        with open(fname, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip().lower()

                if len(line) > 0:
                    subwords = tokenizer.encode(line) + [tokenizer.eos_token_id]
                    tokens += subwords
                    num_words += len(line.split(" ")) + 1  # include eos
                    num_subwords += len(subwords)

        tokens = np.array(tokens, dtype=np.int32)
        np.save(f"data/wikitext-103/{split}.npy", tokens)
        print(f"{fname}: number of words {num_words}, number of subwords: {num_subwords}")


if __name__ == "__main__":
    preprocess()

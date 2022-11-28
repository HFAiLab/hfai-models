
# GPT-2

[简体中文](README.md) | English

This is an example of haiscale to train a GPT-2 model on Wikitext-103 dataset.

## Data Preparation

0. Install huggingface transformers:

    ```
    pip install transformers
    ```

1. Download Wikitext-103 and decompress it to `data/wikitext-103`. The directory structure is as follows:

    ```
    data/wikitext-103
    ├── wiki.test.tokens
    ├── wiki.train.tokens
    └── wiki.valid.tokens
    ```

2. Run the preprocess script to tokenize the raw text with BPE:

    ```
    python preprocess.py
    ```

    Now the directory structure is as follows:

    ```
    data/wikitext-103
    ├── test.npy
    ├── train.npy
    ├── valid.npy
    ├── wiki.test.tokens
    ├── wiki.train.tokens
    └── wiki.valid.tokens
    ```

## Training

This example includes multiple training scripts. Each script uses a different parallelism method:

1. data parallel

    ```
    python train_ddp.py
    ```

2. fully sharded data parallel

    ```
    python train_fsdp.py
    ```

3. pipeline parallel

    ```
    python train_pipeline.py
    ```

4. data parallel + pipeline parallel

    ```
    python train_ddp_pipeline.py
    ```

## Evaluation

```
python eval.py
```

The evaluation results on valid and test set are as follows:

| split    | BPE PPL  | word PPL  |
|----------|----------|-----------|
| valid    | 16.222   | 27.519    |
| test     | 16.608   | 28.731    |


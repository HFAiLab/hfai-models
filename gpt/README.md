
# GPT-2

简体中文 | [English](README_en.md)

这是一个在 Wikitext-103 上使用 haiscale 训练 GPT2 模型的示例。

## Data Preparation

0. 安装 haiscale：

    ```
    pip install haiscale --extra-index-url https://pypi.hfai.high-flyer.cn/simple --trusted-host pypi.hfai.high-flyer.cn
    ```

1. 下载 Wikitext-103 并解压到 `data/wikitext-103`，解压后目录结构如下：

    ```
    data/wikitext-103
    ├── wiki.test.tokens
    ├── wiki.train.tokens
    └── wiki.valid.tokens
    ```

2. 运行数据处理脚本，对文本做 BPE 分词处理：

    ```
    python preprocess.py
    ```

    处理后目录结构如下：

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

本示例中包含了多个训练脚本，分别对应着不同的分布式并行训练方法:

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

在验证集和测试集上的评估结果如下：

| split    | BPE PPL  | word PPL  |
|----------|----------|-----------|
| valid    | 16.222   | 27.519    |
| test     | 16.608   | 28.731    |


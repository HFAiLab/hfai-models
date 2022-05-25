## BERT

本项目实现了论文 《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》

![BERT](assets/BERT.jpg)

## Requirements

- torch >= 1.8
- hfai (to be released soon)
- [ffrecord](https://github.com/HFAiLab/ffrecord/)

## hfai.nn.to_hfai & hfai DDP

我们比较了 100 个预训练 step 所消耗的时间：

| to_hfai  | hfai ddp  | time   | performance  |
| -------- | --------- | ------ | ------------ |
| No       | No        | 47.39  | 100%         |
| Yes      | No        | 42.56  | 111%         |
| No       | Yes       | 43.08  | 110%         |
| Yes      | Yes       | 37.86  | 125%         |

我们可以看到，使用了 `hfai.nn.to_hfai` 和 `hfai DDP` 后训练速度能够提升 25 %。

## Pretrain

在萤火二号集群上运行：

```shell
hfai python pretrain.py -- -n 8 -p 40
```

本地运行：

```shell
python pretrain.py
```

## Finetune

```shell
python finetune.py
```

## Result

| model       | data size   | AFQMC       | TNEWS       | IFLYTEK     | CMNLI       |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| BERT-base   | 14 G        | 70.27%      | 56.23%      | 60.18%      | 76.60%      |

## Citation

```
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

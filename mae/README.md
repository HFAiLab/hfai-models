## Masked Autoencoders Are Scalable Vision Learners

本项目在萤火二号集群上实现了论文《Masked Autoencoders Are Scalable Vision Learners》。

![mae](assets/MAE.png)

## Requirements

- torch >= 1.8
- hfai (to be released soon)
- [ffrecord](https://github.com/HFAiLab/ffrecord/)


## Pretrain

在萤火二号集群上运行：

```shell
hfai python pretrain.py -- -n 8 -p 30
```

本地运行：

```shell
python pretrain.py
```

## Finetune

在萤火二号集群上运行：

```shell
hfai python finetune.py -- -n 8 -p 30
```

本地运行：

```shell
python finetune.py
```

## Accuracy

以下是我们训练得到的结果：

| model | vit_base_patch16 | vit_large_patch16 | vit_huge_patch14 |
|-------|------------------|-------------------|------------------|
| Acc@1 | 83.43%           | 85.71%            | 86.14%           |
| Acc@5 | 96.53%           | 97.63%            | 97.74%           |


## hfai DDP & hfai.nn.to_hfai

我们比较在 8 个计算节点，64 块 GPU 上预训练 100 个 step 所消耗的时间，用的模型是 `mae_vit_huge_patch14`，batch size 设为 64

| to_hfai  | hfai ddp  | time (s)  | performance  |
| -------- | --------- | --------- | ------------ |
| No       | No        | 147.55    | 100%         |
| Yes      | No        | 145.03    | 102%         |
| No       | Yes       | 99.05     | 149%         |
| Yes      | Yes       | 97.67     | 151%         |

我们可以看到，使用了 `hfai DDP` 和 `hfai.nn.to_hfai` 后训练速度有显著提升，相比于原来能够提升 51 %。


## References

- https://github.com/facebookresearch/mae


## Citation

```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

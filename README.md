# 幻方 AI 模型仓库

这里是幻方 AI 的模型仓库，包含了从计算机视觉、自然语言处理到生物计算、气象预测等各个领域方面的模型，这些模型结合了幻方萤火超算集群的特点，使用并行训练、高效算子、高性能存储等多种方式，大幅提升原有模型的性能，节省训练时间。


## 依赖

- torch >= 1.8
- [hfai](https://doc.hfai.high-flyer.cn/index.html)
- [ffrecord](https://github.com/HFAiLab/ffrecord/)

## 模型列表

Tags:

- `hfai.nn.to_hfai`: 是否使用了算子加速
- `hfai DDP`: 是否使用了 HFAI DDP 加速
- `hfai.datasets`: 是否使用了 HFAI 数据仓库


1. 生物计算

    | name                                                         | maintainer                                | hfai.nn.to_hfai  | hfai DDP  | hfai.datasets  |
    |--------------------------------------------------------------|-------------------------------------------|------------------|-----------|----------------|
    | [AlphaFold2](https://github.com/HFAiLab/alphafold-optimized) | [@mingchuan](https://github.com/Revnize)  | Yes              | Yes       | No             |


2. 自动驾驶

    | name                                              | maintainer                                | hfai.nn.to_hfai  | hfai DDP  | hfai.datasets  |
    |---------------------------------------------------|-------------------------------------------|------------------|-----------|----------------|
    | [HDMapNet](https://github.com/HFAiLab/hdmapnet)   | [@bixiao](https://github.com/Freja71122)  | No               | No        | Yes            |
    | [BEVFormer](https://github.com/HFAiLab/BEVFormer) | [@bixiao](https://github.com/Freja71122)  | Yes              | No        | Yes            |


3. 时间序列

    | name                                                      | maintainer                             | hfai.nn.to_hfai  | hfai DDP  | hfai.datasets  |
    |-----------------------------------------------------------|----------------------------------------|------------------|-----------|----------------|
    | [LTSF-formers](https://github.com/HFAiLab/LTSF-formers)   | [@wenjie](https://github.com/VachelHU) | No               | No        | Yes            |


4. 多模态

    | name                                             | maintainer                                    | hfai.nn.to_hfai  | hfai DDP  | hfai.datasets  |
    |--------------------------------------------------|-----------------------------------------------|------------------|-----------|----------------|
    | [CLIP](clip)                                     | [@chengqi](https://github.com/KinglittleQ)    | Yes              | Yes       | Yes            |
    | [CLIP-GEN](https://github.com/HFAiLab/clip-gen)  | [@chengqi](https://github.com/KinglittleQ)    | No               | Yes       | Yes            |


5. 计算机视觉

    | name                     | maintainer                                    | hfai.nn.to_hfai  | hfai DDP  | hfai.datasets  |
    |--------------------------|-----------------------------------------------|------------------|-----------|----------------|
    | [ResNet50](resnet)       | @hfai                                         | No               | No        | Yes            |
    | [MaskedAutoEncoder](mae) | [@chengqi](https://github.com/KinglittleQ)    | Yes              | Yes       | Yes            |


6. 自然语言处理

    | name         | maintainer                                    | hfai.nn.to_hfai  | hfai DDP  | hfai.datasets  |
    |--------------|-----------------------------------------------|------------------|-----------|----------------|
    | [BERT](bert) | @hfai                                         | Yes              | Yes       | Yes            |

7. 气象预测

    | name                                                   | maintainer                             | hfai.nn.to_hfai  | hfai DDP  | hfai.datasets  |
    |--------------------------------------------------------|----------------------------------------|------------------|-----------|----------------|
    | [FourCastNet](https://github.com/HFAiLab/FourCastNet)  | [@wenjie](https://github.com/VachelHU) | Yes              | No        | Yes            |

8. 图神经网络

    | name                                                         | maintainer                             | hfai.nn.to_hfai  | hfai DDP  | hfai.datasets  |
    |--------------------------------------------------------------|----------------------------------------|------------------|-----------|----------------|
    | [DeepGCNs](https://github.com/HFAiLab/Distributed-DeepGCNs)  | [@wenjie](https://github.com/VachelHU) | Yes              | Yes       | Yes            |


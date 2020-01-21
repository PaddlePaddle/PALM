# PaddlePALM

PaddlePALM (Paddle for Multi-task) 是一个灵活通用且易用的NLP大规模预训练与多任务学习框架。通过PaddlePALM，用户可以轻松完成复杂的多任务学习与参数复用，无缝集成「**单任务训练**」、「**多任务辅助训练**」和「**多目标任务联合训练**」这 *3* 种训练方式和灵活的保存与预测机制，且仅需书写极少量代码即可”一键启动”高性能单机单卡和分布式训练与推理。

框架中内置了丰富的[主干网络](#附录b内置主干网络backbone)及其[预训练模型](#预训练模型)（BERT、ERNIE等）、常见的[任务范式](#附录c内置任务范式paradigm)（分类、匹配、机器阅读理解等）和相应的[数据集读取与处理工具](#附录a内置数据集载入与处理工具reader)。同时框架提供了用户自定义接口，若内置工具、主干网络和任务无法满足需求，开发者可以轻松完成相关组件的自定义。各个组件均为零耦合设计，用户仅需完成组件本身的特性开发即可完成与框架的融合。

PaddlePALM (PArallel Learning from Multi-tasks) is a flexible, general and easy-to-use NLP large-scale pretraining and multi-task learning friendly framework. PALM is a high level framework aiming at **fastly** develop **high-performance** NLP models. With PALM, a typical NLP task can be achieved just in 8 steps. 
s   

然后给出一些成功案例和一些公开数据集的各个backbone的实验结果（BERT、ERNIE、RoBERTa）和一些成功的多任务学习示例。

## 目录

- [安装](#安装)
- [前期准备](#前期准备)
    - [理论准备](#理论准备)
    - [框架原理](#框架原理)
    - [预训练模型](#预训练模型)
- [X行代码实现文本分类](#三个demo入门paddlepalm)
    - 
- []
    - [DEMO1：单任务训练](#demo1单任务训练)
    - [DEMO2：多任务辅助训练与目标任务预测](#demo2多任务辅助训练与目标任务预测)
    - [DEMO3：多目标任务联合训练与任务层参数复用](#demo3多目标任务联合训练与任务层参数复用)
- [进阶篇](#进阶篇)
    - [配置广播机制](#配置广播机制)
    - [reader、backbone与paradigm的选择](#readerbackbone与paradigm的选择)
    - [多目标任务下的训练终止条件与预期训练步数](#多目标任务下的训练终止条件与预期训练步数)
        - [多个目标任务](#多个目标任务)
        - [训练终止条件](#训练终止条件)
        - [任务采样概率与预期训练步数](#任务采样概率与预期训练步数)
        - [多个目标任务时预期训练步数的计算](#多个目标任务时预期训练步数的计算)
    - [模型保存与预测机制](#模型保存与预测机制)
    - [分布式训练](#分布式训练)
- [附录A：内置数据集载入与处理工具（reader）](#附录a内置数据集载入与处理工具reader)
- [附录B：内置主干网络（backbone）](#附录b内置主干网络backbone)
- [附录C：内置任务范式（paradigm）](#附录c内置任务范式paradigm)
- [附录D：可配置的全局参数列表](#附录d可配置的全局参数列表)


## Package Overview

| **paddlepalm** | an open source NLP pretraining and multitask learning framework, built on paddlepaddle. |
| **paddlepalm.reader** | a collection of elastic task-specific dataset readers. |
| **paddlepalm.backbone** | a collection of classic NLP representation models, e.g., BERT. |
| **paddlepalm.head** | a collection of task-specific output layers. |
| **paddlepalm.lr_sched** | a collection of learning rate schedualers. |
| **paddlepalm.optimizer** | a collection of optimizers. |
| **paddlepalm.downloader** | a download module for pretrained models with configure and vocab files. |
| **paddlepalm.Trainer** | the core unit to start a single task training/predicting session. A trainer is to build computation graph, manage training and evaluation process, achieve model/checkpoint saving and pretrain_model/checkpoint loading.|
| **paddlepalm.MultiHeadTrainer** | the core unit to start a multi-task training/predicting session. A MultiHeadTrainer is built based on several Trainers. Beyond the inheritance of Trainer, it additionally achieves model backbone reuse across tasks, trainer sampling for multi-task learning, and multi-head inference for effective evaluation and prediction. |


## Installation

PaddlePALM support both python2 and python3, linux and windows, CPU and GPU. The preferred way to install PaddlePALM is via `pip`. Just run following commands in your shell environment.

```bash
pip install paddlepalm
```

### Installing via source

```shell
git clone https://github.com/PaddlePaddle/PALM.git
cd PALM && python setup.py install
```

### Library Dependencies
- Python >= 2.7 (即将支持python3)
- cuda >= 9.0
- cudnn >= 7.0
- PaddlePaddle >= 1.6.3 (请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装)


### Downloading pretrain models
We incorporate many pretrained models to initialize model backbone parameters. Training big NLP model, e.g., 12-layer transformers, with pretrained models is practically much more effective than that with randomly initialized parameters. To see all the available pretrained models and download, run following code in python interpreter (input command `python` in shell):

```python
>>> from paddlepalm import downloader
>>> downloader.ls('pretrain')
Available pretrain items:
  => roberta-cn-base
  => roberta-cn-large
  => bert-cn-base
  => bert-cn-large
  => bert-en-uncased-base
  => bert-en-uncased-large
  => bert-en-cased-base
  => bert-en-cased-large
  => ernie-en-uncased-base
  => ernie-en-uncased-large
  ...

>>> downloader.download('pretrain', 'bert-en-uncased-base', './pretrain_models')
...
```


## Usage

8 steps to start a typical NLP training task.

1. use `paddlepalm.reader` to create a *reader* for dataset loading and input features generation, then call `reader.load_data` method to load your training data.
2. use `paddlepalm.backbone` to create a model *backbone* to extract text features (e.g., contextual word embedding, sentence embedding).
3. register your *reader* with your *backbone* through `reader.register_with` method. After this step, your reader is able to yield input features used by backbone.
4. use `paddlepalm.head` to create a task output *head*. This head can provide task loss for training and predicting results for model inference.
5. create a task *trainer* with `paddlepalm.Trainer`, then build forward graph with backbone and task head (created in step 2 and 4) through `trainer.build_forward`.
6. use `paddlepalm.optimizer` (and `paddlepalm.lr_sched` if is necessary) to create a *optimizer*, then build backward through `trainer.build_backward`.
7. fit prepared reader and data (achieved in step 1) to trainer with `trainer.fit_reader` method.
8. randomly initialize model parameters (and `trainer.load_pretrain` if needed), then do training with `trainer.train`.

More implementation details see following demos: [Sentiment Classification](), [Quora Question Pairs matching](), [Tagging](), [SQuAD machine Reading Comprehension]().

To save models/checkpoints during training, just call `trainer.set_saver` method. More implementation details see [this]().

To do predict/evaluation after a training stage, just create another three reader, backbone and head instance with `phase='predict'` (repeat step 1~4 above). Then do predicting with `predict` method in trainer (no need to create another trainer). More implementation details see [this]().

To run with multi-task learning mode:

1. repeatedly create components (i.e., reader, backbone and head) for each task followed with step 1~5 above. 
2. create empty trainers (each trainer is corresponded to one task) and pass them to create a `MultiHeadTrainer`. 
3. build multi-task forward graph with `multi_head_trainer.build_forward` method.
4. use `paddlepalm.optimizer` (and `paddlepalm.lr_sched` if is necessary) to create a *optimizer*, then build backward through `multi_head_trainer.build_backward`.
5. fit all prepared readers and data to multi_head_trainer with `multi_head_trainer.fit_readers` method.
6. randomly initialize model parameters with `multi_head_trainer.random_init_params` (and `multi_head_trainer.load_pretrain` if needed), then do training with `multi_head_trainer.train`.

The save/load and predict operations of a multi_head_trainer is the same as a trainer.

More implementation details of running multi-task learning with multi_head_trainer can be found [here]().


## License

This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE).

## 许可证书

此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE)许可认证。


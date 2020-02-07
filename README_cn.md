# PaddlePALM

[English](./README.md) | 简体中文
PaddlePALM (PArallel Learning from Multi-tasks) 是一个灵活，通用且易于使用的NLP大规模预训练和多任务学习框架。 PALM是一个旨在**快速开发高性能NLP模型**的上层框架。

使用PaddlePALM，可以非常轻松灵活的探索具有多种任务辅助训练的“高鲁棒性”阅读理解模型，基于PALM训练的模型[D-Net](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/MRQA2019-D-NET)在[EMNLP2019国际阅读理解评测](mrqa .github.io)中夺得冠军。

<p align="center">
	<img src="https://tva1.sinaimg.cn/large/006tNbRwly1gbjkuuwrmlj30hs0hzdh2.jpg" alt="Sample"  width="300" height="333">
	<p align="center">
		<em>MRQA2019 Leaderboard</em>
	</p>
</p>

除了降低NLP研究成本以外，PaddlePALM已被应用于“百度搜索引擎”，有效的提高了用户查询的理解准确度和挖掘出的答案质量，具备高可靠性和高训练/推理性能。

#### 特点:

- **Easy-to-use:** with PALM, *8 steps* to achieve a typical NLP task. Moreover, the model backbone, dataset reader and task output layers have been decoupled, which allows the replacement of any component to other candidates with quite minor changes of your code. 
- **Multi-task Learning friendly:** *6 steps* to achieve multi-task learning for prepared tasks. 
- **Large Scale and Pre-training freiendly:** automatically utilize multi-gpus (if exists) to accelerate training and inference. Minor codes is required for distributed training on clusters.
- **Popular NLP Backbones and Pre-trained models:** multiple state-of-the-art general purpose model architectures and pretrained models (e.g., BERT,ERNIE,RoBERTa,...) are built-in. 
- **Easy to Customize:** support customized development of any component (e.g, backbone, task head, reader and optimizer) with reusement of pre-defined ones, which gives developers high flexibility and effeciency to adapt for diverse NLP scenes. 

You can easily re-produce following competitive results with minor codes, which covers most of NLP tasks such as classification, matching, sequence labeling, reading comprehension, dialogue understanding and so on. More details can be found in `examples`.

<table>
  <tbody>
    <tr>
      <th><strong>Dataset</strong>
        <br></th>
      <th colspan="3"><center><strong>chnsenticorp</strong></center></th>
      <th colspan="3"><center><strong>Quora Question Pairs matching</strong><center></th>
      <th colspan="3"><strong>MSRA-NER<br>(SIGHAN2006)</strong></th>
      <th colspan="2"><strong>CMRC2018</strong></th>
    </tr>
    <tr>
      <td rowspan="2">
        <p>
          <strong>Metric</strong>
          <br></p>
      </td>
      <td colspan="1">
        <center><strong>precision</strong></center>
        <br></td>
      <td colspan="1">
        <strong>recall</strong>
        <br></td>
      <td colspan="1">
        <strong>f1-score</strong>
        <strong></strong>
        <br></td>
      <td colspan="1">
        <center><strong>precision</strong></center>
        <br></td>
      <td colspan="1">
        <strong>recall</strong>
        <br></td>
      <td colspan="1">
        <strong>f1-score</strong>
        <strong></strong>
        <br></td>
      <td colspan="1">
        <center><strong>precision</strong></center>
        <br></td>
      <td colspan="1">
        <strong>recall</strong>
        <br></td>
      <td colspan="1">
        <strong>f1-score</strong>
        <strong></strong>
        <br></td>
      <td colspan="1">
        <strong>em</strong>
        <br></td>
      <td colspan="1">
        <strong>f1-score</strong>
        <br></td>
    </tr>
    <tr>
      <td colspan="3" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="3" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="3" width="">
        <strong>test</strong>
        <br></td>
      <td colspan="2" width="">
        <strong>dev</strong>
        <br></td>
    </tr>
    <tr>
      <td><strong>ERNIE Base</strong></td>
      <td>95.7</td>
      <td>95.0</td>
      <td>95.7</td>
      <td>85.8</td>
      <td>82.4</td>
      <td>81.5</td>
      <td>94.9</td>
      <td>94.5</td>
      <td>94.7</td>
      <td>96.3</td>
      <td>84.0</td>
    </tr>

  </tbody>
</table>



## Package Overview

| module | illustration | 
| - | - |
| **paddlepalm** | an open source NLP pretraining and multitask learning framework, built on paddlepaddle. |
| **paddlepalm.reader** | a collection of elastic task-specific dataset readers. |
| **paddlepalm.backbone** | a collection of classic NLP representation models, e.g., BERT, ERNIE, RoBERTa. |
| **paddlepalm.head** | a collection of task-specific output layers. |
| **paddlepalm.lr_sched** | a collection of learning rate schedualers. |
| **paddlepalm.optimizer** | a collection of optimizers. |
| **paddlepalm.downloader** | a download module for pretrained models with configure and vocab files. |
| **paddlepalm.Trainer** | the core unit to start a single task training/predicting session. A trainer is to build computation graph, manage training and evaluation process, achieve model/checkpoint saving and pretrain_model/checkpoint loading.|
| **paddlepalm.MultiHeadTrainer** | the core unit to start a multi-task training/predicting session. A MultiHeadTrainer is built based on several Trainers. Beyond the inheritance of Trainer, it additionally achieves model backbone reuse across tasks, trainer sampling for multi-task learning, and multi-head inference for effective evaluation and prediction. |


## Installation

PaddlePALM support both python2 and python3, linux and windows, CPU and GPU. The preferred way to install PaddlePALM is via `pip`. Just run following commands in your shell.

```bash
pip install paddlepalm
```

### Installing via source

```shell
git clone https://github.com/PaddlePaddle/PALM.git
cd PALM && python setup.py install
```

### Library Dependencies
- Python >= 2.7
- cuda >= 9.0
- cudnn >= 7.0
- PaddlePaddle >= 1.7.0 (请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装)


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
8. load pretrain model with `trainer.load_pretrain`, or load checkpoint with `trainer.load_ckpt` or nothing to do for training from scratch, then do training with `trainer.train`.

For more implementation details, see following demos: 

- [Sentiment Classification](https://github.com/PaddlePaddle/PALM/tree/master/examples/classification)
- [Quora Question Pairs matching](https://github.com/PaddlePaddle/PALM/tree/master/examples/matching)
- [Tagging](https://github.com/PaddlePaddle/PALM/tree/master/examples/tagging)
- [SQuAD machine Reading Comprehension](https://github.com/PaddlePaddle/PALM/tree/master/examples/mrc).

### set saver

To save models/checkpoints and logs during training, just call `trainer.set_saver` method. More implementation details see [this](https://github.com/PaddlePaddle/PALM/tree/master/examples).

### do prediction
To do predict/evaluation after a training stage, just create another three reader, backbone and head instance with `phase='predict'` (repeat step 1~4 above). Then do predicting with `predict` method in trainer (no need to create another trainer). More implementation details see [this](https://github.com/PaddlePaddle/PALM/tree/master/examples/predict).

### multi-task learning
To run with multi-task learning mode:

1. repeatedly create components (i.e., reader, backbone and head) for each task followed with step 1~5 above. 
2. create empty trainers (each trainer is corresponded to one task) and pass them to create a `MultiHeadTrainer`. 
3. build multi-task forward graph with `multi_head_trainer.build_forward` method.
4. use `paddlepalm.optimizer` (and `paddlepalm.lr_sched` if is necessary) to create a *optimizer*, then build backward through `multi_head_trainer.build_backward`.
5. fit all prepared readers and data to multi_head_trainer with `multi_head_trainer.fit_readers` method.
6. randomly initialize model parameters with `multi_head_trainer.random_init_params` (and `multi_head_trainer.load_pretrain` if needed), then do training with `multi_head_trainer.train`.

The save/load and predict operations of a multi_head_trainer is the same as a trainer.

For more implementation details with `multi_head_trainer`, see

- [ATIS: joint training of dialogue intent recognition and slot filling](https://github.com/PaddlePaddle/PALM/tree/master/examples/multi-task)
- [MRQA: learning reading comprehension auxilarized with mask language model]() (初次发版先不用加)


## License

This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE).

## 许可证书

此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE)许可认证。

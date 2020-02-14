# PaddlePALM

[English](./README.md) | 简体中文

PaddlePALM (PArallel Learning from Multi-tasks) 是一个灵活，通用且易于使用的NLP大规模预训练和多任务学习框架。 PALM是一个旨在**快速开发高性能NLP模型**的上层框架。

使用PaddlePALM，可以非常轻松灵活的探索具有多种任务辅助训练的“高鲁棒性”阅读理解模型，基于PALM训练的模型[D-Net](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/MRQA2019-D-NET)在[EMNLP2019国际阅读理解评测](mrqa .github.io)中夺得冠军。

<p align="center">
	<img src="https://tva1.sinaimg.cn/large/006tNbRwly1gbjkuuwrmlj30hs0hzdh2.jpg" alt="Sample"  width="300" height="333">
	<p align="center">
		<em>MRQA2019 排行榜</em>
	</p>
</p>

除了降低NLP研究成本以外，PaddlePALM已被应用于“百度搜索引擎”，有效的提高了用户查询的理解准确度和挖掘出的答案质量，具备高可靠性和高训练/推理性能。

#### 特点:

- **易于使用**：使用PALM， *8个步骤*即可实现一个典型的NLP任务。此外，模型主干网络、数据集读取工具和任务输出层已经解耦，只需对代码进行相当小的更改，就可以将任何组件替换为其他候选组件。
- **支持多任务学习**：*6个步骤*即可实现多任务学习任务。
- **支持大规模任务和预训练**：可自动利用多gpu加速训练和推理。集群上的分布式训练需要较少代码。
- **流行的NLP骨架和预训练模型**：内置多种最先进的通用模型架构和预训练模型(如BERT、ERNIE、RoBERTa等)。
- **易于定制**：支持任何组件的定制开发(e。g，主干网络，任务头，读取工具和优化器)与预定义组件的复用，这给了开发人员高度的灵活性和效率，以适应不同的NLP场景。

你可以很容易地用较小的代码重新得到很好的结果，涵盖了大多数NLP任务，如分类、匹配、序列标记、阅读理解、对话理解等等。更多细节可以在`examples`中找到。

<table>
  <tbody>
    <tr>
      <th><strong>数据集</strong>
        <br></th>
      <th colspan="3"><center><strong>chnsenticorp</strong></center></th>
      <th colspan="3"><center><strong>Quora Question Pairs matching</strong><center></th>
      <th colspan="3"><strong>MSRA-NER<br>(SIGHAN2006)</strong></th>
      <th colspan="2"><strong>CMRC2018</strong></th>
    </tr>
    <tr>
      <td rowspan="2">
        <p>
          <strong>评价标准</strong>
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



## Package概览

<p align="center">
	<img src="https://github.com/PaddlePaddle/PALM/blob/master/img/architecture.png" alt="Sample"  width="600px" height="auto">
	<p align="center">
		<em>PALM架构图</em>
	</p>
</p>

| 模块 | 描述 | 
| - | - |
| **paddlepalm** | 一个开源的NLP预训练和多任务学习框架，建立在paddlepaddle框架上。 |
| **paddlepalm.reader** | 特定于任务的数据集读取工具的集合。|
| **paddlepalm.backbone** | 一系列经典的NLP表示模型，如BERT, ERNIE, RoBERTa。|
| **paddlepalm.head** | 任务特定输出层的集合。|
| **paddlepalm.lr_sched** | 一个学习率时间表的集合。|
| **paddlepalm.optimizer** | 优化器的集合。|
| **paddlepalm.downloader** | 预训练模型与配置和vocab文件的下载模块。|
| **paddlepalm.Trainer** | 单一任务训练/预测。一个训练器是建立计算图，管理训练和评估过程，实现模型/检查点保存和pretrain_model/检查点加载。|
| **paddlepalm.MuiliHeadTrainer** | 核心单位开始多任务训练/预测会议。一个多教练是建立在几个Trainer的基础上。在继承Trainer的基础上，实现了模型主干网络跨任务复用，训练器采用多任务学习，多任务推理，来保证更有效的评估和预测。|

## 安装

PaddlePALM 支持 python2 和 python3, linux 和 windows, CPU 和 GPU。安装PaddlePALM的首选方法是通过`pip`。只需运行以下命令：

```bash
pip install paddlepalm
```

### 通过源码安装

```shell
git clone https://github.com/PaddlePaddle/PALM.git
cd PALM && python setup.py install
```

### 库依赖
- Python >= 2.7
- cuda >= 9.0
- cudnn >= 7.0
- PaddlePaddle >= 1.7.0 (请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装)


### 下载预训练模型
我们合并了许多预训练的模型来初始化模型主干网络参数。用预先训练好的模型训练大的NLP模型，如12层Transformer，实际上比用随机初始化的参数更有效。要查看所有可用的预训练模型并下载，请在python解释器中运行以下代码(在shell中输入命令`python`):

```python
>>> from paddlepalm import downloader
>>> downloader.ls('pretrain')
Available pretrain items:
  => RoBERTa-zh-base
  => RoBERTa-zh-large
  => ERNIE-v2-en-base
  => ERNIE-v2-en-large
  => XLNet-cased-base
  => XLNet-cased-large
  => ERNIE-v1-zh-base
  => ERNIE-v1-zh-base-max-len-512
  => BERT-en-uncased-large-whole-word-masking
  => BERT-en-cased-large-whole-word-masking
  => BERT-en-uncased-base
  => BERT-en-uncased-large
  => BERT-en-cased-base
  => BERT-en-cased-large
  => BERT-multilingual-uncased-base
  => BERT-multilingual-cased-base
  => BERT-zh-base

>>> downloader.download('pretrain', 'bert-en-uncased-base', './pretrain_models')
...
```


## 使用

8个步骤开始一个典型的NLP训练任务。

1. 使用`paddlepalm.reader` 要为数据集加载和输入特征生成创建一个`reader`，然后调用`reader.load_data`方法加载训练数据。
2. 使用`paddlepalm.load_data`创建一个模型*主干网络*来提取文本特征(例如，上下文单词嵌入，句子嵌入)。
3. 通过`reader.register_with`将`reader`注册到主干网络上。在这一步之后，reader能够使用主干网络产生的输入特征。
4. 使用`paddlepalm.head`。创建一个任务输出*head*。该头可以为训练提供任务损失，为模型推理提供预测结果。
5. 使用`paddlepalm.Trainer`创建一个任务`Trainer`，然后通过`Trainer.build_forward`构建包含主干网络和任务头的前向图(在步骤2和步骤4中创建)。
6. 使用`paddlepalm.optimizer`（如果需要，创建`paddlepalm.lr_sched`）来创建一个*优化器*，然后通过`train.build_back`向后构建。
7. 使用`trainer.fit_reader`将准备好的reader和数据（在步骤1中实现）给到trainer。
8. 使用`trainer.load_pretrain`加载预训练模型或使用 `trainer.load_pretrain`加载checkpoint，或不加载任何已训练好的参数，然后使用`trainer.train`进行训练。

更多实现细节请见示例: 

- [Sentiment Classification](https://github.com/PaddlePaddle/PALM/tree/master/examples/classification)
- [Quora Question Pairs matching](https://github.com/PaddlePaddle/PALM/tree/master/examples/matching)
- [Tagging](https://github.com/PaddlePaddle/PALM/tree/master/examples/tagging)
- [SQuAD machine Reading Comprehension](https://github.com/PaddlePaddle/PALM/tree/master/examples/mrc).

### 设置saver

在训练时保存 models/checkpoints 和 logs， 调用 `trainer.set_saver` 方法. 更多实现细节见[这里](https://github.com/PaddlePaddle/PALM/tree/master/examples).

### 预测
训练结束后进行预测和评价, 只需创建额外的reader, backbone和head示例（重复上面1~4步骤），注意创建时需设`phase='predict'`。 然后使用trainer的`predict`方法进行预测（不需创建额外的trainer）。更多实现细节请见[这里](https://github.com/PaddlePaddle/PALM/tree/master/examples/predict).

### 多任务学习

多任务学习模式下运行:

1. 重复创建组件（每个任务按照上述第1~5步执行）。
2. 创建空的训练器(每个训练器对应一个任务)，并通过它们创建一个`MultiHeadTrainer`。
3. 使用`multi_head_trainer.build_forward`构建多任务前向图。
4. 使用`paddlepalm.optimizer`（如果需要，创建`paddlepalm.lr_sched`）来创建一个*optimizer*，然后通过` multi_head_trainer.build_backward`创建反向。
5. 使用`multi_head_trainer.fit_readers`将所有准备好的读取器和数据放入`multi_head_trainer`中。
6. 使用`multi_head_trainer.load_pretrain`加载预训练模型或使用 `multi_head_trainer.load_pretrain`加载checkpoint，或不加载任何已经训练好的参数，然后使用`multi_head_trainer.train`进行训练。

multi_head_trainer的保存/加载和预测操作与trainer相同。


更多实现`multi_head_trainer`的细节, 请见

- [ATIS: joint training of dialogue intent recognition and slot filling](https://github.com/PaddlePaddle/PALM/tree/master/examples/multi-task)
- [MRQA: learning reading comprehension auxilarized with mask language model]() (初次发版先不用加)



## 许可证书

此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE)许可认证。

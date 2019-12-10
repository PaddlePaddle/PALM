# PaddlePALM

PaddlePALM (Paddle for Multi-task) 是一个强大快速、灵活易用的 ***任务级（task level）*** 大规模多任务学习框架。框架实现了简洁易懂的任务实例创建与管理机制，强大易用的参数管理与复用接口，以及高效稳定的多任务调度算法。通过PaddlePALM，用户可以轻松完成复杂的多任务学习与参数复用，无缝集成「**单任务训练**」、「**多任务辅助训练**」和「**多目标任务联合训练**」这 *3* 种训练方式和灵活的保存与预测机制，且仅需书写极少量代码即可”一键启动”高性能单机单卡和分布式训练与推理。

框架中内置了丰富的[主干网络](#附录b内置主干网络backbone)及其[预训练模型](#预训练模型)（BERT、ERNIE等）、常见的[任务范式](#附录c内置任务范式paradigm)（分类、匹配、机器阅读理解等）和相应的[数据集读取与处理工具](#附录a内置数据集载入与处理工具reader)。同时框架提供了用户自定义接口，若内置工具、主干网络和任务无法满足需求，开发者可以轻松完成相关组件的自定义。各个组件均为零耦合设计，用户仅需完成组件本身的特性开发即可完成与框架的融合。

## 目录

- [安装](#安装)
- [前期准备](#前期准备)
    - [理论准备](#理论准备)
    - [框架原理](#框架原理)
    - [预训练模型](#预训练模型)
- [三个DEMO入门PaddlePALM](#三个demo入门paddlepalm)
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


## 安装

推荐使用pip安装paddlepalm框架：

```shell
pip install paddlepalm
```

对于离线机器，可以使用基于源码的安装方式：

```shell
git clone https://github.com/PaddlePaddle/PALM.git
cd PALM && python setup.py install
```



**环境依赖**

- Python >= 2.7
- cuda >= 9.0
- cudnn >= 7.0
- PaddlePaddle >= 1.6.1 (请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装)

## 框架代码结构

```text
.
├── mtl_controller.py                 # 任务控制器，负责创建和调度各个任务实例来完成多任务学习
├── task_instance.py                  # 任务实例类，完成任务实例的配置管理、训练进程管理、保存与载入等
├── downloader.py                     # 下载器，用于支持预训练模型的下载
├── default_settings.py               # 默认的环境变量和框架配置
├── utils                             # 框架核心工具集
│   ├── config_helper.py                  # 配置工具类，完成命令行与json、yaml的联合解析
│   ├── reader_helper.py                  # 完成多任务数据集iterators的合并、采样、调度和归一化，连接python生成器与计算图
│   ├── saver.py                          # 模型保存与载入
│   ├── print_helper.py                   # 日志打印规范化工具 
│   ├── plot_helper.py                    # 命令行绘图工具   
│   └── textprocess_helper.py             # 文本数据处理工具函数
├── backbone                          # 框架预置的主干网络
│   ├── ernie.py                          # ERNIE模型
│   ├── bert.py                           # BERT模型
│   └── utils                             # 实现主干网络的一些可复用的工具函数
├── reader                            # 框架内置的数据集载入与处理工具
│   ├── cls.py                            # 文本分类数据集工具
│   ├── match.py                          # 文本匹配数据集工具
│   ├── mrc.py                            # 机器阅读理解数据集工具
│   └── mlm.py                            # 掩码语言模型（mask language model）数据集生成与处理工具
└── paradigm                          # 任务范式
    ├── cls.py                            # 文本分类
    ├── match.py                          # 文本匹配
    ├── mrc.py                            # 机器阅读理解
    └── mlm.py                            # 掩码语言模型（mask language model）
```


## 前期准备

### 理论准备

框架默认采用一对多（One-to-Many）的参数共享方式，如图


![image-20191022194400259](https://tva1.sinaimg.cn/large/006y8mN6ly1g88ajvpqmgj31hu07wn5s.jpg)


例如我们有一个目标任务MRC和两个辅助任务MLM和MATCH，我们希望通过MLM和MATCH来提高目标任务MRC的测试集表现（即提升模型泛化能力），那么我们可以令三个任务共享相同的文本特征抽取模型（例如BERT、ERNIE等），然后分别为每个任务定义输出层，计算各自的loss值。

框架默认采用任务采样+mini-batch采样的方式（alternating mini-batches optimization）进行模型训练，即对于每个训练step，首先对候选任务进行采样（采样权重取决于用户设置的`mix_ratio`），而后从该任务的训练集中采样出一个mini-batch（采样出的样本数取决于用户设置的`batch_size`）。

### 框架原理
paddlepalm框架的运行原理图如图所示

![PALM原理图](https://tva1.sinaimg.cn/large/006y8mN6ly1g8j1isf3fcj31ne0tyqbd.jpg)

首先用户为数据集载入与处理、主干网络和任务编写配置文件（框架实现了灵活易用的[配置广播机制](#配置广播机制)），而后用其创建多任务学习的控制器（`Controller`）。进而控制器创建任务实例，并根据用户调用的训练和预测接口对其参数和各个任务实例进行管理和调度。下面我们通过三个DEMO和进阶篇来快速入门并更深入的理解paddlepalm。

### 预训练模型

#### 下载
我们提供了BERT、ERNIE等主干网络的相关预训练模型。为了加速模型收敛，获得更佳的测试集表现，我们强烈建议用户在多任务学习时尽量在预训练模型的基础上进行（而不是从参数随机初始化开始）。用户可以查看可供下载的预训练模型：

```shell
python download_models.py -l
```

用户可通过运行`python download_models.py -d <model_name>`下载需要的预训练模型，例如，下载预训练BERT模型（uncased large）的命令如下：

```shell
python download_models.py -d bert-en-uncased-large
```

此外，用户也可通过运行`python download_models.py -d all`下载已提供的所有预训练模型。

脚本会自动在**当前文件夹**中创建一个pretrain目录（注：运行DEMO时，需保证pretrain文件夹在PALM项目目录下），并在其中创建bert子目录，里面存放预训练模型(`params`文件夹内)、相关的网络参数(`bert_config.json`)和字典(`vocab.txt`)。除了BERT模型，脚本还提供了ERNIE预训练模型（uncased large）的一键下载，将`<model_name>`改成`ernie-en-uncased-large`即可。全部可用的预训练模型列表见[paddlenlp/lark](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/PaddleLARK)



## 三个DEMO入门PaddlePALM

### DEMO1：单任务训练

> 本demo路径位于`demo/demo1`

框架支持对任何一个内置任务进行传统的单任务训练。接下来我们启动一个复杂的机器阅读理解任务的训练，我们在`data/mrqa`文件夹中提供了[EMNLP2019 MRQA机器阅读理解评测](https://mrqa.github.io/shared)的部分比赛数据。下面我们利用该数据尝试完成一个基于BERT的机器阅读理解任务MRQA的单任务学习。

用户进入本demo目录后，可通过运行如下脚本一键开始本节任务的训练

```shell
bash run.sh
```

下面以该任务为例，讲解如何基于paddlepalm框架轻松实现该任务。

**1. 配置任务实例**

首先，我们编写该任务实例的配置文件`mrqa.yaml`，若该任务实例参与训练或预测，则框架将自动解析该配置文件并创建相应的任务实例。配置文件需符合yaml格式的要求。一个任务实例的配置文件最少应包含`train_file`，`reader`和`paradigm`这三个字段，分别代表训练集的文件路径`train_file`、使用的数据集载入与处理工具`reader`、任务范式`paradigm`。

```yaml
train_file: data/mrqa/train.json
reader: mrc
paradigm: mrc
```

*注：框架内置的其他数据集载入与处理工具见[这里](#附录a内置数据集载入与处理工具reader)，任务范式列表见[这里](#附录c内置任务范式paradigm)*

此外，我们还需要配置reader的预处理规则，各个预置reader支持的预处理配置和规则请参考[这里](#附录a内置数据集载入与处理工具reader)。预处理规则同样直接写入`mrqa.yaml`中。

```yaml
max_seq_len: 512
max_query_len: 64
doc_stride: 128 # 在MRQA数据集中，存在较长的文档，因此我们这里使用滑动窗口处理样本，滑动步长设置为128
do_lower_case: True
vocab_path: "../../pretrain/bert-en-uncased-large/vocab.txt"
```

更详细的任务实例配置方法（为任务实例选择合适的reader、paradigm和backbone）可参考[这里](#readerbackbone与paradigm的选择)

**2.配置backbone和训练规则**

然后我们编写全局配置文件`config_demo1.yaml`。在这里可以完成对主干网络(backbone)、多任务学习规则以及[广播到任务实例](#配置广播机制)的配置。同样使用yaml格式描述，例如在这里我们可以配置一下需要学习的任务`task_instance`、模型的保存路径`save_path`、基于的主干网络`backbone`、优化器`optimizer`等。

```yaml
task_instance: "mrqa"

save_path: "output_model/firstrun"

backbone: "bert"
backbone_config_path: "../../pretrain/bert-en-uncased-large/bert_config.json"

optimizer: "adam"
learning_rate: 3e-5
batch_size: 4

num_epochs: 2                                                                                    
warmup_proportion: 0.1 
```

这里的task_instance即填写我们刚刚编写的任务实例配置文件的文件名`mrqa`**（注意不要包括.yaml后缀！）**。框架启动多任务学习后会根据`task_instance`中指定的任务实例来寻找相关配置文件，并创建任务实例。

此外，backbone的相关配置除了可以直接写入全局配置文件以外，还可以在额外的一个json文件中进行描述，并在全局配置文件中通过`backbone_config_path`进行该配置文件路径的指定。

*注：框架支持的其他内置全局参数见[这里](#附录d可配置的全局参数列表)*

**3.开始训练**

下面我们开始尝试启动MRQA任务的训练（该代码位于`demo1.py`中）。如[框架原理](#框架原理)所述，框架的核心组件是`Controller`，负责多任务学习的启动。

```python
# Demo 1: single task training of MRQA 
import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config.yaml')
    controller.load_pretrain('../../pretrain/bert-en-uncased-large/params')
    controller.train()
```

训练日志如下，可以看到loss值随着训练收敛。在训练结束后，`Controller`自动为mrqa任务保存预测模型。

```
Global step: 10. Task: mrqa, step 10/135 (epoch 0), loss: 5.928, speed: 0.67 steps/s
Global step: 20. Task: mrqa, step 20/135 (epoch 0), loss: 4.594, speed: 0.75 steps/s
Global step: 30. Task: mrqa, step 30/135 (epoch 0), loss: 1.663, speed: 0.75 steps/s
...
Global step: 250. Task: mrqa, step 115/135 (epoch 1), loss: 1.391, speed: 0.75 steps/s
Global step: 260. Task: mrqa, step 125/135 (epoch 1), loss: 1.871, speed: 0.75 steps/s
Global step: 270. Task: mrqa, step 135/135 (epoch 1), loss: 1.544, speed: 0.75 steps/s
mrqa: train finished!
mrqa: inference model saved at output_model/firstrun/mrqa/infer_model
```

### DEMO2：多任务辅助训练与目标任务预测

> 本demo路径位于`demo/demo2`

本节我们考虑更加复杂的学习目标，我们引入一个掩码语言模型（Mask Language Model，MLM）问答匹配（QA Match）任务来辅助上一节MRQA任务的训练，相关训练数据分别位于`data/mlm4mrqa`和`data/match4mrqa`。并且我们这里换用ERNIE模型作为主干网络，来获得更佳的效果。在多任务训练结束后，我们使用训练好的模型来对MRQA任务的测试集进行预测。

用户可通过运行如下脚本直接开始本节任务的训练

```shell
bash run.sh
```

下面以该任务为例，讲解如何基于paddlepalm框架轻松实现这个复杂的多任务学习。

**1. 配置任务实例**

首先，我们像上一节一样为MLM任务和Matching任务分别创建任务实例的配置文件`mlm4mrqa.yaml`和`match4mrqa.yaml`，并将两个文件放入`task`文件夹中：

```yaml
----- mlm4mrqa.yaml -----
train_file: "data/mlm4mrqa/train.tsv"
reader: mlm
paradigm: mlm

----- match4mrqa.yaml -----
train_file: "data/match/train.tsv"
reader: match
paradigm: match
```

由于我们在训练结束后要对MRQA任务的测试集进行预测，因此我们要在之前写好的`mrqa.yaml`中追加预测相关的配置
```yaml
pred_file: data/mrqa/dev.json
pred_output_path: 'mrqa_output'
max_answer_len: 30
n_best_size: 20
```

**2.配置全局参数**

由于MRQA、MLM和Matching任务有相同的字典、大小写配置、截断长度等，因此我们可以将这些各个任务中相同的参数写入到全局配置文件`mtl_config.yaml`中，**框架会自动将该文件中的配置广播（broadcast）到各个任务实例。**

```yaml
task_instance: "mrqa, mlm4mrqa, match4mrqa"
target_tag: 1,0,0 

save_path: "output_model/secondrun"

backbone: "ernie"
backbone_config_path: "../../pretrain/ernie-en-uncased-large/ernie_config.json"

vocab_path: "../../pretrain/ernie-en-uncased-large/vocab.txt"
do_lower_case: True
max_seq_len: 512 # 写入全局配置文件的参数会被自动广播到各个任务实例

batch_size: 4
num_epochs: 2
optimizer: "adam"
learning_rate: 3e-5
warmup_proportion: 0.1 
weight_decay: 0.1 
```

这里我们可以使用`target_tag`来标记目标任务和辅助任务，各个任务的tag使用逗号`,`隔开。target_tag与task_instance中的元素一一对应，当某任务的tag设置为1时，表示对应的任务被设置为目标任务；设置为0时，表示对应的任务被设置为辅助任务，默认情况下所以任务均被设置为目标任务（即默认`target_tag`为全1）。

辅助任务不会保存预测模型，且不会影响训练的终止，仅仅起到“陪同训练”的作用以期提高模型的泛化能力。当所有的目标任务达到预期的训练步数后多任务学习终止，框架自动为每个目标任务保存预测模型（inference model）到设置的`save_path`位置。

同时需要注意的是，这里`num_epochs`指代目标任务`mrqa`的训练epoch数量（训练集遍历次数）。

在训练过程中，默认每个训练step会从各个任务等概率采样，来决定当前step训练哪个任务。但包括辅助任务在内，各个任务的采样概率是可以被控制的。若用户希望改变采样比率，可以通过`mix_ratio`字段来进行设置，例如

```yaml
mix_ratio: 1.0, 0.5, 0.5
```

若将如上设置加入到全局配置文件中，则辅助任务`mlm4mrqa`和`match4mrqa`的采样概率/预估的训练步数仅为`mrqa`任务的一半。关于采样概率的更多介绍请参考进阶篇。



**3.开始多任务训练**

```python
import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config.yaml', task_dir='tasks')
    controller.load_pretrain('../../pretrain/ernie-en-uncased-large/params')
    controller.train()

```

训练日志如下，在训练过程中可以看到每个任务的loss下降
```
Global step: 10. Task: mrqa, step 4/135 (epoch 0), loss: 6.235, speed: 0.75 steps/s
Global step: 20. Task: mrqa, step 8/135 (epoch 0), loss: 5.652, speed: 0.75 steps/s
Global step: 30. Task: mrqa, step 13/135 (epoch 0), loss: 6.031, speed: 0.75 steps/s
Global step: 40. Task: match4mrqa, step 13/25 (epoch 0), loss: 0.758, speed: 2.52 steps/s
Global step: 50. Task: mlm4mrqa, step 14/30 (epoch 0), loss: 7.322, speed: 3.24 steps/s
...
Global step: 547. Task: match4mrqa, step 13/25 (epoch 5), loss: 0.400, speed: 2.23 steps/s
Global step: 548. Task: match4mrqa, step 14/25 (epoch 5), loss: 0.121, speed: 3.03 steps/s
Global step: 549. Task: mrqa, step 134/135 (epoch 1), loss: 0.824, speed: 0.75 steps/s
Global step: 550. Task: mlm4mrqa, step 22/30 (epoch 4), loss: 6.903, speed: 3.59 steps/s
Global step: 551. Task: mrqa, step 135/135 (epoch 1), loss: 3.408, speed: 0.75 steps/s

mrqa: train finished!
mrqa: inference model saved at output_model/secondrun/mrqa/infer_model
```

**4.预测**

在得到目标任务的预测模型（inference_model）后，我们可以加载预测模型对该任务的测试集进行预测。在多任务训练阶段，在全局配置文件的`save_path`指定的路径下会为每个目标任务创建同名子目录，子目录中都有预测模型文件夹`infermodel`。我们可以将该路径传给框架的`controller`来完成对该目标任务的预测。

例如，我们在上一节得到了mrqa任务的预测模型。首先创建一个新的*Controller*，**并且创建时要将`for_train`标志位置为*False***。而后调用*pred*接口，将要预测的任务实例名字和预测模型的路径传入，即可完成相关预测。预测的结果默认保存在任务实例配置文件的`pred_output_path`指定的路径中。代码段如下：

```python
    controller = palm.Controller(config='config.yaml', task_dir='tasks', for_train=False)
    controller.pred('mrqa', inference_model_dir='output_model/secondrun/mrqa/infermodel') 
```

我们可以在刚刚yaml文件中设置的`mrqa_output/`文件夹下的`predictions.json`文件中看到类似如下的预测结果

```json
{                                                                                           
    "3f02f171c82e49828580007a71eefc31": "Ethan Allen", 
    "98d0b8ce19d1434abdb42aa01e83db61": "McDonald's", 
    "f0bc45a4dd7a4d8abf91a5e4fb25fe57": "Jesse James", 
    ...
}
```

其中的每一行是测试集中的一个question对应的预测答案（其中的key为question的id，详情见mrc reader的说明文档）。

### DEMO3：多目标任务联合训练与任务层参数复用

> 本demo路径位于`demo/demo3`

本节我们考虑一个更加复杂的大规模多任务学习场景。假如手头有若干任务，其中每个任务都可能将来被用于预测（即均为目标任务），且鉴于这若干个任务之间存在一些相关性，我们希望将其中一部分任务的任务层参数也进行复用。分类数据集位于`data/cls4mrqa`内。

具体来说，例如我们有6个分类任务（CLS1 ~ CLS6），均为目标任务（每个任务的模型都希望未来拿来做预测和部署），且我们希望任务1，2，5的任务输出层共享同一份参数，任务3、4共享同一份参数，任务6自己一份参数，即希望对6个任务实现如图所示的参数复用关系。

![image2](https://tva1.sinaimg.cn/large/006y8mN6ly1g8issdoli5j31ow08ogxv.jpg)

如图，在同一个方框内的任务共享相同的任务层参数。

用户可通过运行如下脚本一键开始学习本节任务目标：

```shell
bash run.sh
```

**1. 配置任务实例**

为了演示方便，我们使用同一份数据集来创建6个分类的任务实例，分别命名为`cls1.yaml`, `cls2.yaml`, `cls3.yaml`, `cls4.yaml`, `cls5.yaml`, `cls6.yaml`。每个实例的配置文件中填入如下必要字段

```yaml
train_file: "data/cls4mrqa/train.tsv"
reader: cls
paradigm: cls

n_classes: 4
```

**2.配置全局参数**

在paddlepalm中可以轻松完成上述的复杂复用关系的定义，我们使用`task_reuse_tag`来描述任务层的参数复用关系，与`target_tag`一样，`task_reuse_tag`中的元素与`task_instance`一一对应，元素取值相同的任务会自动共享任务层参数，取值不同的任务不复用任务层参数。因此可以在全局配置文件中如下描述

```yaml
task_instance: "cls1, cls2, cls3, cls4, cls5, cls6"
task_reuse_tag: 0, 0, 1, 1, 0, 2
```

同时，这6个任务均为目标任务，因此我们不需要手动设置`target_tag`了（任务默认即为目标任务）。不过，**设置多个目标的情况下，依然可以添加辅助任务陪同这些目标任务进行训练**，这时候就需要引入`target_tag`来区分目标任务和辅助任务了。而后，我们在全局配置文件中写入其他必要的参数（backbone、优化器等）。

```yaml
save_path: "output_model/secondrun"

backbone: "ernie"
backbone_config_path: "../../pretrain/ernie-en-uncased-large/ernie_config.json"

vocab_path: "../../pretrain/ernie-en-uncased-large/vocab.txt"
do_lower_case: True
max_seq_len: 512 # 写入全局配置文件的参数会被自动广播到各个任务实例

batch_size: 4
num_epochs: 2
optimizer: "adam"
learning_rate: 3e-5
warmup_proportion: 0.1 
weight_decay: 0.1 
```
**3.开始多目标任务训练**

最后，我们像DEMO1和DEMO2一样创建`Controller`，实例化各个任务实例、载入预训练模型并启动多任务训练：

```yaml
import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config.yaml', task_dir='tasks')
    controller.load_pretrain('../../pretrain/ernie-en-uncased-large/params')
    controller.train()
 ```

可以看到如下日志输出。

```
Global step: 1. Task: cls4, step 1/15 (epoch 0), loss: 1.344, speed: 0.50 steps/s
Global step: 10. Task: cls4, step 5/15 (epoch 0), loss: 1.398, speed: 2.19 steps/s
Global step: 20. Task: cls2, step 5/15 (epoch 0), loss: 1.260, speed: 2.64 steps/s
cls4: train finished!
cls4: inference model saved at output_model/thirdrun/infer_model
cls5: train finished!
cls5: inference model saved at output_model/thirdrun/infer_model
Global step: 30. Task: cls2, step 7/15 (epoch 0), loss: 0.961, speed: 0.04 steps/s
cls2: train finished!
cls2: inference model saved at output_model/thirdrun/infer_model
Global step: 40. Task: cls6, step 4/15 (epoch 0), loss: 1.412, speed: 2.74 steps/s
Global step: 50. Task: cls2, step 12/15 (epoch 0), loss: 1.011, speed: 2.19 steps/s
cls6: train finished!
cls6: inference model saved at output_model/thirdrun/infer_model
cls1: train finished!
cls1: inference model saved at output_model/thirdrun/infer_model
Global step: 60. Task: cls3, step 7/15 (epoch 0), loss: 1.363, speed: 2.72 steps/s
cls3: train finished!
cls3: inference model saved at output_model/thirdrun/infer_model
```

对本DEMO更深入的理解可以参考[多目标任务下的训练终止条件与预期训练步数](#多目标任务下的训练终止条件与预期训练步数)。

## 进阶篇
本章节更深入的对paddlepalm的使用方法展开介绍，并提供一些提高使用效率的小技巧。

### 配置广播机制

![PALM原理图](https://tva1.sinaimg.cn/large/006y8mN6ly1g8j1isf3fcj31ne0tyqbd.jpg)

要完成多任务学习，我们需要对主干网络、各个任务以及训练方式进行必要的配置，为此，框架实现了一套高效的配置广播机制。如上图，通过yaml语言可以描述主干网络和各个任务实例的相关配置，并存储于文件中。由于任务实例可能有多个，且部分超参数会同时被主干网络和任务实例用到，因此对于这些需要“重复配置”却取值相同的超参数，可以写入全局配置文件中，框架在解析全局配置文件时会自动将其“广播”给主干网络和各个任务实例。

此外，全局配置文件的优先级要高于主干网络和任务实例的配置文件，因此当某个超参数在全局配置文件的取值与其在其余位置的取值冲突时，框架以全局配置文件中的取值为准。

同时，为了方便进行大规模实验和超参数调优，凡是在**全局配置文件**中出现的超参数，均可以通过命令行进行控制，例如，对于如下全局配置文件

```yaml
...
learning_rate: 1e-3
batch_size: 32
...
```

我们可能希望通过命令行临时调整学习率`learning_rate`和批大小`batch_size`，因此我们在运行训练脚本时可以通过如下方式对其进行改变。

```shell
python demo3.py --learning_rate 1e-4 --batch_size 64
```

因此，各种配置方式的优先级如下

**命令行 > 全局配置文件 > 任务实例配置文件&主干网络配置文件**

### reader、backbone与paradigm的选择

reader、backbone和paradigm是实现各类任务的三大基础组件，其中reader为数据集载入与处理工具，将一定格式的输入数据集自动转换成确定的输出元素字典（如单词id序列，位置id序列等）；backbone为主干网络，将来自reader的一部分输出转换为高阶抽象的输出元素字典（如词向量、句向量、编码器输出的上下文相关词向量等）；paradigm为任务范式，将来自reader的一部分输出和backbone输出的对原始输入的高阶抽象转换为训练所需要的loss以及预测所需要的输出等。

框架对这三部分组件的实现基于一种解耦合的设计，每个组件都会包括对输入对象的描述inputs_attr(s)和对输出对象的描述outputs_attr，每个输入或输出对象都会包含名字（描述含义）、形状（tensor shape）和数值类型（data type）。例如，主干网络BERT的输入输出对象的声明如下

```python
    @property
    def inputs_attr(self):
        return {"token_ids": [[None, None], 'int64'],
                "position_ids": [[None, None], 'int64'],
                "segment_ids": [[None, None], 'int64'],
                "input_mask": [[None, None], 'float32']}

    @property
    def outputs_attr(self):
        return {"word_embedding": [[None, None, self._emb_size], 'float32'],
                "embedding_table": [[None, self._voc_size, self._emb_size], 'float32'],
                "encoder_outputs": [[None, None, self._emb_size], 'float32'],
                "sentence_embedding": [[None, self._emb_size], 'float32'],
                "sentence_pair_embedding": [[None, self._emb_size], 'float32']}
```

其中`inputs_attr`描述了BERT的输入对象，包含`token_ids`, `position_ids`, `segment_ids`和`input_mask`，并且附带了它们的形状（None表示Tensor在该维度的大小可变）和数据类型。`outputs_attr`则描述了BERT模块能提供的输出对象，包含`word_embedding`, `embedding_table`, `encoder_outputs`等。

当用户创建任务实例时，只需要保证每个组件的输入对象是包含在上游组件的输出内的，那么这些组件就可以搭配在一起使用。其中，backbone的上游组件是reader，paradigm的上游组件同时包含reader和backbone。


### 多目标任务下的训练终止条件与预期训练步数

#### 多个目标任务
框架支持设定多个目标任务，当全局配置文件的`task_instance`字段指定超过一个任务实例时，**这多个任务实例默认均为目标任务（即`target_tag`字段被自动填充为全1）**。对于被设置成目标任务的任务实例，框架会为其计算预期的训练步数并在达到预期训练步数后为其保存预测模型。

当框架存在多个目标任务时，全局配置文件中的`num_epochs`（训练集遍历次数）仅会作用于第一个出现的目标任务，称为主任务（main task）。框架会根据主任务的训练步数来推理其他目标任务的预期训练步数（可通过`mix_ratio`控制，详情见下一节）。**注意，除了用来标记`num_epochs`的作用对象外，主任务与其他目标任务没有任何不同。**

*注意：在多目标任务训练时，依然可以使用辅助任务来提升所有目标任务的测试集表现，但是要注意使用target_tag为引入的辅助任务打上辅助标记「0」*


#### 训练终止条件
在训练开始前，`Controller`会为所有每个目标任务计算出预期的训练步数。当某个目标任务的完成预期的训练步数后，`Controller`保存该任务的预测模型，而后继续按照设定的各任务的采样概率进行多任务训练。当所有目标任务均达到预期的训练步数后，多任务学习终止。需要注意的是，`Controller`不会为辅助任务计算预期训练步数，也不会为其保存预测模型，其仅仅起到“陪同目标任务训练”的作用，不会影响到多任务学习的终止与否。

#### 任务采样概率与预期训练步数
此外，在默认情况下，每个训练step的各个任务被采样到的概率均等，若用户希望更改其中某些任务的采样概率（比如某些任务的训练集较小，希望减少对其采样的次数；或某些任务较难，希望被更多的训练），可以在全局配置文件中通过`mix_ratio`字段控制各个任务的采样概率。例如，我们有三个任务，其中mrqa任务为目标任务，其余为辅助任务，我们对其`mix_ratio`进行如下设定：

```yaml
task_instance: mrqa, match4mrqa, mlm4mrqa
mix_ratio: 1.0, 0.5, 0.5
```

上述设置表示`match4mrqa`和`mlm4mrqa`任务的期望被采样次数均为`mrqa`任务的一半。此时，在mrqa任务被设置为目标任务的情况下，若mrqa任务训练一个epoch要经历5000 steps，且全局配置文件中设置了num_epochs为2，则根据上述`mix_ratio`的设置，mrqa任务将被训练5000\*2\*1.0=10000个steps，而`match4mrqa`任务和`mlm4mrqa`任务都会被训练5000个steps**左右**。

> 注意：若match4mrqa, mlm4mrqa被设置为辅助任务，则实际训练步数可能略多或略少于5000个steps。对于目标任务，则是精确的5000 steps。

#### 多个目标任务时预期训练步数的计算

当存在多个目标任务时，`num_epochs`仅作用于**第一个设定的目标任务（称为“主任务(main task)”）**，而后根据`mix_ratio`的设定为其余目标任务和辅助任务计算出预期的训练步数。

### 模型保存与预测机制

`Controller`可以在训练过程中保存两类模型，一类称为检查点模型（checkpoint），一类为预测模型（inference model）。

检查点模型会描述当前训练时刻的网络全局状态，包括backbone、所有任务以及优化器的全局参数，局部参数，长期变量等，即完整的多任务学习计算图。检查点模型用于训练意外终止时的断点恢复，或分阶段的对相同的模型进行连续训练。对于检查点模型，`Controller`默认不进行保存，但是用户可以通过在全局配置文件中添加`save_every_n_steps`来控制检查点模型的保存频率，例如设置为5000，则表示每5000个全局训练steps就会保存一次检查点模型。检查点模型放置在全局配置文件中设置的`save_path`指定的路径下。

预测模型则描述的是某个任务的完整预测模型，该模型内不会包含其他任务的参数，也不会保存优化器、dropout层等推理阶段不需要的节点。在保存预测模型时，`Controller`会同时保存预测相关的必要配置，如预测模型的输入输出列表，在进行预测时，可以调用实例化后的`Controller`的预测接口`pred`直接对相关任务进行预测。关于预测的用法示例可以参加DEMO2。

### 分布式训练

框架将单机单卡训练与单机多卡训练进行了无缝集成。当环境内有多张可用的GPU显卡时，框架会自动将模型复制到多张卡上，并且对于每个step，每张卡都会计算`batch_size`个训练样本，框架会自动对多卡的梯度进行合并。例如，环境中存在8张显卡，且`batch_size`设置为32时，这时每个step的实际batch size为32\*8=256。

当用户在多卡环境下希望仅用一张卡进行训练时，可以通过改变环境变量[CUDA_VISIBLE_DEVICES](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)来进行控制。



## 附录A：内置数据集载入与处理工具（reader）

所有的内置reader均同时支持中英文输入数据，**默认读取的数据为英文数据**，希望读入中文数据时，需在配置文件中设置

```yaml
for_cn: True
```

所有的内置reader，均支持以下字段

```yaml
vocab_path（REQUIRED）: str类型。字典文件路径。
max_seq_len（REQUIRED）: int类型。切词后的序列最大长度（即token ids的最大长度）。注意经过分词后，token ids的数量往往多于原始的单词数（e.g., 使用wordpiece tokenizer时）。
batch_size（REQUIRED）: int类型。训练或预测时的批大小（每个step喂入神经网络的样本数）。
train_file（REQUIRED）: str类型。训练集文件所在路径。仅进行预测时，该字段可不设置。
pred_file（REQUIRED）: str类型。测试集文件所在路径。仅进行训练时，该字段可不设置。

do_lower_case（OPTIONAL）: bool类型，默认为False。是否将大写英文字母转换成小写。
shuffle（OPTIONAL）: bool类型，默认为True。训练阶段打乱数据集样本的标志位，当置为True时，对数据集的样本进行全局打乱。注意，该标志位的设置不会影响预测阶段（预测阶段不会shuffle数据集）。
seed（OPTIONAL）: int类型，默认为。
pred_batch_size（OPTIONAL）: int类型。预测阶段的批大小，当该参数未设置时，预测阶段的批大小取决于`batch_size`字段的值。
print_first_n（OPTIONAL）: int类型。打印数据集的前n条样本和对应的reader输出，默认为0。
```

#### 文本分类数据集reader工具：cls

该reader完成文本分类数据集的载入与处理，reader接受[tsv格式](https://en.wikipedia.org/wiki/Tab-separated_values)的数据集输入，数据集应该包含两列，一列为样本标签`label`，一列为原始文本`text_a`。数据集范例可参考`data/cls4mrqa`中的数据集文件，格式形如

```
label   text_a                                                                                   
1   when was the last time the san antonio spurs missed the playoffshave only missed the playoffs four times since entering the NBA
0   the creation of the federal reserve system was an attempt toReserve System ( also known as the Federal Reserve or simply the Fed ) is the central banking system of the United States of America . 
2   group f / 64 was a major backlash against the earlier photographic movement off / 64 was formed , Edward Weston went to a meeting of the John Reed Club , which was founded to support Marxist artists and writers . 
0   Bessarabia eventually became under the control of which country?
```
***注意：数据集的第一列必须为header，即标注每一列的列名***

该reader额外包含以下配置字段

```yaml
n_classes（REQUIRED）: int类型。分类任务的类别数。
```

reader的输出（生成器每次yield出的数据）包含以下字段

```yaml
token_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的单词id。
position_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的位置id。
segment_ids: 一个shape为[batch_size, seq_len]的全0矩阵，用于支持BERT、ERNIE等模型的输入。
input_mask: 一个shape为[batch_size, seq_len]的矩阵，其中的每个元素为0或1，表示该位置是否是padding词（为1时代表是真实词，为0时代表是填充词）。
label_ids: 一个shape为[batch_size]的矩阵，其中的每个元素为该样本的类别标签。
task_ids: 一个shape为[batch_size, seq_len]的全0矩阵，用于支持ERNIE模型的输入。
```

当处于预测阶段时，reader所yield出的数据不会包含`label_ids`字段。


#### 文本匹配数据集reader工具：match

该reader完成文本匹配数据集的载入与处理，reader接受[tsv格式](https://en.wikipedia.org/wiki/Tab-separated_values)的数据集输入，数据集应该包含三列，一列为样本标签`label`，其余两列分别为待匹配的文本`text_a`和文本`text_b`。数据集范例可参考`data/match4mrqa`中的数据集文件，格式形如

```yaml
label   text_a  text_b                                                                           
1   From what work of Durkheim's was interaction ritual theory derived? **[TAB]** Subsequent to these developments, Randall Collins (2004) formulated his interaction ritual theory by drawing on Durkheim's work on totemic rituals that was extended by Goffman (1964/2013; 1967) into everyday focused encounters. Based on interaction ritual theory, we experience different levels
0   where is port au prince located in haiti **[TAB]** Its population is difficult to ascertain due to the rapid growth of slums in the hillsides
0   What is the world’s first-ever pilsner type blond lager, the company also awarded the Master Homebrewer Competition held in San Francisco to an award-winning brewer who won the prestigious American Homebrewers Associations' Homebrewer of the Year award in 2013? **[TAB]** of the Year award in 2013, becoming the first woman in thirty years, and the first African American person ever to ever win the award.
1   What has Pakistan told phone companies? **[TAB]** Islamabad, Pakistan (CNN) -- Under heavy criticism for a telling cell phone carriers to ban certain words in text messages, the Pakistan Telecommunication Authority went into damage control mode Wednesday.
```

***注意：数据集的第一列必须为header，即标注每一列的列名***

reader的输出（生成器每次yield出的数据）包含以下字段：

```yaml
token_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本（文本对），其中的每个元素为文本对中的每个token对应的单词id，文本对使用`[SEP]`所对应的id隔开。
position_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的位置id。
segment_ids: 一个shape为[batch_size, seq_len]的矩阵，在文本1的token位置，元素取值为0；在文本2的token位置，元素取值为1。用于支持BERT、ERNIE等模型的输入。
input_mask: 一个shape为[batch_size, seq_len]的矩阵，其中的每个元素为0或1，表示该位置是否是padding词（为1时代表是真实词，为0时代表是填充词）。
label_ids: 一个shape为[batch_size]的矩阵，其中的每个元素为该样本的类别标签，为0时表示两段文本不匹配，为1时代表构成匹配。
task_ids: 一个shape为[batch_size, seq_len]的全0矩阵，用于支持ERNIE模型的输入。
```

当处于预测阶段时，reader所yield出的数据不会包含`label_ids`字段。


#### 机器阅读理解数据集reader工具：mrc

该reader支持基于滑动窗口的机器阅读理解数据集载入，可以自动将较长的context按照步长切分成若干子文档，每个子文档与question分别计算答案片段，并在最终阶段合并。该reader接受[json格式]()的数据集。数据集范例可参考`data/mrqa`中的数据集文件，格式如下。

```json
{
    "version": "1.0",
    "data": [
        {"title": "...",
         "paragraphs": [
            {"context": "...",
             "qas": [
                {"question": "..."
                 "id": "..."
                 "answers": [
                    {"text": "...",
                     "answer_start": ...}
                    {...}
                    ...
                    ]
                 }
                 {...}
                 ...
             {...},
             ...
             ]
         }
         {...}
         ...
     ]
 }
 ```
 
数据集的最外层数据结构为字典，包含数据集版本号`version`和数据集`data`。在`data`字段内为各个样本，每个样本包含文章标题`title`和若干段落`paragraphs`，在`paragraphs`中的每个元素为一个段落`context`，基于该段落的内容，可以包含若干个问题和对应的答案`qas`，答案均位于该段落内。对于`qas`中的每个元素，包含一个问题`question`和一个全局唯一的标识`id`，以及（若干）答案`answers`。答案中的每个元素包含答案本身`text`及其在`context`中的起始位置`answer_start`。注意起始位置为字符级。此外，在测试集中，`qas`可以不包含`answers`字段。

该reader包含如下额外的可配置字段：

```yaml
doc_stride (REQUIRED): int类型。对context应用滑动窗口时的滑动步长。
max_query_len (REQUIRED): int类型。query的最大长度。
max_answer_len (REQUIRED): int类型。预测阶段answer的最大长度，不训练时该字段可为空。
n_best_size (OPTIONAL): int类型。预测阶段合并滑动窗口的样本时，每个样本所取的n_best列表大小。
```


reader的输出（生成器每次yield出的数据）包含以下字段：

```yaml
token_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本（文本对），文本1为context，文本2为question，其中的每个元素为文本对中的每个token对应的单词id，文本对使用`[SEP]`所对应的id隔开。
position_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的位置id。
segment_ids: 一个shape为[batch_size, seq_len]的矩阵，在文本1的token位置，元素取值为0；在文本2的token位置，元素取值为1。用于支持BERT、ERNIE等模型的输入。
input_mask: 一个shape为[batch_size, seq_len]的矩阵，其中的每个元素为0或1，表示该位置是否是padding词（为1时代表是真实词，为0时代表是填充词）。
task_ids: 一个shape为[batch_size, seq_len]的全0矩阵，用于支持ERNIE模型的输入。
start_positions: 一个shape为[batch_size]的向量，每个元素代表当前样本的答案片段的起始位置。
end_positions: 一个shape为[batch_size]的向量，每个元素代表当前样本的答案片段的结束位置。
```

当处于预测阶段时，reader所yield出的数据不会包含`label_ids`字段，但会额外的包含`unique_ids`字段：

```yaml
unique_ids: 一个shape为[batch_size, seq_len]的矩阵，代表每个样本的全局唯一的id，用于预测后对滑动窗口的结果进行合并。
```


#### 掩码语言模型数据集reader工具：mlm
该reader完成掩码语言模型数据集的载入与处理，reader接受[tsv格式](https://en.wikipedia.org/wiki/Tab-separated_values)的数据集输入，MLM任务为自监督任务，数据集仅包含一列`text_a`，reader会自动为每个样本生成随机的训练标签。格式如下

```
text_a                                                                                           
Subsequent to these developments, Randall Collins (2004) formulated his interaction ritual theory by drawing on Durkheim's work on totemic rituals that was extended by Goffman (1964/2013; 1967) into everyday focused encounters. 
Presidential spokesman Abigail Valte earlier Saturday urged residents of low-lying and mountainous areas that could be hit hard by the storm to evacuate, the state news agency said, citing an interview conducted on a government radio station. World Vision, the Christian humanitarian organization, said Saturday that it had to postpone some of its relief efforts due to Nalgae, with two of three emergency teams set to deploy once the storm passes. Another team is in Bulcan province, most of which is "still submerged" because of Nesat. The group is focusing its post-Nesat efforts on two communities in Manila and three in the northern Isabela and Zambales provinces. 
of the Year award in 2013, becoming the first woman in thirty years, and the first African American person ever to ever win the award. After an extensive career with the California State Legislature she began working for PicoBrew, a product development company in Seattle, WA that specializes in automated brewing equipment. 
the gakkel ridge is a boundary between which two tectonic plates Mid-Atlantic Ridge ( MAR ) is a mid-ocean ridge , a divergent tectonic plate or constructive plate boundary located along the floor of the Atlantic Ocean , and part of the longest mountain range in the world . The ridge extends from a junction with the Gakkel Ridge ( Mid-Arctic Ridge ) northeast of Greenland southward to the Bouvet Triple Junction in the South Atlantic .
```

***注意：数据集的第一列必须为header，即标注每一列的列名***

reader的输出（生成器每次yield出的数据）包含以下对象：

```yaml
token_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的单词id。
position_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的位置id。
segment_ids: 一个shape为[batch_size, seq_len]的全0矩阵，用于支持BERT、ERNIE等模型的输入。
input_mask: 一个shape为[batch_size, seq_len]的矩阵，其中的每个元素为0或1，表示该位置是否是padding词（为1时代表是真实词，为0时代表是填充词）。
mask_label: 一个shape为[None]的向量，其中的每个元素为被mask掉的单词的真实单词id。
mask_pos: 一个shape为[None]的向量，长度与`mask_pos`一致且元素一一对应。每个元素表示被mask掉的单词的位置。
task_ids: 一个shape为[batch_size, seq_len]的全0矩阵，用于支持ERNIE模型的输入。
```

## 附录B：内置主干网络（backbone）

框架中内置了BERT和ERNIE作为主干网络，未来框架会引入更多的骨干网络如XLNet等。

#### BERT

BERT包含了如下输入对象

```yaml
token_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的单词id。
position_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的位置id。
segment_ids: 一个shape为[batch_size, seq_len]的0/1矩阵，用于支持BERT、ERNIE等模型的输入，当元素为0时，代表当前token属于分类任务或匹配任务的text1，为1时代表当前token属于匹配任务的text2。
input_mask: 一个shape为[batch_size, seq_len]的矩阵，其中的每个元素为0或1，表示该位置是否是padding词（为1时代表是真实词，为0时代表是填充词）。
```

提供了如下输出对象供下游组件使用。

```yaml
word_embedding: 一个shape为[batch_size, seq_len, emb_size]的张量（Tensor），float32类型。表示当前batch中各个样本的（上下文无关）词向量序列。
embedding_table: 一个shape为[vocab_size, emb_size]的矩阵，float32类型。表示BERT当前维护的词向量查找表矩阵。
encoder_outputs: 一个shape为[batch_size, seq_len, hidden_size]的Tensor, float32类型。表示BERT encoder对当前batch中各个样本的encoding结果。
sentence_embedding: 一个shape为[batch_size, hidden_size]的matrix, float32类型。每一行代表BERT encoder对当前batch中相应样本的句子向量（sentence embedding）
sentence_pair_embedding: 一个shape为[batch_size, hidden_size]的matrix, float32类型。每一行代表BERT encoder对当前batch中相应样本的句子向量（sentence embedding）
```

#### ERNIE

ERNIE包含了如下输入对象

```yaml
token_ids: 。一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的单词id。
position_ids: 一个shape为[batch_size, seq_len]的矩阵，每行是一条样本，其中的每个元素为文本中的每个token对应的位置id。
segment_ids: 一个shape为[batch_size, seq_len]的0/1矩阵，用于支持BERT、ERNIE等模型的输入，当元素为0时，代表当前token属于分类任务或匹配任务的text1，为1时代表当前token属于匹配任务的text2.
input_mask: 一个shape为[batch_size, seq_len]的矩阵，其中的每个元素为0或1，表示该位置是否是padding词（为1时代表是真实词，为0时代表是填充词）。
segment_ids: 一个shape为[batch_size, seq_len]的全0矩阵，用于支持BERT、ERNIE等模型的输入。
task_ids: 一个shape为[batch_size, seq_len]的全0矩阵，用于支持ERNIE finetuning。
```

提供了如下输出对象供下游组件使用。

```yaml
word_embedding: 一个shape为[batch_size, seq_len, emb_size]的张量（Tensor），float32类型。表示当前batch中各个样本的（上下文无关）词向量序列。
embedding_table: 一个shape为[vocab_size, emb_size]的矩阵，float32类型。表示BERT当前维护的词向量查找表矩阵。
encoder_outputs: 一个shape为[batch_size, seq_len, hidden_size]的Tensor, float32类型。表示BERT encoder对当前batch中各个样本的encoding结果。
sentence_embedding: 一个shape为[batch_size, hidden_size]的matrix, float32类型。每一行代表BERT encoder对当前batch中相应样本的句子向量（sentence embedding）
sentence_pair_embedding: 一个shape为[batch_size, hidden_size]的matrix, float32类型。每一行代表BERT encoder对当前batch中相应样本的句子向量（sentence embedding）
```


## 附录C：内置任务范式（paradigm）


#### 分类范式：cls

分类范式额外包含以下配置字段：

```yaml
n_classes（REQUIRED）: int类型。分类任务的类别数。
pred_output_path (OPTIONAL) : str类型。预测输出结果的保存路径，当该参数未空时，保存至全局配置文件中的`save_path`字段指定路径下的任务目录。
save_infermodel_every_n_steps (OPTIONAL) : int类型。周期性保存预测模型的间隔，未设置或设为-1时仅在该任务训练结束时保存预测模型。默认为-1。
```

分类范式包含如下的输入对象：

训练阶段：
```yaml
sentence_embedding: 一个shape为[batch_size, hidden_size]的matrix, float32类型。每一行代表BERT encoder对当前batch中相应样本的句子向量（sentence embedding）
label_ids: 一个shape为[batch_size]的矩阵，其中的每个元素为该样本的类别标签。
```

预测阶段：
```yaml
sentence_embedding: 一个shape为[batch_size, hidden_size]的matrix, float32类型。每一行代表BERT encoder对当前batch中相应样本的句子向量（sentence embedding）
```

在训练阶段，输出loss；预测阶段输出各个类别的预测概率。

#### 匹配范式：match


匹配范式额外包含以下配置字段：

```yaml
pred_output_path (OPTIONAL) : str类型。预测输出结果的保存路径，当该参数未空时，保存至全局配置文件中的`save_path`字段指定路径下的任务目录。
save_infermodel_every_n_steps (OPTIONAL) : int类型。周期性保存预测模型的间隔，未设置或设为-1时仅在该任务训练结束时保存预测模型。默认为-1。
```

匹配范式包含如下的输入对象：

训练阶段：
```yaml
sentence_pair_embedding: 一个shape为[batch_size, hidden_size]的matrix, float32类型。每一行代表BERT encoder对当前batch中相应样本的句子向量（sentence embedding）
label_ids: 一个shape为[batch_size]的矩阵，其中的每个元素为该样本的类别标签，为0时表示两段文本不匹配，为1时代表构成匹配
```

预测阶段：
```yaml
sentence_pair_embedding: 一个shape为[batch_size, hidden_size]的matrix, float32类型。每一行代表BERT encoder对当前batch中相应样本的句子向量（sentence embedding）
```

在训练阶段，输出loss；预测阶段输出匹配与否的概率分布。

#### 机器阅读理解范式：mrc


分类范式额外包含以下配置字段：

```yaml
max_answer_len（REQUIRED）: int类型。预测的最大答案长度
n_best_size (OPTIONAL) : int类型，默认为20。预测时保存的nbest回答文件中每条样本的n_best数量
pred_output_path (OPTIONAL) : str类型。预测输出结果的保存路径，当该参数未空时，保存至全局配置文件中的`save_path`字段指定路径下的任务目录
save_infermodel_every_n_steps (OPTIONAL) : int类型。周期性保存预测模型的间隔，未设置或设为-1时仅在该任务训练结束时保存预测模型。默认为-1。
```

机器阅读理解范式包含如下的输入对象：

训练阶段：
```yaml
encoder_outputs: 一个shape为[batch_size, seq_len, hidden_size]的Tensor, float32类型。表示BERT encoder对当前batch中各个样本的encoding结果。
start_positions: 一个shape为[batch_size]的向量，每个元素代表当前样本的答案片段的起始位置。
end_positions: 一个shape为[batch_size]的向量，每个元素代表当前样本的答案片段的结束位置。
```

预测阶段：
```yaml
encoder_outputs: 一个shape为[batch_size, seq_len, hidden_size]的Tensor, float32类型。表示BERT encoder对当前batch中各个样本的encoding结果。
unique_ids: 一个shape为[batch_size, seq_len]的矩阵，代表每个样本的全局唯一的id，用于预测后对滑动窗口的结果进行合并。
```


#### 掩码语言模型范式：mlm

该任务范式为无监督任务范式，不支持预测，仅用于（辅助）训练。包含如下的输入对象：

```yaml
mask_label: 一个shape为[None]的向量，其中的每个元素为被mask掉的单词的真实单词id。
mask_pos": 一个shape为[None]的向量，长度与`mask_pos`一致且元素一一对应。每个元素表示被mask掉的单词的位置。
embedding_table: 一个shape为[vocab_size, emb_size]的矩阵，float32类型。表示BERT当前维护的词向量查找表矩阵。
encoder_outputs: 一个shape为[batch_size, seq_len, hidden_size]的Tensor, float32类型。表示BERT encoder对当前batch中各个样本的encoding结果。
```

## 附录D：可配置的全局参数列表

```yaml

task_instance（REQUIRED）: str类型。需要进行训练或预测的任务实例名。在多任务模式下，多个任务之间使用逗号`,`隔开。名称选取自任务实例配置文件的文件名（不包含后缀.yaml）。
mix_ratio (OPTIONAL): str类型。每个任务的训练阶段的采样概率，各个值通过逗号`,`隔开，且与task_instance中的元素一一对应。默认每个任务的采样概率均为1.0，即所有任务等概率采样（代表与主任务采样次数的期望相同）。详情见 《进阶篇-训练终止条件与预期训练步数》。
target_tag (OPTIONAL): str类型。目标/辅助任务标志位，各个值通过逗号`,`隔开，且与task_instance中的元素一一对应。标记为1的任务代表目标任务，标记为0的任务代表辅助任务。默认每个值均为1（即默认每个任务为目标任务）。相关使用示例见DEMO2。
task_reuse_tag (OPTIONAL): str类型。任务层复用标志位，各个值通过逗号`,`隔开，且与task_instance中的元素一一对应。元素取值相同的任务会自动共享任务层参数，取值不同的任务不复用任务层参数。相关使用示例见DEMO3。

backbone（REQUIRED）: str类型。主干网络名。
backbone_config_path (OPTIONAL): str类型。主干网络配置文件路径。

save_path（REQUIRED）: str类型。checkpoint文件和各个目标任务的预测模型保存路径。
vocab_path（REQUIRED）: str类型。字典文件，纯文本格式存储，其中每行为一个单词，供reader、backbone和各个任务使用。
do_lower_case (OPTIONAL): bool类型。大小写标志位。默认为False，即区分大小写。
for_cn: bool类型。中文模式标志位。默认为False，即默认输入为英文，设置为True后，分词器、后处理等按照中文语言进行处理。

print_every_n_steps (OPTIONAL): int类型。默认为5。训练阶段打印日志的频率（step为单位）。
save_ckpt_every_n_steps (OPTIONAL): int类型。默认为-1。训练过程中保存完整计算图的检查点（checkpoint）的频率，默认-1，仅在最后一个step自动保存检查点。
save_infermodel_every_n_steps (OPTIONAL) : int类型。周期性保存预测模型的间隔，未设置或设为-1时仅在该任务训练结束时保存预测模型。默认为-1。

optimizer（REQUIRED）: str类型。优化器名称，目前框架只支持adam，未来会支持更多优化器。
learning_rate（REQUIRED）: str类型。训练阶段的学习率。
batch_size（REQUIRED）: int类型。批大小，即每个训练或推理step所使用样本数。
epoch（REQUIRED）: int类型。主任务的训练epoch数。

use_gpu (OPTIONAL): bool类型。默认为True。框架默认使用GPU进行单机单卡或分布式训练，若希望使用cpu训练或推理，可将该标志位置为False。

warmup_proportion (OPTIONAL): float类型。默认为0。对预训练模型finetuning时的warmup的训练step占预估的全部训练步数的比例。
use_ema (OPTIONAL): bool类型。默认为False。是否开启[ema](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average) 进行训练和推理。
ema_decay (OPTIONAL): float类型。默认为0。开启ema时的权重衰减指数。
random_seed (OPTIONAL): int类型。随机种子，默认1。
```

## License

This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE).

## 许可证书

此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE)许可认证。


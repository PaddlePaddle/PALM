# PaddlePALM

PaddlePALM (PAddLe for Multi-task) 是一个强大快速、灵活易用的NLP多任务学习框架，用户仅需书写极少量代码即可完成复杂的多任务训练与推理。同时框架提供了定制化接口，若内置工具、主干网络和任务无法满足需求，开发者可以轻松完成相关组件的自定义。

框架中内置了丰富的主干网络及其预训练模型（BERT、ERNIE等）、常见的任务范式（分类、匹配、机器阅读理解等）和相应的数据集读取与处理工具。相关列表见这里。

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
- PaddlePaddle >= 1.6 (请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装)



## 前期准备

#### 理论准备

框架默认采用一对多（One-to-Many）的参数共享方式，如图


![image-20191022194400259](https://tva1.sinaimg.cn/large/006y8mN6ly1g88ajvpqmgj31hu07wn5s.jpg)


例如我们有一个目标任务MRC和两个辅助任务MLM和MATCH，我们希望通过MLM和MATCH来提高目标任务MRC的测试集表现（即提升模型泛化能力），那么我们可以令三个任务共享相同的文本特征抽取模型（例如BERT、ERNIE等），然后分别为每个任务定义输出层，计算各自的loss值。

框架默认采用任务采样+mini-batch采样的方式（alternating mini-batches optimization）进行模型训练，即对于每个训练step，首先对候选任务进行采样（采样权重取决于用户设置的`mix_ratio`），而后从该任务的训练集中采样出一个mini-batch（采样出的样本数取决于用户设置的`batch_size`）。

#### 模型准备

我们提供了BERT、ERNIE等主干模型及其相关预训练模型。为了加速模型收敛，获得更佳的测试集表现，我们强烈建议用户在预训练模型的基础上进行多任务学习（而不是从参数随机初始化开始）。用户可通过运行脚本`script/download_pretrain_models`下载需要的预训练模型，例如，下载预训练BERT模型的命令如下

```shell
bash script/download_pretrain_backbone.sh bert
```

脚本会自动在当前文件夹中创建一个pretrain_model目录，并在其中创建bert子目录，里面存放预训练模型(`params`文件夹内)、相关的网络参数(`bert_config.json`)和字典(`vocab.txt`)。

然后通过运行`script/convert_params.sh`将预训练模型转换成框架可用的预训练backbone

```shell
bash script/convert_params.sh pretrain_model/bert/params
```

*注：未来框架将支持更多的预置主干网络，如XLNet、多层LSTM等。此外，用户可以自定义添加新的主干网络，详情见[这里]()*

## DEMO1：单任务训练

接下来我们启动一个复杂的机器阅读理解任务的训练，我们在`data/mrqa`文件夹中提供了EMNLP2019 MRQA机器阅读理解评测的部分比赛数据。

用户可通过运行如下脚本一键开始本节任务的训练

```shell
bash run_demo1.sh
```


下面以该任务为例，讲解如何基于paddlepalm框架轻松实现该任务。

**1. 配置任务实例**

首先，我们编写该任务实例的配置文件`mrqa.yaml`，若该任务实例参与训练或预测，则框架将自动解析该配置文件并创建相应的任务实例。配置文件需符合yaml格式的要求。一个任务实例的配置文件最少应包含`train_file`，`reader`和`paradigm`这三个字段，分别代表训练集的文件路径`train_file`、使用的数据集载入与处理工具`reader`、任务范式`paradigm`。

```yaml
train_file: data/mrqa/mrqa-combined.train.raw.json
reader: mrc4ernie # 我们接下来会使用ERNIE作为主干网络，因此使用ernie配套的数据集处理工具mrc4ernie
paradigm: mrc
```

*注：框架内置的其他数据集载入与处理工具和任务范式列表见[这里]()*

此外，我们还需要配置reader的预处理规则，各个预置reader支持的预处理配置和规则请参考【这里】。预处理规则同样直接写入`mrqa.yaml`中。

```yaml
max_seq_len: 512
max_query_len: 64
doc_stride: 128 # 在MRQA数据集中，存在较长的文档，因此我们这里使用滑动窗口处理样本，滑动步长设置为128
do_lower_case: True
vocab_path: "pretrain_model/ernie/vocab.txt"
```

更详细的任务实例配置方法可参考这里

**2.配置全局参数**

然后我们配置全局的学习规则，同样使用yaml格式描述，我们新建`mtl_conf.yaml`。在这里我们配置一下需要学习的任务、模型的保存路径`save_path`和规则、使用的模型骨架`backbone`、学习器`optimizer`等。

```yaml
task_instance: "mrqa"

save_path: "output_model/firstrun"

backbone: "ernie"
backbone_config_path: "pretrain_model/ernie/ernie_config.json"

optimizer: "adam"
learning_rate: 3e-5
batch_size: 5

num_epochs: 2                                                                                    
warmup_proportion: 0.1 
```

*注：框架支持的其他backbone参数如日志打印控制等见[这里]()*

**3.开始训练**

下面我们开始尝试启动MRQA任务的训练（该代码位于`demo1.py`中）。框架的核心组件是`Controller`，

```python
# Demo 1: single task training of MRQA 
import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('demo1_config.yaml', task_dir='demo1_tasks')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()
```

默认情况下每5个训练step打印一次训练日志，如下（该日志在8卡P40机器上运行得到），可以看到loss值随着训练收敛。在训练结束后，`Controller`自动为mrqa任务保存预测模型。

```
Global step: 5. Task: mrqa, step 5/13 (epoch 0), loss: 4.976, speed: 0.11 steps/s
Global step: 10. Task: mrqa, step 10/13 (epoch 0), loss: 2.938, speed: 0.48 steps/s
Global step: 15. Task: mrqa, step 2/13 (epoch 1), loss: 2.422, speed: 0.47 steps/s
Global step: 20. Task: mrqa, step 7/13 (epoch 1), loss: 2.809, speed: 0.53 steps/s
Global step: 25. Task: mrqa, step 12/13 (epoch 1), loss: 1.744, speed: 0.50 steps/s
mrqa: train finished!
mrqa: inference model saved at output_model/firstrun/infer_model
```

## DEMO2：多任务训练与推理

本节我们考虑更加复杂的学习目标，我们引入一个问答匹配（QA Matching）任务来辅助MRQA任务的学习。在多任务训练结束后，我们希望使用训练好的模型来对MRQA任务的测试集进行预测。

用户可通过运行如下脚本直接开始本节任务的训练

```shell
bash run_demo2.sh
```



下面以该任务为例，讲解如何基于paddlepalm框架轻松实现这个复杂的多任务学习。

**1. 配置任务实例**

首先，我们像上一节一样为Matching任务分别配置任务实例`match4mrqa.yaml`：

```yaml
train_file: "data/match4mrqa/train.txt"
reader: match4ernie
paradigm: match
```

*注：更详细的任务实例配置方法可参考[这里]()*

**2.配置全局参数**

由于MRQA和Matching任务有相同的字典、大小写配置、截断长度等，因此我们可以将这些各个任务中相同的参数写入到全局配置文件`mtl_config.yaml`中，**框架会自动将该文件中的配置广播（broadcast）到各个任务实例。**

```yaml
task_instance: "mrqa, match4mrqa"
target_tag: 1,0 

save_path: "output_model/secondrun"

backbone: "ernie"
backbone_config_path: "pretrain_model/ernie/ernie_config.json"

vocab_path: "pretrain_model/ernie/vocab.txt"
do_lower_case: True
max_seq_len: 512 # 写入全局配置文件的参数会被自动广播到各个任务实例

batch_size: 5
num_epochs: 2
optimizer: "adam"
learning_rate: 3e-5
warmup_proportion: 0.1 
weight_decay: 0.1 
```

这里我们可以使用`target_tag`来标记目标任务和辅助任务，各个任务的tag使用逗号`,`隔开。target_tag与task_instance中的元素一一对应，当某任务的tag设置为1时，表示对应的任务被设置为目标任务；设置为0时，表示对应的任务被设置为辅助任务，默认情况下所以任务均被设置为目标任务（即默认`target_tag`为全1）。

辅助任务不会保存预测模型，且不会影响训练的终止。当所有的目标任务达到预期的训练步数后多任务学习终止，框架自动为每个目标任务保存预测模型（inference model）到设置的`save_path`位置。

在训练过程中，默认每个训练step会从各个任务等概率采样，来决定当前step训练哪个任务。若用户希望改变采样比率，可以通过`mix_ratio`字段来进行设置，例如

```yaml
mix_ratio: 1.0, 0.5
```

若将如上设置加入到全局配置文件中，则辅助任务`match4mrqa`的采样概率仅为`mrqa`任务的一半。

这里`num_epochs`指代目标任务`mrqa`的训练epoch数量（训练集遍历次数），**当目标任务有多个时，该参数将作用于第一个出现的目标任务（称为主任务，main task）**。

**3.开始多任务训练**

```python
import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('demo2_config.yaml', task_dir='demo2_tasks')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()

```

**4.预测**

在得到目标任务的预测模型（inference_model）后，我们可以加载预测模型对该任务的测试集进行预测。在多任务训练阶段，在全局配置文件的`save_path`指定的路径下会为每个目标任务创建同名子目录，子目录中都有预测模型文件夹`infermodel`。我们可以将该路径传给框架的`controller`来完成对该目标任务的预测。

例如，我们在上一节得到了mrqa任务的预测模型。首先创建一个新的*Controller*，**并且创建时要将`for_train`标志位置为*False***。而后调用*pred*接口，将要预测的任务实例名字和预测模型的路径传入，即可完成相关预测。预测的结果默认保存在任务实例配置文件的`pred_output_path`指定的路径中。代码段如下：

```python
    controller = palm.Controller(config='demo2_config.yaml', task_dir='demo2_tasks', for_train=False)
    controller.pred('mrqa', inference_model_dir='output_model/secondrun/mrqa/infermodel') 
```


## 进阶篇

### 设置多个目标任务

框架内支持设定多个目标任务，当全局配置文件的`task_instance`字段指定超过一个任务实例时，这多个任务实例默认均为目标任务（即`target_tag`字段被自动填充为全1）。对于被设置成目标任务的任务实例，框架会为其计算预期的训练步数（详情见下一节）并在达到预期训练步数后为其保存预测模型。

当框架存在多个目标任务时，全局配置文件中的`num_epochs`（训练集遍历次数）仅会作用于第一个出现的目标任务，称为主任务（main task）。框架会根据主任务的训练步数来推理其他目标任务的预期训练步数（可通过`mix_ratio`控制）。**注意，除了用来标记`num_epochs`的作用对象外，主任务与其他目标任务没有任何不同。**例如

```yaml
task_instance: domain_cls, mrqa, senti_cls, mlm, qq_match
target_tag: 0, 1, 1, 0, 1
```
在上述的设置中，mrqa，senti_cls和qq_match这三个任务被标记成了目标任务（其中mrqa为主任务），domain_cls和mlm被标记为了辅助任务。辅助任务仅仅“陪同”目标任务训练，框架不会为其保存预测模型（inference_model），也不会计算预期训练步数。但包括辅助任务在内，各个任务的采样概率是可以被控制的，详情见下一小节。

### 更改任务采样概率（期望的训练步数）

在默认情况下，每个训练step的各个任务被采样到的概率均等，若用户希望更改其中某些任务的采样概率（比如某些任务的训练集较小，希望减少对其采样的次数；或某些任务较难，希望被更多的训练），可以在全局配置文件中通过`mix_ratio`字段控制各个任务的采样概率。例如

```yaml
task_instance: mrqa, match4mrqa, mlm4mrqa
mix_ratio: 1.0, 0.5, 0.5
```

上述设置表示`match4mrqa`和`mlm4mrqa`任务的期望被采样次数均为`mrqa`任务的一半。此时，在mrqa任务被设置为主任务的情况下（第一个目标任务即为主任务），若mrqa任务训练一个epoch要经历5000 steps，且全局配置文件中设置了num_epochs为2，则根据上述`mix_ratio`的设置，mrqa任务将被训练5000\*2\*1.0=10000个steps，而`match4mrqa`任务和`mlm4mrqa`任务都会被训练5000个steps**左右**。

> 注意：若match4mrqa, mlm4mrqa被设置为辅助任务，则实际训练步数可能略多或略少于5000个steps。对于目标任务，则是精确的5000 steps。

### 共享任务层参数

### 分布式训练



## License

This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE).

## 许可证书

此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE)许可认证。



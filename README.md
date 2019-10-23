# PALM

PALM (PAddLe for Multi-task) 是一个灵活易用的多任务学习框架，并且支持用户定制化新型任务场景。

框架中内置了丰富的模型backbone（BERT、ERNIE等）、常见的任务范式（分类、匹配、序列标注、机器阅读理解等）和数据集读取与处理工具。

## 安装

推荐使用pip安装paddlepalm框架：

```shell
pip install paddlepalm
```

对于离线机器，可以使用基于源码的安装方式：

```shell
git clone https://github.com/PaddlePaddle/PALM.git
cd PALM && python setup.py
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


例如我们有一个目标任务MRC和两个辅助任务MLM和MATCH，我们希望通过MLM和MATCH来提高目标任务MRC的测试集表现（即提升模型泛化能力），那么我们可以令三个任务共享相同的文本特征抽取模型（例如图中的ERNIE），然后分别为每个任务定义输出层，计算各自的loss值。

框架默认采用任务采样+mini-batch采样的方式（alternating mini-batches optimization）进行模型训练，即对于每个训练step，首先对候选任务进行采样（采样权重取决于用户设置的`mix_ratio`），而后从该任务的训练集中采样出一个mini-batch（采样出的样本数取决于用户设置的`batch_size`）。

#### 模型准备

我们提供了ERNIE作为框架默认的主干模型，为了加速模型收敛，获得更佳的测试集表现，我们强烈建议用户在预训练模型的基础上进行多任务学习（而不是从参数随机初始化开始）。用户可通过运行脚本`script/download_pretrain_models`下载需要的预训练模型，下载预训练ERNIE的命令如下

```shell
bash script/download_pretrain_backbone.sh ernie
```

然后通过`script/convert_params.sh`将预训练模型转换成框架可用的预训练backbone

```python
bash script/convert_params.sh pretrain_model/ernie/params
```

*注：目前框架还支持BERT作为主干模型，未来将支持更多的预置主干网络，如XLNet、多层LSTM等。*

## 启动单任务训练

接下来我们启动一个复杂的机器阅读理解任务的训练，我们在`data/mrqa`文件夹中提供了EMNLP2019 MRQA机器阅读理解评测的部分比赛数据。

用户可通过运行如下脚本一键开始本节任务的训练

```shell
bash run_demo1.sh
```



下面以该任务为例，讲解如何基于paddlepalm框架轻松实现该任务。

首先，我们编写该任务实例的配置文件`mrqa.yaml`，框架将自动解析该配置文件并创建相应的任务实例。配置文件需符合yaml格式的要求。一个任务实例的配置文件最少应包含`train_file`，`reader`和`paradigm`这三个字段，分别代表训练集的文件路径`train_file`、使用的数据集载入与处理工具`reader`、任务范式`paradigm`。

```yaml
train_file: data/mrqa/mrqa-combined.train.raw.json
reader: mrc4ernie
paradigm: mrc
```

*注：框架内置的其他数据集载入与处理工具和任务范式列表见这里*

此外，我们还需要配置reader的预处理规则，各个预置reader支持的预处理配置和规则请参考【这里】。预处理规则同样直接写入`mrqa.yaml`中。

```yaml
max_seq_len: 512
max_query_len: 64
doc_stride: 128 # 在MRQA数据集中，存在较长的文档，因此我们这里使用滑动窗口处理样本，滑动步长设置为128
do_lower_case: True
vocab_path: "pretrain_model/ernie/vocab.txt"
```

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

*注：框架支持的其他backbone参数如日志打印控制等见这里*

而后我们就可以启动MRQA任务的训练了（该代码位于`demo1.py`中）。

```python
# Demo 1: single task training of MRQA 
import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('demo1_config.yaml', task_dir='demo1_tasks')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()
```



## 启动多任务训练

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

这里我们可以使用`target_tag`来标记目标任务和辅助辅助，每个任务默认均为目标任务，对于tag设置为0的任务，标记为辅助任务。辅助任务不会保存推理模型，且不会影响训练的终止。当所有的目标任务达到预期的训练步数后多任务学习终止，框架自动为每个目标任务保存推理模型（inference model）到设置的`save_path`位置。

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

```python
    controller = palm.Controller(config='demo2_config.yaml', task_dir='demo2_tasks', for_train=False)
    controller.pred('mrqa', inference_model_dir='output_model/secondrun/infer_model') 
```

## License

This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE).

## 许可证书

此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE)许可认证。



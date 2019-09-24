PALM
===
PALM (PAddLE Multitask) 是一个灵活易用的多任务学习框架，框架中内置了丰富的模型backbone（BERT、ERNIE等）、常见的任务范式（分类、匹配、序列标注、机器阅读理解等）和数据集读取与处理工具。对于典型的任务场景，用户几乎无需书写代码便可完成新任务的添加；对于特殊的任务场景，用户可通过对预置接口的实现来完成对新任务的支持。

## 安装

目前仅支持git clone源码的方式使用:
```shell
git clone https://github.com/PaddlePaddle/PALM.git
```

**环境依赖**
- Python >= 2.7
- cuda >= 9.0
- cudnn >= 7.0
- PaddlePaddle >= 1.5.0 (请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装)

## 目录结构

- backbone: 多任务学习的主干网络表示，支持bert, ernie, xlnet等，用户可自定义添加
- config：存放各个任务的配置文件，用户添加任务时需在此建立该任务的配置文件
- data: 存放各个任务的数据集
- pretrain_model: 存放预训练模型、字典及其相关配置
- optimizer: 优化器，用户可在此自定义优化器
- reader: 各个任务的数据读取与处理模块以及做reader融合的joint_reader文件
- paradigm: 任务输出层相关网络结构描述
- utils: 通用工具函数文件
- mtl_run.py: 多任务学习的主要流程描述
- run.sh: 多任务学习启动脚本

## 使用说明

框架给出了三个添加完成的任务示例：*Machine Reading Comprehension*、*Mask Language Model*和*Question Answer Matching*。其中在mtl_config.yaml中将*Machine Reading Comprehension*设置为了主任务，其他为辅助任务，用户可通过如下命令启动多任务学习

```
bash run.sh
```

### 多任务学习配置

在`mtl_config.yaml`中完成对多任务训练和推理的主配置，配置包含如下

***必选字段***

- main_task：*(str)* 指定主任务的名称，目前仅支持单个主任务。名称选取自config文件夹中的配置的文件名（不包含后缀.yaml和为task共享而设置的中间后缀）
- auxiliary_task：*(str)* 指定辅助任务，支持多个辅助任务，辅助任务之间使用空格隔开。名称选取自config文件夹中的配置的文件名（不包含后缀.yaml和为task共享而设置的中间后缀）
- do_train：*(bool)* 训练标志位
- do_predict：*(bool)* 预测标志位，目前仅支持对主任务进行预测
- checkpoint_path: *(str)* 模型保存、训练断点恢复和预测模型载入路径，从该路径载入模型时默认读取最后一个训练step的模型
- backbone_model：*(str)* 使用的骨干网络，名称选取自backbone目录下的模块
- vocab_path：*(str)* 字典文件，纯文本格式存储，其中每行为一个单词
- optimizer：*(str)* 优化器名称，名称选取自optimizer中的文件名
- learning_rate：*(str)* 训练阶段的学习率
- skip_steps：*(int)* 训练阶段打印日志的频率（step为单位）
- epoch：*(int)* 主任务的训练epoch数
- use_cuda：*(bool)* 使用GPU训练的标志位
- warmup_proportion：*(float)* 预训练模型finetuning时的warmup比例
- use_ema：*(bool)* 是否开启ema进行训练和推理
- ema_decay：*(float)* 开启ema时的衰减指数
- random_seed：*(int)* 随机种子
- use_fp16：*(bool)* 开启混合精度训练标志位
- loss_scaling：*(float)* 开启混合精度训练时的loss缩放因子

***可选字段***

- pretrain_model_path：*(str)* 预训练模型的载入路径，该路径下应包含存储模型参数的params文件夹
- pretrain_config_path：*(str)* 预训练模型的配置文件，json格式描述
- do_lower_case：*(bool)* 预处理阶段是否区分大小写
- 其他用户自定义字段

### 添加新任务

用户添加任务时，在准备好该任务的数据集后，需要完成如下3处开发工作：

***config模块***

位于`./config`目录。存放各个任务实例的配置文件，使用`yaml`格式描述。配置文件中的必选字段包括

- in_tokens：是否使用lod tensor的方式构造batch，当`in_tokens`为False时，使用padding方式构造batch。
- batch_size：每个训练或推理step所使用样本数。当`in_tokens`为True时，`batch_size`表示每个step所包含的tokens数量。

训练阶段包含的必选字段包括

- train_file：训练集文件路径
- mix_ratio：该任务的训练阶段采样权重（1.0代表与主任务采样次数的期望相同）

推理阶段包含的必选字段包括

- predict_file：测试集文件路径

此外用户可根据任务需要，自行定义其他超参数，该超参可在创建任务模型时被访问

***reader模块***

位于`./reader`目录下。完成数据集读取与处理。新增的reader应放置在`paradigm`目录下，且包含一个`get_input_shape`方法和`DataProcessor`类。

- **get_input_shape**: *(function)*  定义reader给backbone和task_paradigm生成的数据的shape和dtype，且需要同时返回训练和推理阶段的定义。
  - 输入参数
    - args: *(dict)* 解析后的任务配置
  - 返回值
    - train_input_shape: *(dict)* 包含backbone和task两个key，每个key对应的value为一个list，存储若干`(shape, dtype)`的元组
    - test_input_shape: *(dict)* 包含backbone和task两个key，每个key对应的value为一个list，存储若干`(shape, dtype)`的元组
- **DataProcessor**：*(class)*   定义数据集的载入、预处理和遍历
  - \_\_init\_\_: 构造函数，解析和存储相关参数，进行必要的初始化
    - 输入参数
      - args: *(dict)* 解析后的任务配置
    - 返回值
      - 无
  - data_generator: *(function)* 数据集的迭代器，被遍历时每次yield一个batch
    - 输入参数
      - phase: *(str)* 任务所处阶段，支持训练`train`和推理`predict`两种可选阶段
      - shuffle: *(bool)* 训练阶段是否进行数据集打乱
      - dev_count: *(int)* 可用的GPU数量或CPU数量
    - yield输出
      - tensors: (list) 根据get_input_shape中定义的任务backbone和task的所需输入shape和类型，来yield相应list结构的数据。其中被yield出的list的头部元素为backbone要求的输入数据，后续元素为task要求的输入数据
  - get_num_examples: *(function)* 返回样本数。注意由于滑动窗口等机制，实际运行时产生的样本数可能多于数据集中的样本数，这时应返回runtime阶段实际样本数
    - 输入参数
      - 无
    - 返回值
      - num_examples: *(int)* 样本数量

***task_paradigm模块***

位于`./paradigm`目录下。描述任务范式（如分类、匹配、阅读理解等）。新增的任务范式应放置在`paradigm`目录下，且应包含`compute_loss`和`create_model`两个必选方法，以及`postprocess`，`global_postprocess`两个可选方法。

- create_model：*(function)* 创建task模型
  - 输入参数
    - reader_input：*(nested Variables)* 数据输入层的输出，定义位于该任务的reader模块的`input_shape`方法中。输入的前N个元素为backbone的输入元素，之后的元素为task的输入。
    - base_model：*(Model)* 模型backbone的实例，可调用backbone的对外输出接口来实现task与backbone的连接。一般来说，backbone的输出接口最少包括`final_sentence_representation`和`final_word_representation`两个属性。
      - base_model.final_sentence_representation：*(Variable)* 输入文本的向量表示，shape为`[batch_size, hidden_size]`。
      - base_model.final_word_representation：*(Variable)* 输入文本中每个单词的向量表示，shape为`[batch_size, max_seqlen, hidden_size]`
    - is_training：*(bool)* 训练标志位
    - args：*(Argument)* 任务相关的参数配置，具体参数在config文件夹中定义
  - 返回值
    - output_tensors: *(dict)* 任务输出的tensor字典。训练阶段的输出字典中应至少包括num_seqs元素，num_seqs记录了batch中所包含的样本数（在输入为lod tensor（args.in_tokens被设置为True）时所以样本压平打平，没有样本数维度）
- compute_loss: *(function)* 计算task在训练阶段的batch平均损失值
  - 输入参数
    - output_tensors: *(dict)* 创建task时（调用`create_model`）时返回值，存储计算loss所需的Variables的名字到实例的映射
    - args：*(Argument)* 任务相关的参数配置，具体参数在config文件夹中定义
  - 返回值
    - total_loss：*(Variable)* 当前batch的平均损失值
- postprocess：*(function)* 推理阶段对每个推理step得到的fetch_results进行的后处理，返回对该step的每个样本的后处理结果
  - 输入参数
    - fetch_results：(dict) 当前推理step的fetch_dict中的计算结果，其中fetch_dict在create_model时定义并返回。
  - 返回值
    - processed_results：(list)当前推理step所有样本的后处理结果。
- global_postprocess: *(function)* 推理结束后，对全部样本的后处理结果进行最终处理（如结果保存、二次后处理等）
  - 输入参数
    - pred_buf：所有测试集样本的预测后处理结果
    - processor：任务的数据集载入与处理类DataProcessor的实例
    - mtl_args：多任务学习配置，在`mtl_conf.yaml`中定义
    - task_args：任务相关的参数配置，在`conf`文件夹中定义
  - 返回值
    - 无

***命名规范***

为新任务创建config，task_paradigm和reader文件后，应将三者文件名统一，且为reader文件的文件名增加`_reader`后缀。例如，用户添加的新任务名为yelp_senti，则config文件名为`yelp_senti.yaml`，放置于config文件夹下；task_paradigm文件名为`yelp_senti.py`，放置于paradigm文件夹下；reader文件名为`yelp_senti_reader.py`，放置于reader文件夹下。

***One-to-One模式（任务层共享）***

框架默认使用one-to-many的模式进行多任务训练，即多任务共享encoder，不共享输出层。该版本同时支持one-to-one模式，即多任务同时共享encoder和输出层（模型参数完全共享，但是有不同的数据源）。该模式通过config文件命名的方式开启，具体流程如下。

```
1. mtl_config.yaml下用户配置任务相关的名称，如main_task: "reading_comprehension"
2. 如果一个任务的数据集是多个来源，请在configs下对同一个任务添加多个任务配置，如任务为"reading_comprehension"有两个数据集需要训练，且每个batch内的数据都来自同一数据集，则需要添加reading_comprehension.name1.yaml和reading_comprehension.name2.yaml两个配置文件，其中name1和name2用户可根据自己需求定义名称，框架内不限定名称定义；
3. 启动多任务学习：sh run.sh
```

## License
This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE).

## 许可证书
此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](https://github.com/PaddlePaddle/models/blob/develop/LICENSE)许可认证。



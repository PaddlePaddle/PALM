
# 多任务学习框架PaddlePALM

# 安装
pip install paddlepalm

# 使用

### 1. 创建任务实例

使用yaml格式描述任务实例，每个任务实例中的必选字段包括

- train_file: 训练集文件路径
- reader: 数据集载入与处理工具名，框架预置reader列表见[这里](https://www.baidu.com/)
- backbone: 骨架模型名，框架预置reader列表见[这里](https://www.baidu.com/)
- paradigm: 任务范式(类型)名，框架预置paradigm列表见[这里](https://www.baidu.com/)

### 2. 完成训练配置

使用yaml格式完成配置多任务学习中的相关参数，如指定任务实例及其相关的主辅关系、参数复用关系、采样权重等

### 3. 开始训练

```python

import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config.yaml', task_dir='task_instance')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()
```

### 4. 预测

用户可在训练结束后直接调用pred接口对某个目标任务进行预测

示例：
```python
import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller(config_path='config.yaml', task_dir='task_instance')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()
    controller.pred('mrqa')
```

也可新建controller直接预测

```python
import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller(config_path='config.yaml', task_dir='task_instance')
    controller.pred('mrqa', infermodel_path='output_model/firstrun2/infer_model')
```


# 运行机制

### 多任务学习机制
pass 

### 训练终止机制

- 默认的设置：
  - **所有target任务达到目标训练步数后多任务学习停止**
  - 未设置成target任务的任务（即辅助任务）不会影响训练终止与否，只是担任”陪训“的角色
  - 注：默认所有的任务都是target任务，用户可以通过`target_tag`来标记目标/辅助任务
  - 每个目标任务的目标训练步数由num_epochs和mix_ratio计算得到

### 保存机制

- 默认的设置：
  - 训练过程中，保存下来的模型分为checkpoint (ckpt)和inference model (infermodel)两种：
    - ckpt保存的是包含所有任务的总计算图（即整个多任务学习计算图），用于训练中断恢复
    - infermodel保存的是某个目标任务的推理计算图和推理依赖的相关配置
  - 对于每个target任务，训练到预期的步数后自动保存inference model，之后不再保存。（注：保存inference model不影响ckpt的保存）
- 用户可改配置
  - 使用`save_ckpt_every_steps`来控制保存ckpt的频率，默认不保存
  - 每个task instance均可使用`save_infermodel_every_steps`来控制该task保存infermodel的频率，默认为-1，即只在达到目标训练步数时保存一下




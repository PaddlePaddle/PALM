## Example 3: Tagging
This task is a named entity recognition task. The following sections detail model preparation, dataset preparation, and how to run the task.

### Step 1: Prepare Pre-trained Models & Datasets

#### Pre-trianed Model

The pre-training model of this mission is: [ERNIE-v1-zh-base](https://github.com/PaddlePaddle/PALM/tree/r0.3-api).

Make sure you have downloaded the required pre-training model in the current folder.


#### Dataset

This task uses the `MSRA-NER(SIGHAN2006)` dataset. 

Download dataset:
```shell
python download.py
```

If everything goes well, there will be a folder named `data/`  created with all the datas in it.

The data should have 2 fields,  `text_a  label`, with tsv format. Here is some example datas:

 ```
text_a  label
在 这 里 恕 弟 不 恭 之 罪 ， 敢 在 尊 前 一 诤 ： 前 人 论 书 ， 每 曰 “ 字 字 有 来 历 ， 笔 笔 有 出 处 ” ， 细 读 公 字 ， 何 尝 跳 出 前 人 藩 篱 ， 自 隶 变 而 后 ， 直 至 明 季 ， 兄 有 何 新 出 ？    O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
相 比 之 下 ， 青 岛 海 牛 队 和 广 州 松 日 队 的 雨 中 之 战 虽 然 也 是 0 ∶ 0 ， 但 乏 善 可 陈 。   O O O O O B-ORG I-ORG I-ORG I-ORG I-ORG O B-ORG I-ORG I-ORG I-ORG I-ORG O O O O O O O O O O O O O O O O O O O
理 由 多 多 ， 最 无 奈 的 却 是 ： 5 月 恰 逢 双 重 考 试 ， 她 攻 读 的 博 士 学 位 论 文 要 通 考 ； 她 任 教 的 两 所 学 校 ， 也 要 在 这 段 时 日 大 考 。    O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
 ```



### Step 2: Train & Predict

The code used to perform this task is in `run.py`. If you have prepared the pre-training model and the data set required for the task, run:

```shell
python run.py
```

If you want to specify a specific gpu or use multiple gpus for training, please use **`CUDA_VISIBLE_DEVICES`**, for example:

```shell
CUDA_VISIBLE_DEVICES=0,1 python run.py
```

Note: On multi-gpu mode, PaddlePALM will automatically split each batch onto the available cards. For example, if the `batch_size` is set 64, and there are 4 cards visible for PaddlePALM, then the batch_size in each card is actually 64/4=16. If you want to change the `batch_size` or the number of gpus used in the example, **you need to ensure that the set batch_size can be divided by the number of cards.**

Some logs will be shown below:

```
step 1/652 (epoch 0), loss: 216.002, speed: 0.32 steps/s
step 2/652 (epoch 0), loss: 202.567, speed: 1.28 steps/s
step 3/652 (epoch 0), loss: 170.677, speed: 1.05 steps/s
```

After the run, you can view the saved models in the `outputs/` folder and the predictions in the `outputs/predict` folder. Here are some examples of predictions:


```
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 6, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
```

### Step 3: Evaluate

Once you have the prediction, you can run the evaluation script to evaluate the model:

```python
python evaluate.py
```

The evaluation results are as follows:

```
data num: 4636
f1: 0.9918
```

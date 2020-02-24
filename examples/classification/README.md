## Example 1: Classification
This task is a sentiment analysis task. The following sections detail model preparation, dataset preparation, and how to run the task.

### Step 1: Prepare Pre-trained Model & Dataset

#### Pre-trained Model

The pre-training model of this mission is: [ERNIE-v1-zh-base](https://github.com/PaddlePaddle/PALM/tree/r0.3-api).

Make sure you have downloaded the required pre-training model in the current folder.


#### Dataset

This example demonstrates with [ChnSentiCorp](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/ChnSentiCorp_htl_all), a Chinese sentiment analysis dataset.

Download dataset:
```shell
python download.py
```

If everything goes well, there will be a folder named `data/`  created with all the data files in it.

The dataset file (for training) should have 2 fields,  `text_a` and `label`, stored with [tsv](https://en.wikipedia.org/wiki/Tab-separated_values) format. Here shows an example:

```
label  text_a
0   当当网名不符实，订货多日不见送货，询问客服只会推托，只会要求用户再下订单。如此服务留不住顾客的。去别的网站买书服务更好。
0   XP的驱动不好找！我的17号提的货，现在就降价了100元，而且还送杀毒软件！
1   <荐书> 推荐所有喜欢<红楼>的红迷们一定要收藏这本书,要知道当年我听说这本书的时候花很长时间去图书馆找和借都没能如愿,所以这次一看到当当有,马上买了,红迷们也要记得备货哦!
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
step 1/154 (epoch 0), loss: 5.512, speed: 0.51 steps/s
step 2/154 (epoch 0), loss: 2.595, speed: 3.36 steps/s
step 3/154 (epoch 0), loss: 1.798, speed: 3.48 steps/s
```


After the run, you can view the saved models in the `outputs/` folder and the predictions in the `outputs/predict` folder. Here are some examples of predictions:


```
{"index": 0, "logits": [-0.2014336884021759, 0.6799028515815735], "probs": [0.29290086030960083, 0.7070990800857544], "label": 1}
{"index": 1, "logits": [0.8593899011611938, -0.29743513464927673], "probs": [0.7607553601264954, 0.23924466967582703], "label": 0}
{"index": 2, "logits": [0.7462944388389587, -0.7083730101585388], "probs": [0.8107157349586487, 0.18928426504135132], "label": 0}
```

### Step 3: Evaluate

Once you have the prediction, you can run the evaluation script to evaluate the model:

```shell
python evaluate.py
```

The evaluation results are as follows:

```
data num: 1200
accuracy: 0.9575, precision: 0.9634, recall: 0.9523, f1: 0.9578
```

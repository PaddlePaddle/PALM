## Examples 5: Predict(Classification)
This task is a sentiment analysis task. The following sections detail model preparation, dataset preparation, and how to run the task.

### Step 1: Prepare Pre-trained Models & Datasets

#### Pre-trianed Model

The pre-training model of this mission is: [ernie-zh-base](https://github.com/PaddlePaddle/PALM/tree/r0.3-api).

Make sure you have downloaded the required pre-training model in the current folder.


#### Dataset

This task uses the `chnsenticorp` dataset. 

Download dataset:
```shell
python download.py
```

If everything goes well, there will be a folder named `data/`  created with all the datas in it.

The data should have 2 fields,  `label  text_a`, with tsv format. Here is some example datas:

```
label  text_a
0   当当网名不符实，订货多日不见送货，询问客服只会推托，只会要求用户再下订单。如此服务留不住顾客的。去别的网站买书服务更好。
0   XP的驱动不好找！我的17号提的货，现在就降价了100元，而且还送杀毒软件！
1   <荐书> 推荐所有喜欢<红楼>的红迷们一定要收藏这本书,要知道当年我听说这本书的时候花很长时间去图书馆找和借都没能如愿,所以这次一看到当当有,马上买了,红迷们也要记得备货哦!
```

### Step 2: Predict

The code used to perform classification task is in `run.py`. If you have prepared the pre-training model and the data set required for the task, run:

```shell
python run.py
```

If you want to specify a specific gpu or use multiple gpus for predict, please use **`CUDA_VISIBLE_DEVICES`**, for example:

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python run.py
```


Some logs will be shown below:

```
step 1/154, speed: 0.51 steps/s
step 2/154, speed: 3.36 steps/s
step 3/154, speed: 3.48 steps/s
```


After the run, you can view the predictions in the `outputs/predict` folder. Here are some examples of predictions:


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

The evaluation results are as follows:  (need to update)

```
precision: 0.956666666667, recall: 0.949013157895, f1: 0.95688225039
```

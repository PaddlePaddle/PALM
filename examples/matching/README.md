## Example 2: Matching
This task is a sentence pair matching task. The following sections detail model preparation, dataset preparation, and how to run the task with PaddlePALM.

### Step 1: Prepare Pre-trained Models & Datasets

#### Download Pre-trained Model

The pre-training model of this mission is: [ERNIE-v2-en-base](https://github.com/PaddlePaddle/PALM/tree/r0.3-api).

Make sure you have downloaded the required pre-training model in the current folder.


#### Dataset

Here takes the [Quora Question Pairs](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset as the testbed for matching.

Download dataset:
```shell
python download.py
```

After the dataset is downloaded, you should convert the data format for training:
```shell
python process.py data/quora_duplicate_questions.tsv data/train.tsv data/test.tsv
```

If everything goes well, there will be a folder named `data/`  created with all the converted datas in it.

The dataset file (for training) should have 3 fields,  `text_a`, `text_b` and `label`, stored with [tsv](https://en.wikipedia.org/wiki/Tab-separated_values) format. Here shows an example:

```
text_a  text_b  label
How can the arrangement of corynebacterium xerosis be described?  How would you describe waves? 0
How do you fix a Google Play Store account that isn't working?  What can cause the Google Play store to not open? How are such probelms fixed?  1
Which is the best earphone under 1000?  What are the best earphones under 1k? 1
What are the differences between the Dell Inspiron 3000, 5000, and 7000 series laptops? "Should I buy an Apple MacBook Pro 15"" or a Dell Inspiron 17 5000 series?" 0
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
step 20/49087 (epoch 0), loss: 1.079, speed: 3.48 steps/s
step 40/49087 (epoch 0), loss: 1.251, speed: 5.18 steps/s
step 60/49087 (epoch 0), loss: 1.193, speed: 5.04 steps/s
```


After the run, you can view the saved models in the `outputs/` folder and the predictions in the `outputs/predict` folder. Here are some examples of predictions:


```
{"index": 0, "logits": [-0.32688724994659424, -0.8568955063819885], "probs": [0.629485011100769, 0.3705149292945862], "label": 0}
{"index": 1, "logits": [-0.2735646963119507, -0.7983021140098572], "probs": [0.6282548904418945, 0.37174513936042786], "label": 0}
{"index": 2, "logits": [-0.3381381630897522, -0.8614270091056824], "probs": [0.6279165148735046, 0.37208351492881775], "label": 0}
```

### Step 3: Evaluate

Once you have the prediction, you can run the evaluation script to evaluate the model:

```shell
python evaluate.py
```

The evaluation results are as follows:

```
data num: 4300
accuracy: 0.8619, precision: 0.8061, recall: 0.8377, f1: 0.8216
```

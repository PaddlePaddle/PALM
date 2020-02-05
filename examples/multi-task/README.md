## Examples 6: Multi-Task Slot Filling
This task is a slot filling task. During training, the task uses intent determination task to assist in training slot filling model. The following sections detail model preparation, dataset preparation, and how to run the task.

### Step 1: Prepare Pre-trained Models & Datasets

#### Pre-trianed Model

The pre-training model of this mission is: [ernie-en-base](https://github.com/PaddlePaddle/PALM/tree/r0.3-api).

Make sure you have downloaded the required pre-training model in the current folder.


#### Dataset

This task uses the `Airline Travel Information System` dataset. 
 
Download dataset:
```shell
python download.py
```

After the dataset is downloaded, you should convert the data format for training:
```shell
python process.py
```

If everything goes well, there will be a folder named `data/atis/`  created with all the datas in it.

Here is some example datas:

`data/atis/atis_slot/train.tsv` :
```
text_a	label
iwanttoflyfrombostonat838amandarriveindenverat1110inthemorning	OOOOOB-fromloc.city_nameOB-depart_time.timeI-depart_time.timeOOOB-toloc.city_nameOB-arrive_time.timeOOB-arrive_time.period_of_day
whatflightsareavailablefrompittsburghtobaltimoreonthursdaymorning	OOOOOB-fromloc.city_nameOB-toloc.city_nameOB-depart_date.day_nameB-depart_time.period_of_day
whatisthearrivaltimeinsanfranciscoforthe755amflightleavingwashington	OOOB-flight_timeI-flight_timeOB-fromloc.city_nameI-fromloc.city_nameOOB-depart_time.timeI-depart_time.timeOOB-fromloc.city_name
cheapestairfarefromtacomatoorlando	B-cost_relativeOOB-fromloc.city_nameOB-toloc.city_name
```

`data/atis/atis_intent/train.tsv` :
```
label	text_a
0	i want to fly from boston at 838 am and arrive in denver at 1110 in the morning
0	what flights are available from pittsburgh to baltimore on thursday morning
1	what is the arrival time in san francisco for the 755 am flight leaving washington
2	cheapest airfare from tacoma to orlando
```

### Step 2: Train & Predict

The code used to perform this task is in `run.py`. If you have prepared the pre-training model and the data set required for the task, run:

```shell
python run.py
```

If you want to specify a specific gpu or use multiple gpus for training, please use **`CUDA_VISIBLE_DEVICES`**, for example:

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python run.py
```

Some logs will be shown below:

```
global step: 5,   slot: step 3/309 (epoch 0), loss: 68.965, speed: 0.58 steps/s
global step: 10, intent: step 3/311 (epoch 0), loss: 3.407, speed: 8.76 steps/s
global step: 15,   slot: step 12/309 (epoch 0), loss: 54.611, speed: 1.21 steps/s
global step: 20, intent: step 7/311 (epoch 0), loss: 3.487, speed: 10.28 steps/s
```


After the run, you can view the saved models in the `outputs/` folder.


If you want to use the trained model to predict the `atis_slot` data, run:

```shell
python predict.py
```

If you want to specify a specific gpu or use multiple gpus for predict, please use **`CUDA_VISIBLE_DEVICES`**, for example:

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python predict.py
```


After the run, you can view the predictions in the `outputs/predict` folder. Here are some examples of predictions:

```

[129, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 5, 19, 1, 1, 1, 1, 1, 21, 21, 68, 129]
[129, 1, 39, 37, 1, 1, 1, 1, 1, 2, 1, 5, 19, 1, 23, 3, 4, 129, 129, 129, 129, 129]
[129, 1, 39, 37, 1, 1, 1, 1, 1, 1, 2, 1, 5, 19, 129, 129, 129, 129, 129, 129, 129, 129]
[129, 1, 1, 1, 1, 1, 1, 14, 15, 1, 2, 1, 5, 19, 1, 39, 37, 129, 129, 129, 129, 129]
```

### Step 3: Evaluate

Once you have the prediction, you can run the evaluation script to evaluate the model:

```shell
python evaluate.py
```

The evaluation results are as follows:

```
precision: 0.894518453811, recall: 0.894323144105, f1: 0.894420788296
```

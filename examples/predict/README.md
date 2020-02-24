## Example 5: Prediction
This example demonstrates how to directly do prediction with PaddlePALM. You can either initialize the model from a checkpoint, a pretrained model or just randomly initialization. Here we reuse the task and data in example 1. Hence repeat the step 1 in example 1 to pretrain data. 

After you have prepared the pre-training model and the data set required for the task, run:

```shell
python run.py
```

If you want to specify a specific gpu or use multiple gpus for predict, please use **`CUDA_VISIBLE_DEVICES`**, for example:

```shell
CUDA_VISIBLE_DEVICES=0,1 python run.py
```

Note: On multi-gpu mode, PaddlePALM will automatically split each batch onto the available cards. For example, if the `batch_size` is set 64, and there are 4 cards visible for PaddlePALM, then the batch_size in each card is actually 64/4=16. If you want to change the `batch_size` or the number of gpus used in the example, **you need to ensure that the set batch_size can be divided by the number of cards.**


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

The evaluation results are as follows:

```
data num: 1200
accuracy: 0.4758, precision: 0.4730, recall: 0.3026, f1: 0.3691
```

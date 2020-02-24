## Example 4: Machine Reading Comprehension
This task is a machine reading comprehension task. The following sections detail model preparation, dataset preparation, and how to run the task.

### Step 1: Prepare Pre-trained Models & Datasets

#### Pre-trianed Model

The pre-training model of this mission is: [ERNIE-v1-zh-base](https://github.com/PaddlePaddle/PALM/tree/r0.3-api).

Make sure you have downloaded the required pre-training model in the current folder.


#### Dataset

This task uses the `CMRC2018` dataset. `CMRC2018` is an evaluation conducted by Chinese information society. The task of evaluation is to extract reading comprehension.

Download dataset:
```shell
python download.py
```

If everything goes well, there will be a folder named `data/`  created with all the datas in it.

Here is some example datas:

 ```json
"paragraphs": [
         {
           "id": "TRAIN_36",
           "context": "NGC 6231是一个位于天蝎座的疏散星团，天球座标为赤经16时54分，赤纬-41度48分，视觉观测大小约45角分，亮度约2.6视星等，距地球5900光年。NGC 6231年龄约为三百二十万年，是一个非常年轻的星团，星团内的最亮星是5等的天蝎座 ζ1星。用双筒望远镜或小型望远镜就能看到个别的行星。NGC 6231在1654年被意大利天文学家乔瓦尼·巴蒂斯特·霍迪尔纳（Giovanni Battista Hodierna）以Luminosae的名字首次纪录在星表中，但是未见记载于夏尔·梅西耶的天体列表和威廉·赫歇尔的深空天体目录。这个天体在1678年被爱德蒙·哈雷（I.7）、1745年被夏西亚科斯（Jean-Phillippe Loys de Cheseaux）（9）、1751年被尼可拉·路易·拉卡伊（II.13）分别再次独立发现。",
           "qas": [
             {
               "question": "NGC 6231的经纬度是多少？",
               "id": "TRAIN_36_QUERY_0",
               "answers": [
                 {
                   "text": "赤经16时54分，赤纬-41度48分",
                   "answer_start": 27
                 }
               ]
             }
         }
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
step 1/1515 (epoch 0), loss: 6.251, speed: 0.31 steps/s
step 2/1515 (epoch 0), loss: 6.206, speed: 0.80 steps/s
step 3/1515 (epoch 0), loss: 6.172, speed: 0.86 steps/s
```


After the run, you can view the saved models in the `outputs/` folder and the predictions in the `outputs/predict` folder. Here are some examples of predictions:


```json
{
    "DEV_0_QUERY_0": "光 荣 和 ω-force 开 发", 
    "DEV_0_QUERY_1": "任 天 堂 游 戏 谜 之 村 雨 城", 
    "DEV_0_QUERY_2": "战 史 演 武 」&「 争 霸 演 武 」。", 
    "DEV_1_QUERY_0": "大 陆 传 统 器 乐 及 戏 曲 里 面 常 用 的 打 击 乐 记 谱 方 法 ， 以 中 文 字 的 声 音 模 拟 敲 击 乐 的 声 音 ， 纪 录 打 击 乐 的 各 种 不 同 的 演 奏 方 法 。", 
    "DEV_1_QUERY_1": "「 锣 鼓 点", 
    "DEV_1_QUERY_2": "锣 鼓 的 运 用 有 约 定 俗 成 的 程 式 ， 依 照 角 色 行 当 的 身 份 、 性 格 、 情 绪 以 及 环 境 ， 配 合 相 应 的 锣 鼓 点", 
    "DEV_1_QUERY_3": "鼓 、 锣 、 钹 和 板 四 类 型", 
    "DEV_2_QUERY_0": "364.6 公 里", 
}
```

### Step 3: Evaluate

#### Library Dependencies
Before the evaluation, you need to install `nltk` and download the `punkt` tokenizer for nltk:

```shell
pip insall nltk
python -m nltk.downloader punkt
```

#### Evaluate
You can run the evaluation script to evaluate the model:

```shell
python evaluate.py
```

The evaluation results are as follows:

```
data_num: 3219
em_sroce: 0.6434, f1: 0.8518
```

## Example 6: Joint Training in Dialogue
This task is a slot filling task. During training, the task uses intent determination task to assist in training slot filling model. The following sections detail model preparation, dataset preparation, and how to run the task.

### Step 1: Prepare Pre-trained Models & Datasets

#### Pre-trianed Model

The pre-training model of this mission is: [ERNIE-v2-en-base](https://github.com/PaddlePaddle/PALM/tree/r0.3-api).

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
i want to fly from boston at 838 am and arrive in denver at 1110 in the morning 	O O O O O B-fromloc.city_name O B-depart_time.time I-depart_time.time O O O B-toloc.city_name O B-arrive_time.time O O B-arrive_time.period_of_day 
what flights are available from pittsburgh to baltimore on thursday morning 	O O O O O B-fromloc.city_name O B-toloc.city_name O B-depart_date.day_name B-depart_time.period_of_day 
what is the arrival time in san francisco for the 755 am flight leaving washington 	O O O B-flight_time I-flight_time O B-fromloc.city_name I-fromloc.city_name O O B-depart_time.time I-depart_time.time O O B-fromloc.city_name 
cheapest airfare from tacoma to orlando 	B-cost_relative O O B-fromloc.city_name O B-toloc.city_name 
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
CUDA_VISIBLE_DEVICES=0,1 python run.py
```

Note: On multi-gpu mode, PaddlePALM will automatically split each batch onto the available cards. For example, if the `batch_size` is set 64, and there are 4 cards visible for PaddlePALM, then the batch_size in each card is actually 64/4=16. If you want to change the `batch_size` or the number of gpus used in the example, **you need to ensure that the set batch_size can be divided by the number of cards.**

Some logs will be shown below:

```
global step: 5,   slot: step 3/309 (epoch 0), loss: 68.965, speed: 0.58 steps/s
global step: 10, intent: step 3/311 (epoch 0), loss: 3.407, speed: 8.76 steps/s
global step: 15,   slot: step 12/309 (epoch 0), loss: 54.611, speed: 1.21 steps/s
global step: 20, intent: step 7/311 (epoch 0), loss: 3.487, speed: 10.28 steps/s
```


After the run, you can view the saved models in the `outputs/` folder.


If you want to use the trained model to predict the `atis_slot & atis_intent` data, run:

```shell
python predict-slot.py
python predict-intent.py
```

If you want to specify a specific gpu or use multiple gpus for predict, please use **`CUDA_VISIBLE_DEVICES`**, for example:

```shell
CUDA_VISIBLE_DEVICES=0,1 python predict-slot.py
CUDA_VISIBLE_DEVICES=0,1 python predict-intent.py
```

Note: On multi-gpu mode, PaddlePALM will automatically split each batch onto the available cards. For example, if the `batch_size` is set 64, and there are 4 cards visible for PaddlePALM, then the batch_size in each card is actually 64/4=16. If you want to change the `batch_size` or the number of gpus used in the example, **you need to ensure that the set batch_size can be divided by the number of cards.**

After the run, you can view the predictions in the `outputs/predict-slot` folder and `outputs/predict-intent` folder. Here are some examples of predictions:

`atis_slot`:
```
[129, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 5, 19, 1, 1, 1, 1, 1, 21, 21, 68, 129]
[129, 1, 39, 37, 1, 1, 1, 1, 1, 2, 1, 5, 19, 1, 23, 3, 4, 129, 129, 129, 129, 129]
[129, 1, 39, 37, 1, 1, 1, 1, 1, 1, 2, 1, 5, 19, 129, 129, 129, 129, 129, 129, 129, 129]
[129, 1, 1, 1, 1, 1, 1, 14, 15, 1, 2, 1, 5, 19, 1, 39, 37, 129, 24, 129, 129, 129]
```

`atis_intent`:
```
{"index": 0, "logits": [9.938603401184082, -0.3914794623851776, -0.050973162055015564, -1.0229418277740479, 0.04799401015043259, -0.9632213115692139, -0.6427211761474609, -1.337939739227295, -0.7969412803649902, -1.4441455602645874, -0.6339573264122009, -1.0393054485321045, -0.9242327213287354, -1.9637483358383179, 0.16733427345752716, -0.5280354619026184, -1.7195699214935303, -2.199411630630493, -1.2833174467086792, -1.3081035614013672, -1.6036226749420166, -1.8527079820632935, -2.289180040359497, -2.267214775085449, -2.2578916549682617, -2.2010505199432373], "probs": [0.999531626701355, 3.26210938510485e-05, 4.585415081237443e-05, 1.7348344044876285e-05, 5.06243304698728e-05, 1.8415948943584226e-05, 2.5373808966833167e-05, 1.266065828531282e-05, 2.174747896788176e-05, 1.1384962817828637e-05, 2.5597169951652177e-05, 1.7066764485207386e-05, 1.914815220516175e-05, 6.771284006390488e-06, 5.70411684748251e-05, 2.8457265216275118e-05, 8.644025911053177e-06, 5.349628736439627e-06, 1.3371440218179487e-05, 1.3044088518654462e-05, 9.706698619993404e-06, 7.5665011536329985e-06, 4.890325726591982e-06, 4.99892985317274e-06, 5.045753368904116e-06, 5.340866664482746e-06], "label": 0}
{"index": 1, "logits": [0.8863624930381775, -2.232290506362915, 8.191509246826172, -0.03161466494202614, -0.9149583578109741, -2.172696352005005, -0.3937145471572876, -0.3954394459724426, 1.5333592891693115, 0.8630291223526001, -0.9684226512908936, -2.722721815109253, -0.0060247331857681274, -0.9865402579307556, 1.6328885555267334, 0.3972966969013214, 0.27919167280197144, -1.4911551475524902, -0.9552251696586609, -0.9169244170188904, -0.810670793056488, -1.5118697881698608, -2.0140435695648193, -1.6299077272415161, -1.8589974641799927, -2.07601261138916], "probs": [0.0006675600307062268, 2.9517297662096098e-05, 0.9932880997657776, 0.0002665741485543549, 0.0001102013120544143, 3.132982965325937e-05, 0.00018559220188762993, 0.00018527248175814748, 0.0012749042361974716, 0.0006521637551486492, 0.00010446414671605453, 1.8075270418194123e-05, 0.0002734838053584099, 0.00010258861584588885, 0.0014083238784223795, 0.00040934717981144786, 0.00036374686169438064, 6.193659646669403e-05, 0.00010585198469925672, 0.00010998480865964666, 0.0001223145518451929, 6.0666847275570035e-05, 3.671637750812806e-05, 5.391232480178587e-05, 4.287416595616378e-05, 3.4510172554291785e-05], "label": 0}
{"index": 2, "logits": [9.789957046508789, -0.1730862706899643, -0.7198237776756287, -1.0460278987884521, 0.23521068692207336, -0.5075851678848267, -0.44724929332733154, -1.2945927381515503, -0.6984466314315796, -1.8749892711639404, -0.4631594121456146, -0.6256799697875977, -1.0252169370651245, -1.951456069946289, -0.17572557926177979, -0.6771697402000427, -1.7992591857910156, -2.1457295417785645, -1.4203097820281982, -1.4963451623916626, -1.692310094833374, -1.9219486713409424, -2.2533645629882812, -2.430952310562134, -2.3094685077667236, -2.2399914264678955], "probs": [0.9994625449180603, 4.708383130491711e-05, 2.725377635215409e-05, 1.9667899323394522e-05, 7.082601223373786e-05, 3.3697724575176835e-05, 3.579350595828146e-05, 1.5339375750045292e-05, 2.784266871458385e-05, 8.58508519741008e-06, 3.522853512549773e-05, 2.9944207199150696e-05, 2.0081495677004568e-05, 7.953084605105687e-06, 4.695970710599795e-05, 2.8441407266655006e-05, 9.26048778637778e-06, 6.548832516273251e-06, 1.3527245755540207e-05, 1.2536826943687629e-05, 1.030578732752474e-05, 8.19125762063777e-06, 5.880556273041293e-06, 4.923717369820224e-06, 5.559719284065068e-06, 5.9597273320832755e-06], "label": 0}
{"index": 3, "logits": [9.787659645080566, -0.6223222017288208, -0.03971472755074501, -1.038114070892334, 0.24018540978431702, -0.8904737830162048, -0.7114139795303345, -1.2315020561218262, -0.5120854377746582, -1.4273980855941772, -0.44618460536003113, -1.0241562128067017, -0.9727545380592346, -1.8587366342544556, 0.020689941942691803, -0.6228570342063904, -1.6020199060440063, -2.130260467529297, -1.370570421218872, -1.40530526638031, -1.6782578229904175, -1.94076669216156, -2.2038567066192627, -2.336832284927368, -2.268157720565796, -2.140028953552246], "probs": [0.9994485974311829, 3.0113611501292326e-05, 5.392447565100156e-05, 1.986949791898951e-05, 7.134198676794767e-05, 2.303065048181452e-05, 2.7546762794372626e-05, 1.6375688574044034e-05, 3.362310235388577e-05, 1.3462414244713727e-05, 3.591357381083071e-05, 2.0148761905147694e-05, 2.12115264730528e-05, 8.74570196174318e-06, 5.728216274292208e-05, 3.0097504350123927e-05, 1.1305383850412909e-05, 6.666126409982098e-06, 1.4249604646465741e-05, 1.3763145034317859e-05, 1.0475521776243113e-05, 8.056933438638225e-06, 6.193143690325087e-06, 5.422014055511681e-06, 5.807448815176031e-06, 6.601325367228128e-06], "label": 0}
```

### Step 3: Evaluate

Once you have the prediction, you can run the evaluation script to evaluate the model:

```shell
python evaluate-slot.py
python evaluate-intent.py
```

The evaluation results are as follows:

`atis_slot`:
```
data num: 891
f1: 0.8934
```

`atis_intent`:
```
data num: 893
accuracy: 0.7088, precision: 1.0000, recall: 1.0000, f1: 1.0000
```

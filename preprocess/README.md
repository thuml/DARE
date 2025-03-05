# Download and Prepare Dataset

## Download

We use the **Taobao** and **Tmall** datasets. 

You can download the **Taobao** dataset from this [link](https://tianchi.aliyun.com/dataset/649). 

- Download ``UserBehavior.csv.zip`` (906M).
- Unzip it to obtain ``UserBehavior.csv`` (3.5G).
- Place it in: ``data/taobao/UserBehavior.csv``.

You can download the **Tmall** dataset from this [link](https://tianchi.aliyun.com/dataset/42).

- Download ``data_format1 .zip`` (361M). 
- Note: There is an extra space in the filename, which may cause issues. It is recommended to rename it to ``data_format1.zip`` before extracting.
- Unzip it to obtain four files. Among them, we only need ``user_log_format1.csv`` (1.8G).
- Place it in: ``data/tmall/user_log_format1.csv``.

## Preprocessing
To our knowledge, some versions of numpy will raise error like *The requested array has an inhomogeneous shape after 1 dimensions.* 
We believe this issue has been resolved in our code. However, if you still encounter it, you can manually add ``dtype=object`` when reading or saving ``.npy`` files to fix it.

Taobao:

```
cd preprocess
python taobao_v4.py
```

After execution, the processed files will be stored in ``data/taobao``, including ``train_{0001~0048}_{data_num}.npy``, ``val.npy`` and ``test.npy``.

Since the training dataset is large, it is **split into multiple files** and loaded sequentially during training to **prevent out-of-memory (OOM)** errors.

Tmall:

```
python tmall_sort_log.py
python tmall_v4.py
```

The ``user_log_format1.csv`` file is not time-ordered. We first sort it before performing further processing.

Other preprocessing steps are similar to those for the Taobao dataset.



## Data Process Details

This section provides useful insights if you wish to modify or analyze the dataset further.

### data format

After pre-processing, each sample consists of the following features:

- uid: the ID of the user (not used in our code).
- target_iid: the item ID of the candidate.
- target_cid: the category ID of the candidate.
- label: whether the user will click the candidate.
- hist_iid: a list of item IDs that the user clicked in the history.
- hist_cid: a list of category IDs that the user clicked in the history.

### Understanding ``hist_iid`` and ``hist_cid``

- The interaction history is sorted by time. hist_iid[-1] is the most recent interaction; while hist_iid[0] is the earliest recorded interaction.
- Clip: the maximum length is 200. If a user has more than 200 historical interactions, only the most recent 200 are kept.
- Pad：If a user has fewer than 200 interactions, zeros are padded.

### train/val/test split

- We filter out users with fewer than 210 interactions.
- Suppose a user has 300 interactions ([1~300], heiger index means more recent), the train/val/test dataset will contain:
  - train: for j in range(19):  
    - i = 298 - 5 * j
    - given: [i-200 ~ i-1]
    - to predict: [i]
  - val:
    - given: [99 ~ 298]
    - to predict: [299]
  - test:
    - given: [100 ~ 299]
    - to predict: [300]

As we pointed out in our paper, In recommendation systems, the number of items is vast, but training data is relatively scarce, leading to underfitting.
To mitigate this, we extract multiple training samples per user.
However, this violates the "independent identical distribution" principle, slightly compromising data quality.

Thus, we finally make a trade-off and process the data like that, mainly for the two reason: 
for each user, there are 20 pieces of data visible in the training process; there is no padding in the test dataset. 
Experimental results show that this strategy significantly improves model performance.

It's not easy to balance data quality and data quantity. If you wish to follow our research, you can use our dataset setup or define your own.

### Candidate Sampling

- The negative candidates are randomly sampled. 
- A negative candidate must **not** have been clicked by the user in history.
- The sampling probability **is proportional to its frequency** in the entire dataset.
- There are twice as many negative candidates as positive ones.

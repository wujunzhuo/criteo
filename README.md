# criteo

### 拆分数据

[link](http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/)

```
mkdir data
cd data
split -d -l 500000 train.txt train_
```

### MinMax缩放、Ordinal编码

```
python transform.py
```

### 模型训练

```
python train.py
```
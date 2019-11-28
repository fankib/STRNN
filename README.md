# STRNN Implementation
[Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts](https://pdfs.semanticscholar.org/5bdf/0970034d0bb8a218c06ba3f2ddf97d29103d.pdf)  
Qiang Liu, Shu Wu, Liang Wang, Tieniu Tan (AAAI-16)

## Perfomance
validation for 3315/41925 users: (8%)
recall@1:  0.01297134238310709
recall@5:  0.03378582202111614
recall@10:  0.051583710407239816
recall@100:  0.14147812971342383
recall@1000:  0.2914027149321267
MAP 0.02549331942266631

## Usage

### 1. Preprocess
[Gowalla](http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz) is required at `../dataset/`.
```bash
$ python preprocess.py
```

### 2. Visualize Gowalla
```bash
$ python statistics.py
```

### 3. Experiment
Find the best hyperparameters (and preprocess the dataset) with:
```bash
$ python experiment.py
```

### 4. Train
```bash
$ python train.py
```



# STRNN Implementation
[Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts](https://pdfs.semanticscholar.org/5bdf/0970034d0bb8a218c06ba3f2ddf97d29103d.pdf)  
Qiang Liu, Shu Wu, Liang Wang, Tieniu Tan (AAAI-16)

## Perfomance
on evaluation

## Usage

### 1. Preprocess
The [Gowalla](http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz) is required at `../dataset/`.
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



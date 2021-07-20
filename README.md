# LSTM demo
## 实现矩阵序列预测

## Setup
All code was developed and tested on Win10x64 with Python 3.7, tensorflow 2.1.0 and keras-gpu 2.3.1.

You should have a GPU from nvidia.

You can setup a virtual environment to run the code like this:
```bash
conda create -n keras-gpu python==3.7
conda activate keras-gpu
conda install -c anaconda keras-gpu
```
Other requirements:
```text
matplotlib==3.4.2
numpy==1.19.1
pandas==1.1.5
```
## Try
```bash
python predict.py     # 默认网格尺寸4000,观察120s预测60s
```
## Train

```bash
python LSTM_def_s_gpu.py      # train and predict, dont save model. Please modify the code to save.
```

## Predict
Like Try:
```bash
python predict.py
```
or
```bash
python predict_norm.py  # load model with Normalization
```
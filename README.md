# Writer Verification (Line-level, Patch Pooling)

This project verifies whether two handwriting **lines** come from the **same writer**.
It uses a robust preprocessing pipeline + content-aware **patch pooling** and learns a writer-style embedding with a **triplet loss**.

## Dataset Layout (explicit)

```
./data/Train/<writer_id>/<doc_id>/*.jpg
./data/Test/<writer_id>/<doc_id>/*.jpg
```


## Create virtual envorntment

```
conda create --name "pt-gpu
```

```
conda activate pt-gpu
```


## Install

```bash
pip install -r requirements.txt
```



## Visualize how the preprocessing pipeline:

```bash
python -m preprocessor.view_preprocessor --file ./data/Test/1/1_1/1_1_1.jpg --show-patches
```



# Train

```bash
python -m modeling.train
```
- Uses `./Train` and splits writers into train/val (unless you specify `paths.val_dir`).
- Saves weights into `./trained_model`.

# Evaluate (on explicit `./Test`)

```bash
python -m modeling.evaluate
```

This builds same-writer pairs and balanced different-writer pairs, computes distances,
sweeps a threshold, and prints Accuracy, Precision/Recall/F1, FAR/FRR, and AUC.


## Predict on Two Lines

```bash
python -m use_model.predict --config config.yml --img_a ./data/Test/26/26_2/26_2_1.jpg --img_b ./data/Test/27/27_1/27_1_2.jpg
```

Outputs cosine distance and a decision using the threshold from `config.yml`.



## Notes
- Preprocessing is in `preprocessor/pipeline.py`.
- Model definitions live in `modeling/model_defs.py`.




---
---
---


# For faster training on NVIDIA GPU:



# Setup-NVIDIA-GPU-for-Deep-Learning

## Step 1: NVIDIA Video Driver

You should install the latest version of your GPUs driver. You can download drivers here:
 - [NVIDIA GPU Drive Download](https://www.nvidia.com/Download/index.aspx)

## Step 2: Visual Studio C++

You will need Visual Studio, with C++ installed. By default, C++ is not installed with Visual Studio, so make sure you select all of the C++ options.
 - [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)

## Step 3: Anaconda/Miniconda

You will need anaconda to install all deep learning packages
 - [Download Anaconda](https://www.anaconda.com/download/success)

## Step 4: CUDA Toolkit

 - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

## Step 5: cuDNN

 - [Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)


## Step 6: Install PyTorch 

 - [Install PyTorch](https://pytorch.org/get-started/locally/)




## Finally run the following script to test your GPU

```bash
python -m modeling.gpu
```



## For training this project the following was used:

**CUDA VERSION: 12.8**

**cuDNN Version: 8.9.7**
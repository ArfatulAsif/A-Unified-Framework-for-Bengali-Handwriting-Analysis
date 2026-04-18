# 3. Methodology

In this section, we detail the complete pipeline of our proposed writer identification framework. We begin by outlining our dataset preparation and preprocessing strategies, followed by the core architecture of our Hierarchical Patch Encoder, the comparative State-of-the-Art (SOTA) architectures considered, and our training objectives and hyperparameter setups. Finally, we break down the framework into four distinct operational pipelines: Line and Page-Level Inference (Verification), Writer-Based Document Retrieval, Unsupervised Document Clustering, and Sequential Multi-Writer Segmentation.

All computational tasks, including data preprocessing and model training, were executed using Python 3.9.13 on a single NVIDIA RTX 3050 GPU, utilizing PyTorch 2.8.0, CUDA 12.8, and cuDNN 8.9.7.

## 3.1 Dataset Collection and Preparation

To construct a comprehensive, highly variable, and inclusive dataset, we aggregated data from three primary sources:

1. **BN-HTRd:** Provided existing line-segmented images and full handwritten pages from 237 distinct writers.
2. **WBSUBNdb_text:** Contributed handwritten pages from 188 writers. We manually processed these pages through our line-segmentation pipeline to create dedicated line-level datasets, visually evaluating and verifying each segmented line to ensure high ground-truth quality.
3. **Custom Dataset:** We compiled an additional, highly challenging set featuring 10 specialized writers. This custom subset was specifically curated to introduce severe real-world impurities, including low-quality mobile phone scans, heavy shadows, and varied camera orientations.

Throughout the aggregation process, we thoroughly reviewed the data to ensure it accurately reflected the unpredictable nature of real-world physical documents, explicitly keeping variations such as exceptionally long and short lines, diverse page colors, and curved or slanted handwriting.

**Dataset Summary:**
* **Total Writers:** 435
* **Total Handwritten Pages:** 2,825 (Average of ~7 pages per writer)
* **Total Segmented Lines:** 29,268 (Average of ~70 lines per writer)

## 3.2 Writer-Disjoint Split and Evaluation Protocol

To ensure our model learns generalized, writer-independent style features rather than memorizing specific handwriting traits from the training set, we adhered to a strict, zero-shot, writer-disjoint evaluation protocol. This guarantees the model will perform reliably on unseen writers in real-world scenarios.

The overarching dataset of 435 writers was strategically partitioned into the non-overlapping subsets described in **Table I**.

### **Table I: Dataset Partitioning and Task Allocation (435 Writers)**

| Pipeline / Task                          | Phase      | Writer IDs | Data Level         |
| :--------------------------------------- | :--------- | :--------- | :----------------- |
| **Patch Encoder**                        | Training   | 1–371      | Line               |
| **Verification**                         | Tuning     | 372–403    | Line & Page        |
| **Retrieval**                            | Evaluation | 404–415    | Page               |
| **Clustering**                           | Tuning     | 372–403    | Page               |
| **Clustering**                           | Evaluation | 404–415    | Page               |
| **Sequential Multi-Writer Segmentation** | Tuning     | 395–420    | Synthesized Pages* |
| **Sequential Multi-Writer Segmentation** | Evaluation | 421–435    | Synthesized Pages* |

---


_> Note: Synthesized pages for segmentation were created by cropping and vertically merging segments from 1–4 writers to simulate intrinsic plagiarism; used for tuning sequential clustering/smoothing and evaluating segmentation via SER._



**Standardized Ablation Cohort:**
Finally, to conduct fair State-of-the-Art (SOTA) comparisons and detailed ablation studies without exhausting our primary test sets, we isolated a standardized mini-cohort of 100 writers. This cohort was strictly partitioned into 80 train/validation writers and 20 test writers for line-level tasks, with 10 of those test writers reserved for page-level evaluations.




## 3.3 Preprocessing Handwriting Lines

To ensure our feature extractor learns robust stylistic representations rather than superficial dataset artifacts (e.g., uneven lighting, arbitrary margins, or scanner noise), we subject each segmented handwriting line to a stringent, multi-stage signal conditioning and geometric normalization pipeline, as illustrated in **Figure 2**.



<img src="images/preprocessing.png">


Let the raw grayscale handwriting line be denoted as $I_{raw}$. First, to maximize ink visibility against degraded backgrounds, we apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to produce an enhanced image, $I_{clahe}$. 

To eliminate arbitrary background margins, we generate a binary adaptive text mask $M$ from $I_{clahe}$. By computing the smoothed 1D pixel projections along the horizontal and vertical axes of $M$, we isolate the tightest bounding coordinates $[y_0, y_1]$ and $[x_0, x_1]$ that contain valid ink. We then crop the enhanced image to these coordinates to yield the content-only image:
<br>

$$I_{crop} = I_{clahe}[y_0:y_1, x_0:x_1]$$

Next, we normalize the physical scale of the handwriting. $I_{crop}$ is resized to a fixed target height $H_{target} = 128$ pixels while strictly preserving its original aspect ratio, resulting in a resized image $I_{res}$ of width $W_{new}$. To unify the tensor dimensions for batch processing without distorting the handwriting geometry, $I_{res}$ is placed onto a fixed-size canvas $C \in \mathbb{R}^{H_{target} \times W_{max}}$, where the maximum width $W_{max} = 1580$. During training, we actuate spatial data augmentation by placing $I_{res}$ at a random horizontal offset (`place_train="random"`), whereas during evaluation, we center it deterministically (`place_eval="center"`). The canvas is then normalized to a continuous float range $[0, 1]$.

#### **Patch Extraction**

Because handwriting lines vary drastically in length, we model each line as a sequence of localized, overlapping visual patches. Let a single patch be mathematically designated as $p_k \in \mathbb{R}^{P \times P \times 1}$, where the patch size $P = 128$. 

We extract these patches using a sliding window approach along the horizontal axis of the valid content region $[x_{start}, x_{end}]$ of the canvas $C$. A patch $p_k$ at step $k$ is extracted starting at coordinate $x_k = x_{start} + k \cdot S$, where the stride length $S = 56$:

<br>

$$p_k = C[:, x_k : x_k + P]$$

To prevent the model from processing empty background space, we enforce a strict foreground density constraint. A patch $p_k$ is only appended to the final sequence if its ink ratio exceeds a minimum threshold $\tau$. By thresholding $p_k$ via Otsu's method, we define the foreground indicator function $f(p_k)$, and strictly enforce:

<br>

$$f(p_k) \geq \tau_{min}$$

<br>

where $\tau_{min} = 0.04$. The final preprocessed output for a single handwriting line is the sequence of valid, highly-dense patches $P_{line} = \{p_1, p_2, \dots, p_N\}$, which is subsequently passed to the patch encoder. 


*(Note: An ablation study validating the impact of this preprocessing pipeline is provided in Section 4).*


## 3.4 Patch Encoder Architecture

To extract highly discriminative features from the preprocessed handwriting patches, we developed and evaluated multiple convolutional and transformer-based architectures. Primarily, we created a custom, lightweight network—the **DPE-Net (Dual-Path Patch Encoder)** architecture—designed specifically to be computationally efficient, fast to train, and capable of rapid inference while maintaining robust accuracy metrics. Alongside our custom architecture, we extensively evaluated several modern State-of-the-Art (SOTA) networks, among which the pretrained **FasterNet-T0** emerged as a highly performant benchmark. 

To rigorously assess the practical viability of these models for large-scale forensic analysis, our evaluation prioritizes the balance between computational efficiency and predictive power. In real-world scenarios involving massive document databases, high-throughput processing is just as critical as absolute accuracy. Therefore, we benchmarked **computational footprint (Total Parameters, FLOPs)** and **processing speed (Inference Latency per Line)** to validate scalability, while tracking **Line-Level Accuracy and AUC** to ensure robust verification performance. The exhaustive comparative benchmarking of these architectures is detailed in **Section 4.1 (Table I)**, and the isolated ablation study of our custom architecture's internal components is provided in **Section 4.2 (Table III).**



### 3.4.1 DPE-Net (Dual-Path Patch Encoder) Model Architecture

To capture both the fine-grained nuances of ink deposition and the broader geometric flow of handwriting, we engineered a custom Dual-Path Convolutional Neural Network, , as shown in **Figure 3.** Rather than relying on deep, parameter-heavy sequential layers, this architecture utilizes two parallel expert branches that process a shared initial feature map.

The network receives an input patch tensor of dimensions $128 \times 128 \times 1$. The processing pipeline is structured as follows:

1. **Shared Stem:** The input patch first passes through a downsampling stem comprising a $3 \times 3$ convolutional layer with a stride of 2 and padding of 1, followed by 2D Batch Normalization and an in-place ReLU activation. This reduces the spatial dimensions to $64 \times 64$ while projecting the features into 32 channels.
2. **Standard Branch (Local Textures):** The first parallel pathway consists of two sequential blocks of $3 \times 3$ convolutions (stride 2, padding 1), Batch Normalization, and ReLU activations. This standard convolutional branch is specifically tuned to capture highly localized micro-textures and intricate stroke variations, outputting a 64-channel feature map.
3. **Dilated Branch (Long-Range Strokes):** The second parallel pathway mirrors the structural depth of the first but utilizes dilated convolutions. It applies two sequential blocks of $3 \times 3$ convolutions with a stride of 2, a padding of 2, and a dilation rate of 2. This dilation artificially expands the receptive field of the $3 \times 3$ kernel to an effective $5 \times 5$ spatial area without increasing the trainable parameter count. This branch is designed to trace long-range stroke connectivity and continuous spatial patterns, yielding a secondary 64-channel feature map.
4. **Feature Fusion and Patch Embedding:** The outputs from both parallel branches are concatenated along the channel dimension to form a comprehensive 128-channel feature map. We apply 2D Global Average Pooling (GAP) to collapse the spatial dimensions, followed by a dense Linear layer that projects the features into the final $128$-dimensional patch embedding vector. 







***

**Line-Level Feature Aggregation:**
Because handwriting lines naturally vary in length, they are modeled as an arbitrary sequence of $K$ valid patches, $P_{line} = \{p_1, p_2, \dots, p_K\}$. The DPE-Net, denoted here as the embedding function $f_{\theta}$, processes these patches simultaneously to extract a corresponding sequence of patch embeddings $E = \{e_1, e_2, \dots, e_K\}$, where each $e_k = f_{\theta}(p_k) \in \mathbb{R}^{D}$ and the embedding dimension $D = 128$. 

To fuse these localized textual and geometric features into a single, comprehensive biometric descriptor, we apply Mean Pooling across the sequential $K$ dimension, computing the raw line-level representation $v$:
<br>

$$v = \frac{1}{K} \sum_{k=1}^{K} e_k$$

Finally, to stabilize the distance metric computations required for the subsequent Triplet Margin Loss optimization, the aggregated descriptor $v$ is subjected to $L_2$ normalization. This mathematically projects the final line embedding $\hat{v}$ onto a unit hypersphere $\mathbb{S}^{D-1}$:
<br>

$$\hat{v} = \frac{v}{\|v\|_2}$$

where $\|\cdot\|_2$ denotes the standard Euclidean norm. This normalized vector $\hat{v}$ serves as the final, highly discriminative feature representation of the entire handwriting line.





### 3.4.2 Pretrained FasterNet-T0

As a high-performance alternative to our custom DPE-Net, we integrated the FasterNet-T0 architecture as a primary State-of-the-Art (SOTA) encoder within our pipeline. FasterNet was selected for its exceptional balance of high predictive accuracy and rapid processing speeds. It operates on a high-speed CNN paradigm designed to maximize floating-point operations per second (FLOPS) by utilizing partial convolutions (PConv).

For our framework, we utilized the FasterNet-T0 variant initialized with weights pretrained on the ImageNet dataset, ensuring robust, generalized foundational feature extraction. We adapted the network natively for single-channel grayscale inputs and replaced the standard classification head with a $128$-dimensional linear projection layer. Identical to our custom architecture, the FasterNet-T0 processes sequences of $K$ patches and aggregates them via Mean Pooling and L2 Normalization to generate the final line-level biometric embedding.

### 3.4.3 Other SOTA Encoders Considered


To rigorously validate our selection of **DPE-Net** and **FasterNet-T0** as our primary operational models, we established a comprehensive benchmarking cohort representing diverse architectural paradigms. We evaluated classic deep convolutional networks (**Pretrained ResNet-18** [Classic CNN]), mobile-optimized lightweight architectures (**Pretrained MobileNetV4 Conv-S** [Mobile CNN]), and pure self-attention mechanisms (**Pretrained ViT-Tiny** [Pure Transformer]). Furthermore, we investigated several cutting-edge hybrid architectures that fuse CNNs and Transformers to balance local feature extraction with global contextual awareness. These included **Pretrained EdgeNeXt-XXS** [Hybrid Edge], **Pretrained EfficientViT-M0** [Hybrid Speed], and custom hybrid combinations linking these pretrained backbones with our proprietary Single-Path aggregation heads (e.g., **Pretrained EdgeNeXt-XXS + Single-Path** [Strided + GAP] and **Pretrained FasterNet-T0 + Single-Path** [Strided + GAP]).





## 3.5 Patch Encoder Model Training

To optimize the feature extraction capabilities of our patch encoder, we employed a metric learning paradigm designed to explicitly separate inter-writer styles while clustering intra-writer variations. The complete end-to-end training architecture and data flow are illustrated in **Figure 4**.

To train the network, we utilized a **Triplet Siamese Architecture**. During each training step, the model processes three distinct handwriting samples simultaneously: an Anchor ($A$), a Positive ($P$), and a Negative ($N$). The Anchor and Positive samples are distinct lines drawn from the same writer, while the Negative sample is drawn from a different, randomly selected writer. 

Following the feature aggregation step described in Section 3.4.1, the network outputs L2-normalized line-level embeddings for each branch, denoted as $\hat{v}_a$, $\hat{v}_p$, and $\hat{v}_n$, respectively. 

**Training Objective (Loss Function):**
To optimize the embedding space, we utilized the Triplet Margin Loss based on squared Euclidean distances. The objective is to minimize the distance between the Anchor and Positive embeddings while maximizing the distance between the Anchor and Negative embeddings by at least a predefined margin $\alpha$. The loss function $\mathcal{L}_{triplet}$ is mathematically defined as:

<br>

  $$\mathcal{L}_{triplet} = \frac{1}{B} \sum_{i=1}^{B} \max \left(0, \|\hat{v}_{a,i} - \hat{v}_{p,i}\|_2^2 - \|\hat{v}_{a,i} - \hat{v}_{n,i}\|_2^2 + \alpha \right)$$

<br>

where $B$ represents the batch size, and $\alpha = 0.4$ enforces a strict margin of separation between identical and distinct writers. 

### 3.5.1 Optimization and Hyperparameters

To prevent data leakage and ensure generalizability, the training dataset was split into an 80/20 writer-disjoint configuration (`val_split_by_writers = 0.2`). During training, each handwriting line was dynamically represented by randomly sampling $K = 8$ spatial patches (`patches_per_line = 8`). This dynamic sampling acts as an aggressive form of spatial data augmentation, preventing the network from memorizing fixed sequence locations.

The model was optimized using the **Adam optimizer** with an initial learning rate of $\eta = 0.0005$. To dynamically adjust the learning rate as the model converged, we implemented a `ReduceLROnPlateau` learning rate scheduler monitoring the validation loss. The scheduler was configured to halve the learning rate (factor of $0.5$) if the validation loss plateaued for 3 consecutive epochs, down to a minimum bound of $10^{-6}$.

To maximize computational efficiency and throughput on our NVIDIA RTX 3050 GPU, we trained the network using a batch size of $64$ triplets. We also enabled Automatic Mixed Precision (AMP) via PyTorch's `GradScaler`, which computes gradients in `float16` while maintaining `float32` weight updates, significantly reducing VRAM consumption without degrading stability. 

The network was trained for a maximum of $50$ epochs. However, to prevent overfitting on the training distribution, we strictly enforced early stopping with a patience of $10$ epochs. Checkpoints were saved exclusively when the validation loss reached a new minimum, ensuring the finalized weights represent the absolute best generalized state of the network. To guarantee total experimental reproducibility across all runs, the environment, data split, and model initializations were locked to a global random seed of $42$.



## 3.6 Evaluation Metrics and Threshold Determination

To objectively evaluate our framework on writer verification tasks (determining whether two handwriting samples belong to the same author), we operate in a biometric distance space rather than a direct classification space. 

Let $e_1$ and $e_2$ represent the $L_2$-normalized feature embeddings of two given handwriting samples (either at the line or page level). The dissimilarity between these samples is computed using Cosine Distance, defined mathematically as:

<br>

  $$D_{cos}(e_1, e_2) = 1 - (e_1 \cdot e_2)$$

A binary prediction is made by comparing this distance against a decision threshold $t$. If $D_{cos} \le t$, the samples are classified as a positive pair (same writer); otherwise, they are classified as a negative pair (different writers). 

**Dynamic Threshold Determination and Balanced Evaluation:**
A critical challenge in open-set verification is that relying on a statically predefined threshold is highly susceptible to dataset bias. Furthermore, skewed evaluation sets can artificially inflate performance metrics. To prevent this, our validation and testing protocols were strictly constructed using an exactly equal number of positive (same-writer) and negative (different-writer) pairs, guaranteeing a perfectly balanced evaluation devoid of class-imbalance artifacts.

To ensure our decision boundary is rigorously generalizable, we dynamically determine the optimal operating threshold $t^*$ using the disjoint validation cohort prior to final testing. We perform a fine-grained continuous threshold sweep across the absolute distance range $t \in [0.0, 2.0]$. For each discrete step, we compute standard evaluation metrics: Accuracy, Precision ($P$), and Recall ($R$). We define the optimal threshold $t^*$ as the Break-Even Point—the exact piecewise-linear intersection where Precision equals Recall:

<br>

  $$t^* = \{ t \mid P(t) = R(t) \}$$

By locking the threshold at this point of equilibrium, we guarantee the model is benchmarked at its most balanced operational state. This established threshold is then strictly applied to the entirely unseen test set to compute the final, reported metrics: Accuracy, Precision, Recall, F1-Score, and the threshold-independent Area Under the ROC Curve (AUC).





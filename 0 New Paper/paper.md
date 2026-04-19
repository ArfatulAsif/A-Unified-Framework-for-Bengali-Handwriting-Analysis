# 3. Methodology

In this section, we detail the complete pipeline of our proposed writer identification framework. We begin by outlining our dataset preparation and preprocessing strategies, followed by the core architecture of our Hierarchical Patch Encoder, the comparative State-of-the-Art (SOTA) architectures considered, and our training objectives and hyperparameter setups. Finally, we break down the framework into four distinct operational pipelines: Line and Page-Level Inference (Verification), Writer-Based Document Retrieval, Unsupervised Document Clustering, and Sequential Multi-Writer Segmentation.

All computational tasks, including data preprocessing and model training, were executed using Python 3.9.13 on a single NVIDIA RTX 3050 GPU, utilizing PyTorch 2.8.0, CUDA 12.8, and cuDNN 8.9.7.

## 3.1 Dataset Collection and Preparation

To construct a comprehensive, highly variable, and inclusive Bangla handwriting dataset, we aggregated data from three primary sources:

1. **BN-HTRd:** Provided existing line-segmented images and full handwritten pages from 237 distinct writers.
2. **WBSUBNdb_text:** Contributed handwritten pages from 188 writers. We manually processed these pages through our line-segmentation pipeline to create dedicated line-level datasets, visually evaluating and verifying each segmented line to ensure high ground-truth quality.
3. **Custom Dataset:** We compiled an additional, highly challenging set featuring 10 specialized writers (yielding the remaining pages to complete the dataset). This custom subset was specifically curated to introduce severe real-world impurities, including low-quality mobile phone scans, heavy shadows, and varied camera orientations.

Throughout the aggregation process, we thoroughly reviewed the data to ensure it accurately reflected the unpredictable nature of real-world physical documents, explicitly keeping variations such as exceptionally long and short lines, diverse page colors, and curved or slanted handwriting.

**Dataset Summary:**
* **Total Writers:** 435
* **Total Handwritten Pages:** 2,825 (Average of ~7 pages per writer)
* **Total Segmented Lines:** 29,268 (Average of ~70 lines per writer)

## 3.2 Writer-Disjoint Split and Evaluation Protocol

To ensure our model learns generalized, writer-independent style features rather than memorizing specific handwriting traits from the training set, we adhered to a strict, zero-shot, writer-disjoint evaluation protocol. This guarantees the model will perform reliably on unseen writers in real-world scenarios.

The overarching dataset of 435 writers was strategically partitioned into the non-overlapping subsets described in **Table I**.

### **Table I: Dataset Partitioning and Task Allocation (435 Writers)**

| Pipeline / Task | Phase | Writer IDs | Data Level |
| :--- | :--- | :--- | :--- |
| **Patch Encoder** | Training | 1–371 | Line |
| **Verification** | Tuning | 372–403 | Line & Page |
| **Retrieval** | Evaluation | 404–415 | Page |
| **Clustering** | Tuning | 372–403 | Page |
| **Clustering** | Evaluation | 404–415 | Page |
| **Sequential Multi-Writer Segmentation** | Tuning | 395–420 | 196 Synthesized Pages* |
| **Sequential Multi-Writer Segmentation** | Evaluation | 421–435 | 97 Synthesized Pages* |

*\*Note: Synthesized pages for segmentation were created by cropping and vertically merging segments from 1–4 distinct writers to simulate intrinsic plagiarism; these were used for tuning sequential clustering/smoothing and evaluating segmentation via SER.*

**Standardized Ablation Cohort:**
Finally, to conduct fair State-of-the-Art (SOTA) comparisons and detailed ablation studies without exhausting our primary test sets, we isolated a standardized mini-cohort of 100 writers. This cohort was strictly partitioned into 80 train/validation writers and 20 test writers for line-level tasks, with 10 of those test writers reserved for page-level evaluations.

## 3.3 Preprocessing Handwriting Lines

To ensure our feature extractor learns robust stylistic representations rather than superficial dataset artifacts (e.g., uneven lighting, arbitrary margins, or scanner noise), we subjected each segmented handwriting line to a stringent, multi-stage signal conditioning and geometric normalization pipeline, as illustrated in **Figure 2**.

Let the raw grayscale handwriting line be denoted as $I_{raw}$. First, to maximize ink visibility against degraded backgrounds, we applied Contrast Limited Adaptive Histogram Equalization (CLAHE) to produce an enhanced image, $I_{clahe}$. 

To eliminate arbitrary background margins, we generated a binary adaptive text mask $M$ from $I_{clahe}$. By computing the smoothed 1D pixel projections along the horizontal and vertical axes of $M$, we isolated the tightest bounding coordinates $[y_0, y_1]$ and $[x_0, x_1]$ that contained valid ink. We then cropped the enhanced image to these coordinates to yield the content-only image:

$$I_{crop} = I_{clahe}[y_0:y_1, x_0:x_1]$$

Next, we normalized the physical scale of the handwriting. $I_{crop}$ was resized to a fixed target height $H_{target} = 128$ pixels while strictly preserving its original aspect ratio, resulting in a resized image $I_{res}$ of width $W_{new}$. To unify the tensor dimensions for batch processing without distorting the handwriting geometry, $I_{res}$ was placed onto a fixed-size canvas $C \in \mathbb{R}^{H_{target} \times W_{max}}$, where the maximum width $W_{max} = 1580$. During training, we actuated spatial data augmentation by placing $I_{res}$ at a random horizontal offset (`place_train="random"`), whereas during evaluation, we centered it deterministically (`place_eval="center"`). The canvas was then normalized to a continuous float range $[0, 1]$.

**Patch Extraction**
Because handwriting lines vary drastically in length, we modeled each line as a sequence of localized, overlapping visual patches. Let a single patch be mathematically designated as $p_k \in \mathbb{R}^{S_{patch} \times S_{patch} \times 1}$, where the patch size $S_{patch} = 128$. 

We extracted these patches using a sliding window approach along the horizontal axis of the valid content region $[x_{start}, x_{end}]$ of the canvas $C$. A patch $p_k$ at step $k$ was extracted starting at coordinate $x_k = x_{start} + k \cdot S$, where the stride length $S = 56$:

$$p_k = C[:, x_k : x_k + S_{patch}]$$

To prevent the model from processing empty background space, we enforced a strict foreground density constraint. A patch $p_k$ was only appended to the final sequence if its ink ratio exceeded a minimum threshold $\tau$. By thresholding $p_k$ via Otsu's method, we defined the foreground indicator function $f(p_k)$, and strictly enforced:

$$f(p_k) \geq \tau_{min}$$

where $\tau_{min} = 0.04$. The final preprocessed output for a single handwriting line was the sequence of valid, highly-dense patches $P_{line} = \{p_1, p_2, \dots, p_N\}$, which was subsequently passed to the patch encoder. 

*(Note: An ablation study validating the impact of this preprocessing pipeline is provided in Section 4).*

## 3.4 Patch Encoder Architecture

To extract highly discriminative features from the preprocessed handwriting patches, our primary objective was to construct a robust and highly accurate biometric pipeline. To this end, we investigated architectures across varying parametric capacities. While large-scale models often yield high predictive power, their practical deployment relies on computationally expensive hardware infrastructure, typically available in cloud-based environments. In real-world document analysis scenarios, continuously uploading massive volumes of high-resolution images to centralized servers introduces severe transmission latency and storage bandwidth bottlenecks. Therefore, an equally critical objective of our framework was to ensure the architecture remained exceptionally fast and lightweight, enabling feasible deployment directly on localized edge devices.

To satisfy this dual mandate of biometric accuracy and edge-device efficiency, we engineered the **DPE-Net (Dual-Path Patch Encoder)**—a custom architecture specifically designed for minimal computational overhead and rapid inference without sacrificing verification performance. Alongside our custom network, we extensively evaluated several modern State-of-the-Art (SOTA) architectures to establish a comparative baseline. Among these, the pretrained **FasterNet-T0** emerged as a highly performant alternative that similarly satisfies these high-speed deployment requirements.

To rigorously assess the practical viability of these models for large-scale analysis, our evaluation prioritized the balance between computational efficiency and predictive power. We benchmarked architectural complexity (**Total Parameters, FLOPs**) and processing speed (**Inference Latency per Line**) to validate edge scalability, while concurrently tracking **Line-Level Accuracy and AUC** to ensure reliable biometric verification. The exhaustive comparative benchmarking of these architectures is detailed in **Section 4.1 (Table I)**, and the isolated ablation study of our custom architecture's internal components is provided in **Section 4.2 (Table III)**.




### 3.4.1 DPE-Net (Dual-Path Patch Encoder) Model Architecture

To capture both the fine-grained nuances of ink deposition and the broader geometric flow of handwriting, we engineered a custom Dual-Path Convolutional Neural Network, as shown in **Figure 3**. Rather than relying on deep, parameter-heavy sequential layers, this architecture utilized two parallel expert branches that processed a shared initial feature map.

The network received an input patch tensor of dimensions $128 \times 128 \times 1$. The processing pipeline was structured as follows:

1. **Shared Stem:** The input patch first passed through a downsampling stem comprising a $3 \times 3$ convolutional layer with a stride of 2 and padding of 1, followed by 2D Batch Normalization and an in-place ReLU activation. This reduced the spatial dimensions to $64 \times 64$ while projecting the features into 32 channels.
2. **Standard Branch (Local Textures):** The first parallel pathway consisted of two sequential blocks of $3 \times 3$ convolutions (stride 2, padding 1), Batch Normalization, and ReLU activations. This standard convolutional branch was specifically tuned to capture highly localized micro-textures and intricate stroke variations, outputting a 64-channel feature map.
3. **Dilated Branch (Long-Range Strokes):** The second parallel pathway mirrored the structural depth of the first but utilized dilated convolutions. It applied two sequential blocks of $3 \times 3$ convolutions with a stride of 2, a padding of 2, and a dilation rate of 2. This dilation artificially expanded the receptive field of the $3 \times 3$ kernel to an effective $5 \times 5$ spatial area without increasing the trainable parameter count. This branch was designed to trace long-range stroke connectivity and continuous spatial patterns, yielding a secondary 64-channel feature map.
4. **Feature Fusion and Patch Embedding:** The outputs from both parallel branches were concatenated along the channel dimension to form a comprehensive 128-channel feature map. We applied 2D Global Average Pooling (GAP) to collapse the spatial dimensions, followed by a dense Linear layer that projected the features into the final $128$-dimensional patch embedding vector. 

**Line-Level Feature Aggregation:**
Because handwriting lines naturally vary in length, they were modeled as an arbitrary sequence of $K$ valid patches, $P_{line} = \{p_1, p_2, \dots, p_K\}$. The DPE-Net, denoted here as the embedding function $f_{\theta}$, processed these patches simultaneously to extract a corresponding sequence of patch embeddings $E = \{e_1, e_2, \dots, e_K\}$, where each $e_k = f_{\theta}(p_k) \in \mathbb{R}^{D}$ and the embedding dimension $D = 128$. 

To fuse these localized textual and geometric features into a single, comprehensive biometric descriptor, we applied Mean Pooling across the sequential $K$ dimension, computing the raw line-level representation $v$:

$$v = \frac{1}{K} \sum_{k=1}^{K} e_k$$

Finally, to stabilize the distance metric computations required for the subsequent Triplet Margin Loss optimization, the aggregated descriptor $v$ was subjected to $L_2$ normalization. This mathematically projected the final line embedding $\hat{v}$ onto a unit hypersphere $\mathbb{S}^{D-1}$:

$$\hat{v} = \frac{v}{\|v\|_2}$$

where $\|\cdot\|_2$ denotes the standard Euclidean norm. This normalized vector $\hat{v}$ served as the final, highly discriminative feature representation of the entire handwriting line.

### 3.4.2 Pretrained FasterNet-T0

As a high-performance alternative to our custom DPE-Net, we integrated the FasterNet-T0 architecture as a primary State-of-the-Art (SOTA) encoder within our pipeline. FasterNet was selected for its exceptional balance of high predictive accuracy and rapid processing speeds. It operated on a high-speed CNN paradigm designed to maximize floating-point operations per second (FLOPs) by utilizing partial convolutions (PConv).

For our framework, we utilized the FasterNet-T0 variant initialized with weights pretrained on the ImageNet dataset, ensuring robust, generalized foundational feature extraction. We adapted the network natively for single-channel grayscale inputs by averaging the pretrained RGB weights across the first convolutional layer, and replaced the standard classification head with a $128$-dimensional linear projection layer. Identical to our custom architecture, the FasterNet-T0 processed sequences of $K$ patches and aggregated them via Mean Pooling and $L_2$ Normalization to generate the final line-level biometric embedding.

### 3.4.3 Other SOTA Encoders Considered

To rigorously validate our selection of **DPE-Net** and **FasterNet-T0** as our primary operational models, we established a comprehensive benchmarking cohort representing diverse architectural paradigms. We evaluated classic deep convolutional networks (**Pretrained ResNet-18** [Classic CNN]), mobile-optimized lightweight architectures (**Pretrained MobileNetV4 Conv-S** [Mobile CNN]), and pure self-attention mechanisms (**Pretrained ViT-Tiny** [Pure Transformer]). Furthermore, we investigated several cutting-edge hybrid architectures that fuse CNNs and Transformers to balance local feature extraction with global contextual awareness. These included **Pretrained EdgeNeXt-XXS** [Hybrid Edge], **Pretrained EfficientViT-M0** [Hybrid Speed], and custom hybrid combinations linking these pretrained backbones with our proprietary Single-Path aggregation heads (e.g., **Pretrained EdgeNeXt-XXS + Single-Path** [Strided + GAP] and **Pretrained FasterNet-T0 + Single-Path** [Strided + GAP]).

## 3.5 Patch Encoder Model Training

To optimize the feature extraction capabilities of our patch encoder, we employed a metric learning paradigm designed to explicitly separate inter-writer styles while clustering intra-writer variations. The complete end-to-end training architecture and data flow are illustrated in **Figure 4**.

To train the network, we utilized a **Triplet Siamese Architecture**. During each training step, the model processed three distinct handwriting samples simultaneously: an Anchor ($x_a$), a Positive ($x_p$), and a Negative ($x_n$). The Anchor and Positive samples were distinct lines drawn from the same writer, while the Negative sample was drawn from a different, randomly selected writer. 

Following the feature aggregation step described in Section 3.4.1, the network output $L_2$-normalized line-level embeddings for each branch, denoted as $\hat{v}_a$, $\hat{v}_p$, and $\hat{v}_n$, respectively. 

**Training Objective (Loss Function):**
To optimize the embedding space, we utilized the Triplet Margin Loss based on squared Euclidean distances. The objective is to minimize the distance between the Anchor and Positive embeddings while maximizing the distance between the Anchor and Negative embeddings by at least a predefined margin $\alpha$. The loss function $\mathcal{L}_{triplet}$ is mathematically defined as:

$$\mathcal{L}_{triplet} = \frac{1}{B} \sum_{i=1}^{B} \max \left(0, \|\hat{v}_{a,i} - \hat{v}_{p,i}\|_2^2 - \|\hat{v}_{a,i} - \hat{v}_{n,i}\|_2^2 + \alpha \right)$$

where $B$ represents the batch size, and $\alpha = 0.4$ enforces a strict margin of separation between identical and distinct writers. 

### 3.5.1 Optimization and Hyperparameters

To prevent data leakage and ensure generalizability, the training dataset was split into an 80/20 writer-disjoint configuration (`val_split_by_writers = 0.2`). During training, each handwriting line was dynamically represented by randomly sampling $K = 8$ spatial patches (`patches_per_line = 8`). This dynamic sampling acted as an aggressive form of spatial data augmentation, preventing the network from memorizing fixed sequence locations.

The model was optimized using the **Adam optimizer** with an initial learning rate of $\eta = 0.0005$. To dynamically adjust the learning rate as the model converged, we implemented a `ReduceLROnPlateau` learning rate scheduler monitoring the validation loss. The scheduler was configured to halve the learning rate (factor of $0.5$) if the validation loss plateaued for 3 consecutive epochs, down to a minimum bound of $10^{-6}$.

To maximize computational efficiency and throughput on our NVIDIA RTX 3050 GPU, we trained the network using a batch size of $64$ triplets. We also enabled Automatic Mixed Precision (AMP) via PyTorch's `GradScaler`, which computes gradients in `float16` while maintaining `float32` weight updates, significantly reducing VRAM consumption without degrading stability. 

The network was trained for a maximum of $50$ epochs. However, to prevent overfitting on the training distribution, we strictly enforced early stopping with a patience of $10$ epochs. Checkpoints were saved exclusively when the validation loss reached a new minimum, ensuring the finalized weights represented the absolute best generalized state of the network. To guarantee total experimental reproducibility across all runs, the environment, data split, and model initializations were locked to a global random seed of $42$.

## 3.6 Page-Level Line Segmentation Pipeline

To evaluate our framework on full unconstrained documents, we required a robust pipeline to systematically extract individual handwriting lines. Because our biometric encoder relied on stroke geometry rather than semantic meaning, we bypassed computationally heavy Optical Character Recognition (OCR) text decoding in favor of a high-speed, scale-normalized approach:

1. **Scale-Normalized Detection:** The document was dynamically resized (preserving aspect ratio) to a maximum dimension of 1024 pixels. It was then passed through a CRAFT-based detection module. By applying a binary threshold to the resulting character and affinity heatmaps and extracting bounding coordinates via connected component analysis, we extracted raw horizontal bounding boxes, skipping the character recognition phase entirely.
2. **Adaptive Tolerance Grouping:** Fragmented word boxes were grouped into continuous horizontal lines based on their vertical center coordinates. Two adjacent boxes were merged if their vertical distance was strictly less than an adaptive tolerance threshold ($\tau = 0.5$) scaled by the height of the preceding box.
3. **Margin Extraction:** The aggregated line coordinates were projected back to the original high-resolution scale, padded with a 4-pixel margin to preserve extreme ascenders and descenders, and cropped.

The selection of this specific detection and adaptive grouping strategy was driven by extensive latency and accuracy benchmarking. A detailed comparative analysis of this pipeline against alternative segmentation strategies is provided in the ablation study in **Section 4.2.3 (Table V)**.

## 3.7 Hierarchical Local-to-Global Feature Aggregation

For full-page verification, we needed to compress the extracted visual information into a single highly discriminative biometric vector. We achieved this through a structured, hierarchical aggregation pipeline (Patch $\rightarrow$ Line $\rightarrow$ Page).

Mathematically, let a full document page $D$ be segmented into $L$ valid handwriting lines, $D = \{l_1, l_2, \dots, l_L\}$. As established in Section 3.4.1, each line $l_i$ is represented by a sequence of $K$ patches, where the encoder mapped each patch to an embedding $e_{i,k}$.

First, we aggregated the localized patches to form the line-level representation $v_i$ via mean pooling:

$$v_i = \frac{1}{K} \sum_{k=1}^{K} e_{i,k}$$

Next, we integrated the sequential line vectors to form the global raw page representation $V$ by applying a second tier of mean pooling across the $L$ dimension:

$$V = \frac{1}{L} \sum_{i=1}^{L} v_i$$

Finally, to stabilize distance metric computations, the aggregated page descriptor was subjected to $L_2$ normalization, projecting the final biometric signature $\hat{V}$ onto a unit hypersphere:

$$\hat{V} = \frac{V}{\|V\|_2}$$

This hierarchical approach structurally preserved the horizontal stroke sequences inherent to human handwriting. By enforcing a Line-Level intermediary, the network forced the patches to maintain their sequential spatial context. This geometry-aware integration proved superior to "Flat" aggregation strategies (Patch $\rightarrow$ Page), which bypassed line segmentation and treated the document as an unordered "bag of patches." The complete experimental validation justifying this hierarchical selection over flat pooling networks is provided in the ablation study in **Section 4.2.4 (Table VI)**.

## 3.8 Evaluation Metrics and Threshold Determination

To objectively evaluate our framework on writer verification tasks (determining whether two handwriting samples belong to the same author), we operated in a biometric distance space rather than a direct classification space.

Let $e_1$ and $e_2$ represent the $L_2$-normalized feature embeddings of two given handwriting samples (either at the line or page level), as illustrated in **Figure 5** The dissimilarity between these samples was computed using Cosine Distance, defined mathematically as:

$$D_{cos}(e_1, e_2) = 1 - (e_1 \cdot e_2)$$

Because all feature vectors are $L_2$-normalized prior to distance calculation, minimizing the Squared Euclidean distance during Triplet Loss optimization mathematically translates directly to maximizing Cosine similarity during evaluation. A binary prediction was made by comparing this distance against a decision threshold $t$. If $D_{cos} \le t$, the samples were classified as a positive pair (same writer); otherwise, they were classified as a negative pair (different writers).


To quantitatively assess the framework's verification performance, we defined the standard binary classification outcomes specifically in the context of writer pairing:
* **True Positives ($TP$):** Same-writer pairs correctly classified as a match ($D_{cos} \le t$).
* **True Negatives ($TN$):** Different-writer pairs correctly classified as non-matches ($D_{cos} > t$).
* **False Positives ($FP$):** Different-writer pairs incorrectly classified as a match (False Acceptance).
* **False Negatives ($FN$):** Same-writer pairs incorrectly classified as non-matches (False Rejection).

Based on these defined biometric pairing outcomes, the primary performance metrics are mathematically formulated as follows:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

$$Precision \ (P) = \frac{TP}{TP + FP}$$

$$Recall \ (R) = \frac{TP}{TP + FN}$$

$$F1\text{-}Score = 2 \cdot \frac{P \cdot R}{P + R}$$



**Dynamic Threshold Determination and Balanced Evaluation:**
A critical challenge in open-set verification is that relying on a statically predefined threshold is highly susceptible to dataset bias. Furthermore, skewed evaluation sets can artificially inflate performance metrics. To prevent this, our validation and testing protocols were strictly constructed using an exactly equal number of positive (same-writer) and negative (different-writer) pairs, guaranteeing a perfectly balanced evaluation devoid of class-imbalance artifacts.

To ensure our decision boundary was rigorously generalizable, we dynamically determined the optimal operating threshold $t^*$ using the disjoint validation cohort prior to final testing. We performed a fine-grained continuous threshold sweep across the absolute distance range $t \in [0.0, 2.0]$. For each discrete step, we computed standard evaluation metrics: Accuracy, Precision ($P$), and Recall ($R$). We defined the optimal threshold $t^*$ as the Break-Even Point—the exact piecewise-linear intersection where Precision equals Recall:

$$t^* = \{ t \mid P(t) = R(t) \}$$

By locking the threshold at this point of equilibrium, we guaranteed the model was benchmarked at its most balanced operational state. This established threshold was then strictly applied to the entirely unseen test set to compute the final, reported metrics: Accuracy, Precision, Recall, F1-Score, and the threshold-independent Area Under the ROC Curve (AUC).







## 3.9 Writer-Based Document Retrieval

While writer verification (1:1 matching) addresses whether two specific documents share an author, writer-based document retrieval (1:N search) involves using a single query document to search a database and retrieve all other pages written by the same individual. To evaluate our framework’s performance in this retrieval space, we implemented a Leave-One-Out (LOO) ranking protocol, independently assessing the retrieval pipelines driven by both our custom DPE-Net (Dual-Path Patch Encoder) and the Pretrained FasterNet-T0.

### 3.9.1 Retrieval Protocol and Ranking Strategy

Let the evaluation dataset consist of a closed set of full document pages, where each page is preprocessed and passed through our hierarchical integration pipeline (utilizing either DPE-Net or FasterNet-T0 as the foundational feature extractor) to produce an $L_2$-normalized global page embedding, $\hat{V}$. The entire dataset can be mathematically defined as a pool of document-embedding pairs, $\mathcal{D} = \{(w_1, \hat{V}_1), (w_2, \hat{V}_2), \dots, (w_N, \hat{V}_N)\}$, where $w_i$ represents the ground-truth writer identity of the $i$-th document.

During the LOO evaluation, every single document in $\mathcal{D}$ was sequentially isolated and treated as a query, $q = (w_q, \hat{V}_q)$. The remaining documents formed the search gallery, $\mathcal{G} = \mathcal{D} \setminus \{q\}$. For each query, we calculated the dissimilarity between the query embedding $\hat{V}_q$ and every gallery embedding $\hat{V}_j \in \mathcal{G}$ using the Cosine Distance:

$$D_{cos}(\hat{V}_q, \hat{V}_j) = 1 - (\hat{V}_q \cdot \hat{V}_j)$$

The gallery documents were then sorted in ascending order of their distance to the query, generating a ranked retrieval list $R_q$. To ensure mathematical validity, any query belonging to a writer with only a single document in the entire dataset (meaning $0$ relevant matches exist in $\mathcal{G}$) was excluded from the final metric aggregation.

### 3.9.2 Evaluation Metrics

To quantitatively assess the ranking quality of our network, we utilized three standard retrieval metrics: **Top-1 Accuracy**, **Top-5 Accuracy**, and **Mean Average Precision (mAP)**. 

**Top-$k$ Accuracy:**
This metric measures the probability that at least one highly relevant document appears within the uppermost results. A query $q$ is considered a "hit" for Top-$k$ accuracy if at least one document in the first $k$ ranks of $R_q$ shares the exact same writer identity ($w_q$). We formally reported Top-1 (the absolute closest match) and Top-5 accuracy to evaluate the model's immediate precision.

**Mean Average Precision (mAP):**
While Top-$k$ accuracy indicates if *any* match was found early, it does not evaluate the model's ability to cluster *all* documents by the same writer together. To evaluate overall ranking quality, we calculated the Mean Average Precision.

First, we computed the Average Precision ($AP$) for a single query $q$. Let $N_q$ represent the total number of relevant documents (true matches) existing in the gallery $\mathcal{G}$. Let $P(r)$ denote the cumulative precision calculated at rank $r$, and let the indicator function $rel(r) \in \{0,1\}$ equal $1$ if the document at rank $r$ is a true match, and $0$ otherwise. The $AP$ for query $q$ is mathematically defined as:

$$AP_q = \frac{1}{N_q} \sum_{r=1}^{|\mathcal{G}|} P(r) \cdot rel(r)$$

This formulation heavily penalizes models that rank true matches lower down the list, as the precision fraction $P(r)$ drops as $r$ increases. Finally, the mAP was computed by averaging the $AP$ scores across all valid queries in the evaluation set $\mathcal{Q}$:

$$mAP = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} AP_q$$

By leveraging mAP alongside Top-$k$ accuracy, we ensured the evaluation protocol captured both the model's absolute precision for immediate document retrieval and its broader capability to correctly group a writer's entire corpus within the embedding space.




## 3.10 Handwriting-Based Document Clustering

In unsupervised document clustering, an investigator is presented with a large, unlabelled corpus of documents and must autonomously group them such that each distinct cluster corresponds to a unique, unknown author. To execute this, we leveraged the global page-level embeddings extracted by our hierarchical pipeline, independently benchmarking the latent spaces generated by both our custom **DPE-Net** and the pretrained **FasterNet-T0**.



### 3.10.1 Algorithm Selection and Distance Formulation

Let an unlabelled corpus consist of $N$ document pages, yielding a set of $L_2$-normalized embeddings $\mathcal{D} = \{\hat{V}_1, \hat{V}_2, \dots, \hat{V}_N\}$. Because traditional clustering algorithms operating in high-dimensional Euclidean space are susceptible to the curse of dimensionality, we explicitly utilized a precomputed Cosine Distance matrix. We constructed a symmetric $N \times N$ distance matrix $M_{dist}$, where each element represents the dissimilarity between two documents:

$$M_{dist}(i, j) = 1 - (\hat{V}_i \cdot \hat{V}_j)$$

To partition this distance space, we evaluated multiple unsupervised clustering methodologies. As detailed in the ablation study in **Section 4.2.5 (Table VII)**, **Agglomerative Hierarchical Clustering** significantly outperformed density-based algorithms such as DBSCAN. While DBSCAN struggled with variable inter-cluster densities and erroneously discarded valid documents as unassigned "noise," Agglomerative Clustering successfully formed cohesive, deterministic groups. 

We applied Agglomerative Clustering using **average linkage**, which merges pairs of clusters based on the average cosine distance between all respective member embeddings. Rather than forcing the algorithm to find a predefined number of clusters ($k$), we controlled the cluster formation dynamically using a maximum distance threshold ($\tau_{cluster}$). If the average distance between two clusters exceeded $\tau_{cluster}$, the merging process halted. 

Because the geometric distribution of the latent space naturally varies between different neural architectures, this optimal stopping threshold was determined independently for each feature encoder. By executing a comprehensive threshold sweep on the disjoint tuning cohort, we selected the thresholds that maximized pairwise Accuracy—yielding $\tau_{cluster} = 0.150$ for the **DPE-Net** and $\tau_{cluster} = 0.110$ for the **FasterNet-T0.**

### 3.10.2 Clustering Evaluation Metrics

To thoroughly evaluate the quality of the unsupervised clustering, we applied two distinct classes of evaluation metrics: Global Partitioning Metrics and Pairwise Assignment Metrics.

Let $W = \{w_1, w_2, \dots, w_N\}$ denote the ground-truth writer identities for the corpus, and let $C = \{c_1, c_2, \dots, c_N\}$ denote the discrete cluster labels assigned by the algorithm. 

**Global Partitioning Metrics:**
To evaluate the structural integrity of the clusters against the ground-truth classes, we utilized the **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**.
* **NMI** measures the mutual dependence between the ground-truth writer distributions and the predicted clusters, normalized by their combined entropy to account for varying cluster sizes. It is defined as:

  $$NMI(W, C) = \frac{2 \cdot I(W; C)}{H(W) + H(C)}$$
  
  where $I$ is the mutual information and $H$ is the Shannon entropy.
* **ARI** measures the similarity between the two data clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters, mathematically adjusted for chance grouping:

  $$ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$$
  
  where $RI$ is the raw Rand Index and $E[RI]$ is its expected value.

**Pairwise Assignment Metrics:**
While ARI and NMI evaluate global dataset structure, real-world application demands a granular understanding of pairwise matching accuracy. We constructed a binary ground-truth matrix where a true pair $y_{i,j} = 1$ if $w_i = w_j$. Similarly, we generated a binary prediction matrix where $\hat{y}_{i,j} = 1$ if $c_i = c_j$. 

By evaluating the upper triangular elements of these matrices (representing all $\frac{N(N-1)}{2}$ unique document combinations), we redefined the standard classification outcomes for the clustering domain:
* **True Positives ($TP$):** Documents by the same writer correctly placed in the same cluster.
* **True Negatives ($TN$):** Documents by different writers correctly placed in different clusters.
* **False Positives ($FP$):** Documents by different writers incorrectly grouped into the same cluster.
* **False Negatives ($FN$):** Documents by the same writer incorrectly separated into different clusters.

Using these clustering-specific outcomes, we computed the pairwise metrics utilizing the standard formulas:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

$$Precision \ (P) = \frac{TP}{TP + FP}$$

$$Recall \ (R) = \frac{TP}{TP + FN}$$

$$F1\text{-}Score = 2 \cdot \frac{P \cdot R}{P + R}$$

By reporting both global (ARI, NMI) and pairwise (Accuracy, F1-Score) metrics, we ensured a highly rigorous and multidimensional assessment of the model's unsupervised grouping capabilities.



















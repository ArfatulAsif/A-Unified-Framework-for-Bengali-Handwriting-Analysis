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

**Mathematical Patch Extraction**
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




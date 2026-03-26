# Literature Review

## Writer Verification and Identification Systems

Automated handwriting analysis has transitioned from heavily engineered, closed-set identification tasks to open-set, text-independent verification frameworks capable of generalizing to unknown writers.

### Legacy Approaches

Early computational methods (1990s–2015) relied predominantly on handcrafted feature extraction and traditional classifiers (e.g., SVMs, k-NNs). Initial online systems focused on dynamic features, such as pen pressure [10.69525/jasqde.7] and multi-template matching [10.1109/CCST.1999.797957], or utilized Genetic Algorithms (GA) to isolate hard-to-imitate strokes [10.1109/ANNES.1995.499465].

In offline, text-independent settings, researchers engineered script-specific descriptors to capture static style cues. For Bangla and other Indic scripts, these included script-aware segmentation utilizing directional and gradient-based features [10.1109/ICPR.2010.494], Radon transform projection profiles [10.1109/DAS.2012.98], and GLCM texture features [10.1109/IC3I.2014.7019776]. Fused approaches combining shape and pen pressure were also explored to improve offline robustness [10.1109/ICFHR.2012.246]. However, these legacy systems were fundamentally constrained by closed-set evaluation protocols, frequently struggling to maintain accuracy when applied to entirely unknown writers.

**Table 1: Summary of Legacy Approaches**

| Paper Title & Year [DOI] | Protocol | Feature Extraction & Methodology | Scope | Accuracy / Metric |
| :--- | :--- | :--- | :--- | :--- |
| Constructing a high performance signature verification system using a GA method, 1995 [10.1109/ANNES.1995.499465] | Closed-set | GA-based selection of dynamic features (pen-up 'virtual strokes'). | Signature | Type I: ~6%, Type II: ~0.8% |
| Pen pressure as an identifying characteristic of signatures, 1998 [10.69525/jasqde.7] | Closed-set | Computational analysis of pen pressure patterns. | Signature | ~99% success; Type I: 1–6% |
| An on-line signature verification system using multi-template matching, 1999 [10.1109/CCST.1999.797957] | Closed-set | Multi-template matching for few-shot training samples. | Signature | FAR: ~1%, FRR: ~7.2% |
| Text Independent Writer Identification for Bengali Script, 2010 [10.1109/ICPR.2010.494] | Closed-set | Script-aware segmentation with directional/gradient features + SVM. | Character | Top-1: ~95%, Top-3: ~99% |
| Writer Identification of Bangla handwritings by Radon Transform, 2012 [10.1109/DAS.2012.98] | Closed-set | Radon transform projection profiles with open-set rejection. | Page | Top-1: 83.63%, Top-3: 92.72% |
| Off-Line Writer Verification Using Shape and Pen Pressure, 2012 [10.1109/ICFHR.2012.246] | Open-set | Fused shape (WDH) with IR-recovered pen pressure. | Line/Page | 97–98% verification acc. |
| Off-line Bangla signature verification: An empirical study, 2013 [10.1109/IJCNN.6707123] | Closed-set | Bitmap, intersections, directional chain code + NN classifier. | Signature | Best AER: 15.75% |
| Text and script independent writer identification, 2014 [10.1109/IC3I.2014.7019776] | Closed-set | GLCM texture features with k-NN across Roman and Indic scripts. | Page | ~82–85% |

### Modern Work

Contemporary writer verification (2017–Present) is driven by deep representation learning, highly optimized for open-set scenarios. These architectures bypass manual heuristics to learn content-invariant style embeddings directly from visual data [10.1007/s11042-018-6577-1]. 

Early deep models isolated writing style using conditional autoencoders [10.1109/ICFHR-2018.2018.00083] or fused CNN patch features with handcrafted descriptors (e.g., LBP) via VLAD encoding [10.1109/ACCESS.2019.2927286]. For the Bangla script specifically, hybrid probability density function (PDF)-CNN pipelines [10.1109/DAS.2018.33] and content-independent local attribute pooling [10.1142/S0218001418560116] demonstrated high efficacy in pairwise verification. Recently, the domain has shifted decisively toward deep metric learning. State-of-the-art systems now utilize Siamese CNNs with contrastive or triplet losses [10.1007/978-981-16-1086-8_39], multi-stream networks to capture micro-deformations [10.1016/j.patcog.2021.108008, 10.1109/CVPR.2019.00591], and CrossViT networks to merge local stroke textures with long-range shape dependencies [10.1007/s40747-025-02011-7].

**Table 2: Summary of Modern Approaches**

| Paper Title & Year [DOI] | Protocol | Methodology & Network Architecture | Scope | Accuracy / Metric |
| :--- | :--- | :--- | :--- | :--- |
| Offline Bengali Writer Verification by PDF-CNN and Siamese Net, 2018 [10.1109/DAS.2018.33] | Closed-set | PDFs of handcrafted features fed into CNN + Siamese network. | Page | 97.64% Acc, 2.36% EER |
| Content Independent Writer ID on Bangla Script, 2018 [10.1142/S0218001418560116] | Closed-set | Local handwriting attributes with MLP/Logistic classifiers. | Page | Top-1: 91.33%, EER: 6.67% |
| Offline Text-Independent Writer ID using Conditional AutoEncoder, 2018 [10.1109/ICFHR-2018.2018.00083] | Open-set | Writer-independent conditional autoencoder isolating style. | Character | Top-25% ≈ 97% (ETL-1) |
| Length Independent Writer ID Based on Deep and Hand-Crafted, 2019 [10.1109/ACCESS.2019.2927286] | Closed-set | CNN patch + LBP fusion with VLAD encoding. | Line | Best on CVL/IAM |
| Inverse Discriminative Networks for Signature Verification, 2019 [10.1109/CVPR.2019.00591] | Open-set | Multi-stream CNN with original/inverted branches + cross-attention. | Signature | CEDAR EER: 3.62% |
| Deep Learning Framework with Histogram Features, 2020 [10.1109/ICFHR2020.2020.00032] | Closed-set | Spatio-temporal histograms + sequence LSTM autoencoder. | Line | 81.75% (English) |
| SigVer - A Deep Learning Based Writer Independent Bangla, 2021 [10.1007/978-981-16-1086-8_39] | Open-set | Siamese CNN with contrastive loss. | Signature | 99.49% |
| Learning micro deformations by max-pooling for offline signature, 2021 [10.1016/j.patcog.2021.108008] | Open-set | CNN max-pooling scheme to learn local micro-deformations. | Signature | CEDAR EER: 2.76% |
| SURDS: Self-Supervised Attention-guided Reconstruction, 2022 [10.48550/arXiv.2021.10138] | Open-set | Self-supervised encoder-decoder pretraining + dual triplet loss. | Signature | N/A (Methodological) |
| Writer verification using feature selection based on genetic algorithm, 2023 [10.4218/etrij.2023-0188] | Closed-set | GA-based feature selection combined with Siamese verification. | Page | Best Acc: 94.54% |
| Multi-scale CNN-CrossViT network for offline signature verification, 2025 [10.1007/s40747-025-02011-7] | Open-set | Multi-scale CNN front-end fused with CrossViT back-end. | Page | Bengali Verif: 95.12% |

***

## Document Retrieval by Writer Handwriting

Recent advancements in document retrieval by writer handwriting demonstrate a clear transition from handcrafted feature engineering to robust deep metric learning and transformer-based architectures. Earlier systems relied on manually engineered features (e.g., HOG, GLBP) combined with Support Vector Machines for dissimilarity learning [10.1007/s11042-020-10162-7]. To capture more generalized style representations in open-set and cross-dataset scenarios, researchers widely adopted convolutional neural networks paired with Vector of Locally Aggregated Descriptors (VLAD) and generalized max-pooling, often enhanced by k-reciprocal nearest neighbors (kRNN) re-ranking [10.1049/bme2.12039, arXiv:2212.07664]. 

More recently, Vision Transformers (ViTs) have dominated the domain, utilizing self-supervised learning on isolated handwriting patches to achieve state-of-the-art retrieval accuracy without requiring enrollment [10.1007/978-3-031-06555-2_24, arXiv:2409.00751]. The latest and most disruptive paradigm proposes using Vision Language Models (VLMs) equipped with late-interaction matching (similar to ColBERT) to perform highly efficient, zero-shot document retrieval directly from visual patch embeddings, completely bypassing traditional OCR and layout parsing pipelines [arXiv:2407.01449].

**Table 3: Summary of Document Retrieval Approaches**

| Paper Title & Year [DOI] | Protocol | Feature Extraction & Methodology | Scope | Accuracy / Metric |
| :--- | :--- | :--- | :--- | :--- |
| ColPali: Efficient Document Retrieval with Vision Language Models, 2025 [arXiv:2407.01449] | Open-set (Zero-shot) | VLM directly encoding document images into multi-vector patch embeddings + Late Interaction (ColBERT-style) matching. Bypasses OCR/parsing pipelines. | Page-level | 81.3% nDCG@5 (ViDoRe benchmark average) |
| Writer identification and writer retrieval based on NetVLAD with Re-ranking, 2022 [10.1049/bme2.12039] | Open-set (Cross-dataset evaluation) | ResNet-20 + NetVLAD layer optimized with triplet semi-hard loss, combined with generalized max-pooling and a k-reciprocal nearest neighbors re-ranking strategy. | Page-level | **Retrieval (mAP):** ICDAR 2013: 97.41%, CVL: 98.6%, KHATT: 97.7% |
| Self-Supervised Vision Transformers for Writer Retrieval, 2024 [arXiv:2409.00751] | Open-set, Cross-dataset/Zero-shot | ViT-small/16 (AttMask self-supervised training with MIM and self-distillation); Foreground patch token extraction; VLAD encoding with PCA whitening; Cosine distance and kRNN reranking. | Page-level | **Historical-WI**: 83.1% mAP, 90.9% Top-1; **HisIR19**: 95.0% mAP, 97.6% Top-1; **CVL**: 98.6% mAP, 99.4% Top-1 |
| SVM-based writer retrieval system in handwritten document images, 2021 [10.1007/s11042-020-10162-7] | Open-set | Handcrafted features (HOG, GLBP, LDF, RLF) extracted via uniform grids; Dissimilarity learning using SVM (RBF kernel) trained on intra-writer and inter-writer difference vectors with soft-max probabilities. | Page-level, Line-level | **CVL**: 100% Top-2; **ICDAR-2011 (Original)**: 100% Top-2; **ICDAR-2011 (Cropped)**: 100% Top-2; **KHATT**: 78.25% Top-2 |
| Writer Retrieval and Writer Identification in Greek Papyri, 2022 [arXiv:2212.07664] | Open-set (Retrieval), Closed-set (Classification) | SIFT descriptors or Self-supervised CNN trained on patches (predicting SIFT cluster IDs) with AngU-Net binarization; VLAD encoding and Generalized Max Pooling (GMP) followed by PCA-whitening; Cosine distance matching. | Page-level (Fragment-level) | **GRK-Papyri (Retrieval)**: 42.2% mAP, 52% Top-1; **GRK-Papyri (Classification)**: 60% Top-1 |
| Writer Identification and Writer Retrieval Using Vision Transformer for Forensic Documents, 2022 [10.1007/978-3-031-06555-2_24] | Closed-set (with enrollment) & Open-set (without enrollment, cross-dataset) | SIFT keypoint detection to extract 32x32 patches followed by binarization; ViT-Lite-7/4 (trained from scratch) processes patches; MLP head used for classification (Closed-set) or ViT feature extraction with kNN and distance metrics like Euclidean/Canberra (Open-set). | Page-level / Region-level | **CVL (Closed-set)**: 99.9% Top-1; **CVL (Open-set)**: 97.4% Top-1, 92.8% mAP; **ICDAR 2013 (Open-set)**: 97.0% Top-1, 84.4% mAP; **WRITE (Cross-dataset)**: 81.7% Top-1, 53.3% mAP |

***

## Document Clustering

Unsupervised document clustering focuses on grouping handwritten documents or sub-character components by writer identity without relying on labeled training data. Approaches in this domain vary significantly based on the underlying feature extraction methodology. Traditional techniques extract structural and spatial features—such as skeletonized disjoint graphs [10.1002/sam.11488] or handcrafted morphologic traits like stroke width and blob orientation. These features are subsequently grouped using outlier-tolerant K-Means or Fuzzy C-Means (FCM) algorithms [10.48550/arXiv.2210.16780]. To handle more complex script variations and minimize manual feature engineering, modern frameworks leverage self-supervised representation learning. By utilizing deep convolutional architectures (e.g., DenseNet121 AutoEmbedders) to extract highly clusterable, low-dimensional embeddings, these modern systems can apply standard K-Means clustering to achieve near-perfect purity and Normalized Mutual Information (NMI) scores on standard benchmark datasets [10.3390/math10244796].

**Table 4: Summary of Document Clustering Approaches**

| Paper Title & Year [DOI] | Protocol | Feature Extraction & Methodology | Scope | Accuracy / Metric |
| :--- | :--- | :--- | :--- | :--- |
| Recognizing Handwriting Styles in a Historical Scanned Document Using Unsupervised Fuzzy Clustering, 2022 [10.48550/arXiv.2210.16780] | Unsupervised | 10 Handcrafted spatial features (stroke width, connected component, corner angle, orientation, convex area, height, width, aspect ratio, blobLoG, blobDoG) followed by PCA/ICA/KernelPCA dimensionality reduction and **Fuzzy C-Means (FCM)** clustering using **Euclidean** distance. | Line-level and Word-level | C-A 35 Dataset: Top-1 100% accuracy in most test cases; Fuzzy Partition Coefficient (FPC): 0.86 (Hand1/Hand2) and 0.72 (Hand2/Hand3). |
| Self-Writer: Clusterable Embedding Based Self-Supervised Writer Recognition from Unlabeled Data, 2022 [10.3390/math10244796] | Self-supervised feature extraction + Unsupervised clustering | DenseNet121 (CNN) in a Siamese/AutoEmbedder architecture extracting 16-dimensional embeddings using Euclidean distance, followed by K-Means clustering. | Text block-level | IAM (50 writers, 0 impurity): 96.9% ACC, 0.988 NMI, 0.934 ARI; CVL (50 writers, 0 impurity): 94.1% ACC, 0.974 NMI, 0.919 ARI. |
| A clustering method for graphical handwriting components and statistical writership analysis, 2021 [10.1002/sam.11488] | Unsupervised clustering for feature extraction + Supervised writer identification | Skeletonized disjoint graphs via 'handwriter' package; Outlier-tolerant K-Means clustering using a custom graph distance metric (combining endpoint location, straight-line distance, and shape) | Sub-character level (graphical structures) | CVL: 98.06% Writer Identification Accuracy (using K=40 template) |

***

## Multi-Writer Segmentation

Research focusing explicitly on multi-writer segmentation—often framed as intrinsic plagiarism detection within physical handwritten documents—remains notably sparse. While extensive methodologies exist for digital text segmentation and authorship attribution, applying these concepts to offline handwriting requires systems to detect subtle, unannounced stylistic shifts within a single document page. Existing methods typically approach this by framing it as a sequence-segmentation or local anomaly detection problem, relying on sliding-window feature extraction to identify transition points between different authors along a line or paragraph. However, dedicated frameworks optimized for detecting these intrinsic boundaries in complex, ligature-rich scripts like Bangla are virtually nonexistent in the current literature, presenting a significant gap in automated forensic analysis capabilities.

***

## Positioning Our Research

While the transition to deep metric learning has vastly improved writer verification and retrieval capabilities, the literature reveals a distinct lack of unified, multi-task frameworks—particularly for the Bangla script. Most existing systems treat verification, retrieval, clustering, and multi-writer segmentation as highly isolated tasks, frequently evaluating them on limited, closed-set datasets. Furthermore, advanced forensic tasks like multi-writer intrinsic plagiarism detection remain largely unexplored. 

Our research bridges these critical gaps by introducing a comprehensive, end-to-end deep embedding framework. By scaling our training cohort to over 400 writers and introducing real-world document variances (such as page color and line orientation), we train an updated, highly optimized CNN architecture to learn a deeply robust metric space. This shared hierarchical embedding space (`patch -> line -> page`) not only sets a new standard for open-set writer verification but scales seamlessly to facilitate 1-to-many document retrieval, unsupervised density-based clustering (via DBSCAN), and line-level multi-writer segmentation. Ultimately, this work evolves previous verification baselines into a versatile, field-ready forensic analysis system capable of handling complex, real-world document intelligence tasks.

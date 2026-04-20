
# 2 Literature Review

##  2.1 Writer Identification and Verification

Research in author biometrics broadly categorizes into signature verification, content independent text-based writer identification (1:N search), and content independent text-based writer verification (1:1 matching). The following sections chronologically review significant developments across these domains.

### Signature Verification
Extensive work has been dedicated to signature verification, evolving from simple statistical models to complex deep learning pipelines. Early deep learning approaches explored both online and offline modalities. For online signature verification, Time-Aligned Recurrent Neural Networks (TA-RNNs) were successfully applied to an open-set dataset of 1,526 writers, achieving an Equal Error Rate (EER) of 4.2% [arXiv:2002.10119]. Conversely, early offline approaches relied on simpler architectures; for instance, a 6-layer Multi-Layer Perceptron (MLP) containing only 500 parameters was evaluated on a closed set of Dutch and English signatures, achieving an accuracy of 82.50% [10.5815/ijisa.2021.01.04]. 

As architectures advanced, researchers integrated specialized feature extraction schemes. A Spatial Variation-dependent Verification (SVV) scheme utilizing Textural Features (TF) was proposed for offline Hebrew signature verification, demonstrating a 95.58% accuracy on an open set of 170 writers [10.1038/s41598-023-48789-9]. Similarly, a custom 3-layer CNN evaluated on the publicly available English CEDAR dataset demonstrated robust open-set capabilities with a 98.5% accuracy [10.1109/ICAC2N63387.2024.10895553]. Most recently, the SignForensics framework introduced a highly complex multi-stage pipeline utilizing YOLOv10 for detection, CycleGAN for noise removal, and SigNet/CapsNet for feature extraction, achieving up to 94.68% offline open-set accuracy [10.1109/ACCESS.2025.3617221].

### Writer Identification
While signature verification focuses on a highly specific biometric marker, general handwriting analysis has historically focused on writer identification within closed-set parameters (where training and testing cohorts share the same identities). Working at the offline patch-level, a Handwriting Thickness Descriptor (HTD) based on an extended ResNet architecture (~11.54M parameters) achieved 97.5% accuracy on the English IAM dataset (657 writers) and 99.61% on the Dutch Firemaker dataset (250 writers) [10.1016/j.engappai.2020.103912]. 

For Indic scripts, an offline word-level Thresholded Gabor-CNN (TGCNN) was proposed. This lightweight model (0.2M parameters, 171M FLOPs) processed black-and-white word images through Gabor filters, achieving 97.4% closed-set accuracy on a 260-writer Bengali dataset and 94.1% on a 12-writer Devanagari dataset [[https://doi.org/10.1109/ACCESS.2021.3114799](https://doi.org/10.1109/ACCESS.2021.3114799)]. Transitioning from CNNs to attention mechanisms, the Residual Swin Transformer Classifier (RSTC) was introduced for offline word-level identification, reporting 90.7% top-1 accuracy on the IAM dataset and 92.7% on the German CVL dataset across a combined 967 closed-set writers [10.1109/ACCESS.2022.3178597]. Moving beyond spatial features entirely, a highly specialized framework utilized Singular Value Decomposition with Linear Discriminant Analysis (SVD-LDA) to process the hyperspectral 1D color wavelengths of individual English ink pixels. While computationally heavy (15,160M FLOPs), it achieved an exceptional 99.99% accuracy on a small closed-set cohort of 61 combined writers [10.1109/ACCESS.2026.3667159].

### Writer Verification
In contrast to identification, writer verification aims to determine if two disparate handwriting samples belong to the same author, frequently demanding open-set protocols (writer-disjoint datasets) to ensure true generalization. An early empirical study investigated offline page-level verification by directly extracting patches from full documents without explicit line segmentation. Evaluated on a closed set of 200 writers, the study established that XceptionNet performed optimally (98.55% accuracy) when handwriting speeds were consistent (e.g., slow vs. slow). However, performance dropped significantly (to 69.8%) when comparing mismatched intra-writer speeds (slow vs. fast), highlighting the fragility of standard CNNs to behavioral variations [10.1109/ACCESS.2019.2899908]. 

To address the need for generalized, open-set matching, contemporary approaches have heavily adopted Siamese architectures. For example, KhmerWriterID introduced an online, word-level hybrid Siamese network (combining CNN and biGRU layers) for the Khmer language. Evaluated on a strictly writer-disjoint dataset of 298 unique writers, the model achieved a 99.74% verification accuracy, establishing the superiority of multi-modal sequential architectures for robust biometric matching [10.1109/ACCESS.2026.3666649].




## 2.2 Writer Retrieval

Writer retrieval (1:N search) differs from direct identification by attempting to rank a gallery of documents based on their biometric similarity to a single query document. 

In 2020, an offline, document-level retrieval system established a strong baseline using classical machine learning and handcrafted features. Evaluated across multiple languages (French, English, German, Greek, and Arabic) using open-set protocols, the framework utilized a Support Vector Machine (SVM) trained on Run Length Features (RLF). Tested on large-scale datasets including CVL (309 writers), ICDAR-2011 (26 writers), and KHATT (1,000 writers), the system achieved 100% Top-2 accuracy on CVL and ICDAR-2011, and 78.25% on KHATT, with rapid inference times of ~0.42 to 0.67 seconds per image [10.1007/s11042-020-10162-7].

Transitioning to deep convolutional architectures, a 2021 study introduced a patch-to-global aggregation pipeline. Small offline image patches from English, Greek, German, and Arabic documents were extracted and processed through a ResNet-20 encoder. To form a single global document descriptor, the network applied a NetVLAD layer combined with Generalized Max-Pooling. Evaluated on strictly open-set splits of ICDAR-2013 (350 total writers), CVL (310 writers), and KHATT (850 writers), the system achieved a 97.41% retrieval accuracy, significantly aided by a k-reciprocal nearest neighbor (krNN-QE) re-ranking strategy [10.1049/bme2.12039].

By 2022, research began integrating self-attention mechanisms. One offline framework utilized a compact Vision Transformer (ViT-Lite-7/4) to process small structural patches extracted around SIFT-detected keypoints. Tested on Latin and Greek scripts across the CVL, ICDAR 2013, and WRITE (16 writers) datasets, the ViT architecture achieved a 97.4% open-set retrieval accuracy (without writer enrollment) and scaled to a 99.9% closed-set identification accuracy [[https://doi.org/10.1007/978-3-031-06555-2_24](https://doi.org/10.1007/978-3-031-06555-2_24)]. 

Simultaneously in 2022, researchers addressed the retrieval of highly degraded historical documents, specifically offline Greek papyri. Operating at the full document-image level on a closed set of 10 unique writers, the study proposed a specialized pipeline utilizing AngU-Net for document binarization, followed by a self-supervised CNN (Cl-S) for latent representation. Due to the extreme degradation of the papyri, this system established a baseline retrieval accuracy of 52% Top-1 and a 42.2% Mean Average Precision (mAP) [arXiv:2212.07664].

More recently, approaches have shifted toward fully self-supervised, page-level Transformers. A recent offline system evaluated German, Latin, French, and English handwriting across an expansive open-set cohort of 1,114 writers (394 for training, 720 for testing). The architecture utilized a self-supervised ViT-small/16 trained with an AttMask technique. Instead of relying on a standard class token, the network extracted local foreground patch tokens, aggregated them via a VLAD codebook, and applied kRNN re-ranking to yield an 83.1% open-set retrieval accuracy [arXiv:2409.00751].



## 2.3 Handwriting-Based Document Clustering

While document clustering based on semantic information or textual content is quite common in natural language processing, research explicitly focused on *writer-based* biometric document clustering remains sparse. Within the domain of authorial grouping, a notable 2018 study presented an offline, document-level framework evaluated during the PAN 2017 Author Clustering Task. Operating in an open-set environment across English, Dutch, and Greek documents (with the exact number of unique writers unstated), the system avoided deep neural networks in favor of classical statistical methods. The top-performing architecture utilized Agglomerative Hierarchical Cluster Analysis (HCA) with average linkage. As a special technique, it applied a log-entropy weighting scheme to the 20,000 most frequent features to map authorial styles. This framework achieved an average F-Bcubed score of 0.5733 across the tested languages and a Mean Average Precision (MAP) of 0.4554 for authorship-link ranking [[https://doi.org/10.1007/978-3-319-98932-7_20](https://doi.org/10.1007/978-3-319-98932-7_20)].


## 2.4 Sequential Multi-Writer Segmentation
Sequential multi-writer segmentation involves detecting and mapping the chronological sequence of multiple distinct authors on a single continuous page. While traditional biometric literature focuses exclusively on single-author documents, to the best of our knowledge, no published work, standardized framework, or dataset currently exists for this specific intra-document segmentation task. This critical gap in forensic document examination establishes our proposed tracking pipeline as a highly novel contribution.








## 2.5 Positioning Our Proposed Framework

Existing research on Bengali handwriting biometrics is largely restricted to closed-set, character- or word-level analysis. In contrast, we introduce a comprehensive, all-in-one offline framework capable of executing verification, document retrieval, unsupervised clustering, and sequential multi-writer segmentation. Operating under a strict zero-shot, open-set paradigm with completely disjoint writer splits, we ensure true generalization using a massive, highly variable corpus aggregated from diverse data sources. Furthermore, rather than proposing an isolated architecture, we exhaustively benchmarked our models against established literature baselines—including classic CNNs (ResNet) and self-attention Transformers (ViT)—reporting parameters, FLOPs, storage sizes, and inference latencies to explicitly differentiate solutions for high-capacity computing resources like the cloud versus resource-constrained environments like edge deployment.

Methodologically, we diverge from standard "flat" (patch $\rightarrow$ page) document analysis. While prior retrieval studies utilized flat aggregation (e.g., NetVLAD pooling), our ablations demonstrate that a geometrically aware *patch $\rightarrow$ line $\rightarrow$ page* hierarchical integration pipeline performs better. By preserving the horizontal spatial sequence of human penmanship, it successfully isolates biometric style from macro-level layout noise. 

Ultimately, this all-in-one framework is the first of its kind for open-set, document-level handwriting analysis in any language. It establishes the first comprehensive open-set baselines for offline Bangla full-page retrieval (evaluated via standard Leave-One-Out Top-$k$ and mAP metrics) and unsupervised writer clustering (adopting literature-proven Agglomerative Clustering). Most critically, it pioneers the task of sequential multi-writer segmentation—detecting and chronologically mapping multiple distinct authors within a single continuous page—thereby addressing a major, previously unexplored gap in forensic document examination.



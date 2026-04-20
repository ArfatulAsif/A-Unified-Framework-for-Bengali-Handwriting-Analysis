\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{hyperref}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2017.DOI}

\title{A Unified Framework for Open-Set Bengali Handwriting Analysis: Writer Verification, Retrieval, Clustering, and Sequential Multi-Writer Segmentation}

\author{\uppercase{Arfatul Islam Asif}\authorrefmark{1},
\uppercase{Sanjoy Das Joy}\authorrefmark{1}, 
and \uppercase{M. Shahidur Rahman}\authorrefmark{1}, \IEEEmembership{Senior Member, IEEE}}

\address[1]{Department of Computer Science and Engineering, Shahjalal University of Science and Technology, Sylhet 3114, Bangladesh (e-mail: arfatulislam2001@gmail.com; sanjoy.cse20@gmail.com; rahmanms@sust.edu). ORCID: 0009-0002-5032-2895 (A. I. Asif); 0009-0000-3319-4540 (S. D. Joy); 0000-0001-5635-4105 (M. S. Rahman)}


\markboth
{Asif \headeretal: A Unified Framework for Open-Set Bengali Handwriting Analysis}
{Asif \headeretal: A Unified Framework for Open-Set Bengali Handwriting Analysis}

\corresp{Corresponding author: M. Shahidur Rahman (e-mail: rahmanms@sust.edu).}













\begin{abstract}
Automated handwriting analysis has various applications in forensic and document archival work. However, most frameworks are restricted to closed-set conditions and isolated character or word-level identification. This becomes more challenging with complex scripts like Bengali (Bangla), where full-document biometric analysis is very rare. In this study, we introduce a unified Deep Embedding Framework that performs four offline tasks: writer verification, writer retrieval, unsupervised document clustering, and sequential multi-writer segmentation at the full-page level while following strict zero-shot, open-set protocols. To separate biometric style from layout noise, we created a Patch-to-Line-to-Page hierarchical integration pipeline that keeps the spatial order of human handwriting intact. To tackle computational challenges, we designed a highly optimized custom CNN called the Dual-Path Patch Encoder (DPE-Net) and adapted the pretrained FasterNet-T0. We tested these models on a curated dataset of 435 writers, offering scalable solutions for edge and cloud environments. This framework sets the first open-set document or page level baselines for all four tasks involving Bangla. For page-level verification, FasterNet-T0 and DPE-Net reached 96.00\% and 95.00\% accuracy, respectively. In retrieval, they achieved 100.00\% and 94.20\% Top-1 accuracies. For unsupervised clustering, our method resulted in pairwise accuracies of 98.64\% and 96.03\%. Finally, we pioneer the novel task of sequential multi-writer segmentation—detecting and mapping distinct authors collaborating on a single page. Here, FasterNet-T0 achieved 79.38\% absolute sequence accuracy, while DPE-Net provided a highly stable Sequence Error Rate of 0.0868. Ultimately, this unified framework significantly expands forensic document examination capabilities. Finally, we pioneer the novel task of sequential multi-writer segmentation, which detects and maps different authors working on a single page. In this task, FasterNet-T0 attained 79.38\% absolute sequence accuracy while DPE-Net had a stable Sequence Error Rate of 0.0868. This unified framework significantly expands forensic document examination capabilities.

\end{abstract}





\begin{keywords}
Offline Bengali Handwriting, Open-Set Writer Biometrics, Document-Level Analysis, Hierarchical Feature Aggregation, Unsupervised Writer Clustering, Writer Retrieval.
\end{keywords}

\titlepgskip=-15pt

\maketitle







\section{Introduction}
\label{sec:introduction}

\PARstart{W}{riter} \textbf{verification} (1:1 matching) determines whether two different handwriting samples were authored by the same person. \textbf{Writer retrieval} (1:N search) involves taking a single query document and ranking a vast database of other documents based on their stylistic similarity. \textbf{Writer-based document clustering} is an unsupervised task that groups a massive, unlabelled stack of documents according to their distinct, unknown authors based on handwriting style only. Finally, \textbf{sequential multi-writer segmentation} is the task of detecting whether a single document page is written by multiple writers, mapping the exact chronological order of multiple distinct authors collaborating on a single, continuous page.

In this paper, we suggest a unified framework that can handle all four tasks at the document (page) level, along with standard line-level verification for Bengali (Bangla) handwriting. It operates strictly under an \textbf{offline, zero-shot open-set} paradigm. Online systems rely on digital devices (like tablets) to capture real-time writing variables such as coordinates, stroke speed, and pen pressure, whereas an "offline" system analyzes static, two-dimensional scanned images of previously written documents—a significantly more challenging computer vision problem. 

Most existing work focuses heavily on signature verification \cite{b1} or analyzes isolated characters and words \cite{b2}. While some studies utilize online tracking data, offline document-level analysis remains rare and is frequently limited to closed-set environments (where training and testing writers overlap) \cite{b3}. Furthermore, while standard writer identification is widely researched than verification. also unsupervised writer clustering is rarely addressed \cite{b4}. Most critically, we found absolutely no published research tackling sequential multi-writer segmentation on a single page. Also, document-level biometric handwriting analysis research applied to the Bangla script is exceptionally rare.

Beyond algorithmic limits, practical, real-world deployment faces a massive logistical bottleneck. The true purpose of these biometric pipelines is to process millions of pages found in historical archives or forensic databases. Even personal uses may contain hundreds or thousands of documents for these purposes (e.g., student handwriting assignment pages or exam paper analysis for verification and cheating detection). For general use cases, continuously uploading massive volumes of high-resolution document images to centralized cloud servers creates severe transmission latency and bandwidth issues. Therefore, to be truly practical, an architecture must be exceptionally fast and lightweight, allowing it to run directly on localized, resource-constrained edge devices without sacrificing accuracy.

Our preliminary work, published in the 2025 28th International Conference on Computer and Information Technology (ICCIT) [pending indexing], addressed the open-set Bangla writer verification task using only a custom Single-Path CNN (MaxPool + Dense). However, when we expanded our scope to full-page retrieval, clustering, and segmentation, which requires processing hundreds of lines and thousands of visual patches per document that custom CNN, alongside many standard SOTA pretrained architectures, proved far too slow and memory-intensive both for training and inference.

To solve this, we propose a unified framework powered by a highly optimized, custom CNN called the \textbf{Dual-Path Patch Encoder (DPE-Net)}. We also adapted and benchmarked the highly performant pretrained SOTA \textbf{FasterNet-T0} across our entire framework, rigorously evaluating both models for model weight, storage signature, and inference speed. 

The primary novel contributions of this paper are summarized as follows:

\begin{itemize}
    \item \textbf{First Unified Framework \& Bangla Baselines:} To our knowledge, this is the first paper to provide a unified, end-to-end framework capable of executing offline page-level handwriting verification, retrieval, clustering, and segmentation within a single system. In doing so, we establish the first zero-shot, open-set baselines for offline Bangla document retrieval and clustering, addressing a major linguistic gap in biometric research.
    
    \item \textbf{Novel Sequential Multi-Writer Segmentation:} We pioneer the task of sequential multi-writer page composition. To our best knowledge, this is the first framework in the literature, for \textit{any} language, capable of detecting and mapping chronological author transitions within a single continuous document page.
    
    \item \textbf{Architectural \& Methodological Innovations:} We propose DPE-Net, a highly optimized dual-path CNN that achieves exceptional biometric accuracy with a tiny storage footprint (707 KB), designed specifically for edge deployment. Furthermore, we empirically demonstrate that our proposed \textit{Patch $\rightarrow$ Line $\rightarrow$ Page} hierarchical integration technique vastly outperforms traditional "flat" (Patch $\rightarrow$ Page) spatial pooling strategies for full-document analysis.
    
    \item \textbf{Massively Scaled \& Curated Datasets:} We constructed a highly variable corpus of 435 writers by processing and line-segmenting existing datasets alongside custom data to introduce severe real-world noise. Additionally, we engineered a novel dataset of nearly 300 visually cohesive, synthesized multi-author pages specifically to train and evaluate chronological tracking algorithms.
    
    \item \textbf{Comprehensive SOTA Benchmarking \& Reproducibility:} We provide an exhaustive comparative analysis against numerous modern SOTA paradigms, accompanied by rigorous pipeline ablation studies. Conducted entirely on standard hardware settings, our methodology is designed to be highly accessible and easily reproducible across diverse deployment environments.
\end{itemize}






\section{Literature Review}
\label{sec:literature_review}

\subsection{Writer Identification and Verification}

Research in author biometrics falls into three main categories: signature verification, content independent text-based writer identification (1:N search), and content independent text-based writer verification (1:1 matching). 

\subsubsection{Signature Verification}
Extensive work has been dedicated to signature verification. It has changed from basic statistical models to advanced deep learning systems. Early deep learning methods looked at both online and offline types. For online signature verification, Time-Aligned Recurrent Neural Networks (TA-RNNs) were effectively used on an open dataset of 1,526 writers. This work achieved an Equal Error Rate (EER) of 4.2\% \cite{b5}. On the other hand, early offline methods used simpler architectures. For example, a 6-layer Multi-Layer Perceptron (MLP) with just 500 parameters was tested on a closed set of Dutch and English signatures, obtaining an accuracy of 82.50\% \cite{b6}.

As architectures advanced, researchers integrated specialized methods for feature extraction. A Spatial Variation-dependent Verification (SVV) scheme using Textural Features (TF) was proposed for offline Hebrew signature verification, showing a 95.58\% accuracy on an open set of 170 writers \cite{b7}. Similarly, a custom 3-layer CNN tested on the publicly available English CEDAR dataset showed strong open-set abilities with a 98.5\% accuracy \cite{b8}. Most recently, the SignForensics framework introduced a complex multi-stage system using YOLOv10 for detection, CycleGAN for noise removal, and SigNet/CapsNet for feature extraction. This setup achieved up to 94.68\% offline open-set accuracy \cite{b1}.

\subsubsection{Writer Identification}

While signature verification targets a specific biometric marker, general handwriting analysis has historically aimed at identifying writers within closed-set parameters, where the training and testing groups share the same identities. By working at the offline patch level, a Handwriting Thickness Descriptor (HTD) using an extended ResNet architecture, which has about 11.54 million parameters, achieved 97.5\% accuracy on the English IAM dataset with 657 writers and 99.61\% on the Dutch Firemaker dataset with 250 writers \cite{b9}.

For Indic scripts, an offline word-level Thresholded Gabor-CNN (TGCNN) was proposed. This lightweight model, with 0.2 million parameters and 171 million FLOPs, processed black-and-white word images using Gabor filters. It achieved 97.4\% closed-set accuracy on a 260-writer Bengali dataset and 94.1\% on a 12-writer Devanagari dataset \cite{b2}. 
Transitioning from CNNs to attention mechanisms, the Residual Swin Transformer Classifier (RSTC) was introduced for offline word-level identification. It reported 90.7\% top-1 accuracy on the IAM dataset and 92.7\% on the German CVL dataset with a total of 967 closed-set writers \cite{b10}. 
Moving beyond spatial features, a highly specialized framework used Singular Value Decomposition with Linear Discriminant Analysis (SVD-LDA) to process the hyperspectral one-dimensional color wavelengths of individual English ink pixels. While computationally heavy with 15,160 million FLOPs, it achieved an outstanding 99.99\% accuracy on a small closed-set group of 61 combined writers \cite{b11}.

\subsubsection{Writer Verification}

In contrast to identification, writer verification aims to determine if two different handwriting samples belong to the same author. This often requires open-set protocols, using writer-disjoint datasets, to ensure true generalization. An early study looked into offline page-level verification by extracting patches from full documents without explicit line segmentation. Evaluated on a closed set of 200 writers, the study established that XceptionNet performed optimally (98.55\% accuracy) when handwriting speeds were consistent (e.g., slow vs. slow). However, performance dropped significantly (to 69.8\%) when comparing mismatched intra-writer speeds (slow vs. fast), highlighting the fragility of standard CNNs to behavioral variations \cite{b3}. 

To address the need for open-set matching, modern methods have widely used Siamese architectures. For instance, KhmerWriterID developed an online, word-level hybrid Siamese network that combines CNN and biGRU layers for the Khmer language. Tested on a dataset with 298 unique writers who were not included in the training, the model reached a 99.74\% verification accuracy. This result highlights the effectiveness of multi-modal sequential architectures for reliable biometric matching \cite{b12}.

\subsection{Writer Retrieval}

Writer retrieval (1:N search) ranks a collection of documents based on their biometric similarity to a single query. Early solid baselines used classical machine learning, including Support Vector Machines (SVM) trained on Run Length Features. When tested across several languages (French, English, German, Greek, and Arabic), this method achieved quick inference and up to 100\% Top-2 open-set accuracy on datasets like CVL and ICDAR-2011 \cite{b13}.

Later research shifted to deep convolutional and attention-based models. A 2021 study introduced a patch-to-global pipeline that combined a ResNet-20 encoder with a NetVLAD layer and krNN re-ranking, achieving 97.41\% open-set accuracy across various scripts (English, Greek, German, and Arabic) \cite{b14}. Vision Transformers (ViTs) made further progress; a compact ViT-Lite-7/4 that processed SIFT-detected keypoint patches reached 97.4\% open-set retrieval accuracy on Latin and Greek scripts \cite{b15}. More recently, a self-supervised ViT-small/16 that used an AttMask technique and VLAD codebooks scaled up to a broad open-set group of 1,114 writers. It evaluated German, Latin, French, and English handwriting, resulting in an 83.1\% retrieval rate \cite{b17}.

Beyond modern handwriting, specialized deep learning frameworks have tackled very damaged historical datasets. Working at the full document image level, a pipeline that combines AngU-Net binarization with a self-supervised CNN set an initial 52\% Top-1 retrieval baseline on a closed-set of offline Greek papyri \cite{b16}.

\subsection{Handwriting-Based Document Clustering}

While there is a lot of research on semantic or information based document clustering, biometric writer-based clustering is not as common. A 2018 study set an important offline baseline by tackling the open-set PAN 2017 Author Clustering Task with English, Dutch, and Greek documents. The authors used classical Agglomerative Hierarchical Cluster Analysis (HCA) with average linkage instead of deep learning. They applied a log-entropy weighting scheme to map author styles, achieving an average F-Bcubed score of 0.5733 and a Mean Average Precision (MAP) of 0.4554 \cite{b4}.

\subsection{Sequential Multi-Writer Segmentation}

Sequential multi-writer segmentation involves detecting and mapping the order of multiple distinct authors on one continuous page. Traditional biometric studies have only looked at single-author documents. To our knowledge, there is no published work, standardized framework, or dataset for this specific segmentation task within a document. This important gap in forensic document examination makes our proposed tracking pipeline a novel contribution.

\subsection{Positioning Our Proposed Framework}

Existing research on Bengali handwriting biometrics is largely restricted to closed-set, character- or word-level analysis. In contrast, we introduce a comprehensive, unified offline framework capable of executing verification, document retrieval, unsupervised clustering, and sequential multi-writer segmentation. Operating under a strict zero-shot, open-set paradigm with completely disjoint writer splits, we ensure true generalization using a massive, highly variable corpus aggregated from diverse data sources. Furthermore, rather than proposing an isolated architecture, we exhaustively benchmarked our models against established literature baselines—including classic CNNs (ResNet) and self-attention Transformers (ViT)—reporting parameters, FLOPs, storage sizes, and inference latencies to explicitly differentiate solutions for high-capacity computing resources like the cloud versus resource-constrained environments like edge deployment.

Methodologically, we diverge from standard "flat" (patch $\rightarrow$ page) document analysis. While prior retrieval studies utilized flat aggregation (e.g., NetVLAD pooling), our ablations demonstrate that a geometrically aware \textit{patch $\rightarrow$ line $\rightarrow$ page} hierarchical integration pipeline performs better. By preserving the horizontal spatial sequence of human penmanship, it successfully isolates biometric style from macro-level layout noise. 

Ultimately, this all-in-one framework is the first of its kind for open-set, document-level handwriting analysis in any language. It establishes the first comprehensive open-set baselines for offline Bangla full-page retrieval (evaluated via standard Leave-One-Out Top-$k$ and mAP metrics) and unsupervised writer clustering (adopting literature-proven Agglomerative Clustering). Most critically, it pioneers the task of sequential multi-writer segmentation—detecting and chronologically mapping multiple distinct authors within a single continuous page—thereby addressing a major, previously unexplored gap in forensic document examination.






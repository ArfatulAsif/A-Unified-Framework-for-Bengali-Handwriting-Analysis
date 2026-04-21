\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    citecolor=blue,
    linkcolor=blue,
    urlcolor=blue
}



\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\usepackage{tabularx}


\usepackage{makecell}




    
\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2026.DOI}

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

Automated handwriting analysis has various uses in forensic and archival processing. However, most frameworks are restricted to closed-set conditions and isolated character or word-level identification. These challenges are especially evident for complex scripts like Bengali (Bangla), where full-document analysis is extremely rare. In this work, we present a unified Deep Embedding Framework executing four offline tasks: writer verification, writer retrieval, unsupervised document clustering, and sequential multi-writer segmentation at the full-page level under strict zero-shot, open-set protocols. To isolate biometric style from page layout noise, we introduce a Patch-to-Line-to-Page hierarchical integration pipeline that preserves the spatial sequence of human penmanship. Addressing computational bottlenecks, we engineered a highly optimized custom CNN, the Dual-Path Patch Encoder (DPE-Net), and adapted the pretrained FasterNet-T0. Evaluated on a curated dataset of 435 writers, these models provide scalable solutions for edge and cloud environments. This framework establishes the first open-set document or page level baselines across all four operations for Bangla. For page-level verification, FasterNet-T0 and DPE-Net achieved 96.00\% and 95.00\% accuracy, respectively. In retrieval, they secured 100.00\% and 94.20\% Top-1 accuracies. For unsupervised clustering, our approach yielded pairwise accuracies of 98.64\% and 96.03\%. Finally, we pioneer the novel task of sequential multi-writer segmentation: detecting and mapping distinct authors collaborating on a single page. Here, FasterNet-T0 achieved 79.38\% absolute sequence accuracy, while DPE-Net provided a highly stable Sequence Error Rate of 0.0868. Ultimately, this unified framework significantly expands forensic document examination capabilities.


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

To solve this, we propose a unified framework powered by a highly optimized, custom CNN called the \textbf{Dual-Path Patch Encoder (DPE-Net)}. We also adapted and benchmarked the highly performant pretrained SOTA \textbf{FasterNet-T0\cite{b20}} across our entire framework, rigorously evaluating both models for model weight, storage signature, and inference speed. 

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

While there is a lot of research on semantic or information-based document clustering, biometric writer-based clustering is not as common. A 2018 study set an important offline baseline by tackling the open-set PAN 2017 Author Clustering Task with English, Dutch, and Greek documents. The authors used classical Agglomerative Hierarchical Cluster Analysis (HCA) with average linkage instead of deep learning. They applied a log-entropy weighting scheme to map author styles, achieving an average F-Bcubed score of 0.5733 and a Mean Average Precision (MAP) of 0.4554 \cite{b4}.

\subsection{Sequential Multi-Writer Segmentation}

Sequential multi-writer segmentation involves detecting and mapping the order of multiple distinct authors on one continuous page. Traditional biometric studies have only looked at single-author documents. To our knowledge, there is no published work, standardized framework, or dataset for this specific segmentation task within a document. This important gap in forensic document examination makes our proposed tracking pipeline a novel contribution.

\subsection{Positioning Our Proposed Framework}

Existing research on Bengali handwriting biometrics is largely restricted to closed-set, character- or word-level analysis. In contrast, we introduce a comprehensive, unified offline framework capable of executing verification, document retrieval, unsupervised clustering, and sequential multi-writer segmentation. Operating under a strict zero-shot, open-set paradigm with completely disjoint writer splits, we ensure true generalization using a massive, highly variable corpus aggregated from diverse data sources. Furthermore, rather than proposing an isolated architecture, we exhaustively benchmarked our models against established literature baselines—including classic CNNs (ResNet) and self-attention Transformers (ViT)—reporting parameters, FLOPs, storage sizes, and inference latencies to explicitly differentiate solutions for high-capacity computing resources like the cloud versus resource-constrained environments like edge deployment.

Methodologically, we diverge from standard "flat" (patch $\rightarrow$ page) document analysis. While prior retrieval studies utilized flat aggregation (e.g., NetVLAD pooling), our ablations demonstrate that a geometrically aware \textit{patch $\rightarrow$ line $\rightarrow$ page} hierarchical integration pipeline performs better. By preserving the horizontal spatial sequence of human penmanship, it successfully isolates biometric style from macro-level layout noise. 

Ultimately, this unified framework is the first of its kind for open-set, document-level handwriting analysis in any language. It establishes the first comprehensive open-set baselines for offline Bangla full-page retrieval (evaluated via standard Leave-One-Out Top-$k$ and mAP metrics) and unsupervised writer clustering (adopting literature-proven Agglomerative Clustering). Most critically, it pioneers the task of sequential multi-writer segmentation—detecting and chronologically mapping multiple distinct authors within a single continuous page—thereby addressing a major, previously unexplored gap in forensic document examination.








\section{Methodology}
\label{sec:methodology}

Section ~\ref{sec:methodology} details the complete pipeline of our proposed framework. Subsections A and B outline the dataset collection and our strict writer-disjoint evaluation protocols. Subsection C covers image preprocessing, followed by the definitions of our patch encoder architectures and training strategies in D and E. Subsections F and G detail the full-page line segmentation and hierarchical feature aggregation processes. Finally, H through K explicitly define the execution pipelines, algorithms, and evaluation metrics for our four core operations: writer verification, document retrieval, unsupervised clustering, and sequential multi-writer segmentation.

We did all our computational work, including data preprocessing and model training, with basic hardware settings on one NVIDIA RTX 3050 GPU. We used Python 3.9.13 and PyTorch 2.8.0, along with CUDA 12.8 and cuDNN 8.9.7.

\subsection{Dataset Collection and Preparation}
\label{subsec:dataset_collection}

To create a varied and inclusive Bangla handwriting dataset, we collected and combined data from three main sources:

\begin{enumerate}
    \item \textbf{BN-HTRd\cite{b18}:} Provided existing line-segmented images and full handwritten pages from 237 different writers.
    \item \textbf{WBSUBNdb\_text\cite{b19}:} Contributed handwritten pages from 188 writers. We manually processed and segmented lines from pages for each writer. Also, we manually checked each segmented line.
    \item \textbf{Custom Dataset:} We compiled an additional challenging set featuring 10 specialized writers. This custom subset was made to introduce extreme real-world challenges, including low-quality mobile phone scans and heavy shadows.
\end{enumerate}
 
During the combining process, we reviewed the data to ensure it reflected the unpredictable nature of real-world documents. We paid special attention to variations like unusually long and short lines, diverse page colors, and curved or slanted handwriting.

\textbf{Dataset Summary:}
\begin{itemize}
    \item \textbf{Total Writers:} 435
    \item \textbf{Total Handwritten Pages:} 2,825 (Average of ~7 pages per writer)
    \item \textbf{Total Segmented Lines:} 29,268 (Average of ~70 lines per writer)
\end{itemize}

\subsection{Writer-Disjoint Split and Evaluation Protocol}
\label{subsec:writer_disjoint}

We followed a strict, zero-shot, writer-disjoint evaluation protocol. This ensures the model will work well with unseen writers in real-world situations.

We divided the dataset of 435 writers into non-overlapping subsets, as shown in Table \ref{tab:dataset_partitioning}.



%-------- Table 1 start ------------------

\begin{table}[htbp]
\caption{Dataset Partitioning (writer disjoint) and Task Allocation.}
\label{tab:dataset_partitioning}
\centering
\small 
\setlength{\tabcolsep}{3pt}

\begin{tabularx}{\columnwidth}{|
>{\raggedright\arraybackslash}X|
c|
c|
>{\centering\arraybackslash}X|}
\hline

\textbf{Pipeline / Task} & \textbf{Phase} & \textbf{IDs} & \textbf{Data Level} \\
\hline

Patch Encoder & Training/val & 1--371 & Line \\
\hline

Verification & Evaluation @Pre=Rec & 372--403 & Line \& Page \\
\hline

Retrieval & Evaluation & 404--415 & Page \\
\hline

Clustering & Tuning & 372--403 & Page \\
\hline 

Clustering & Evaluation & 404--415 & Page \\
\hline

Sequential Multi-Writer Segmentation & Tuning & 395--420 & 196 Synthesized Pages* \\
\hline 

Sequential Multi-Writer Segmentation & Evaluation & 421--435 & 97 Synthesized Pages* \\
\hline

\end{tabularx}

\par\medskip
\raggedright
\footnotesize \textit{*Note: Synthesized pages made by cropping/merging segments from 1--4 writers to mimic multi-author pages. Used for tuning sequential clustering/smoothing and evaluating segmentation via SER.}
\end{table}


%-------- Table 1 end ------------------



\textbf{Standardized Ablation Cohort:}

To conduct comparisons with State-of-the-Art (SOTA) methods and to enable detailed studies, we used a subset of 100 writers. This group was divided into 80 train/validation writers and 20 test writers for line-level tasks, with 10 of those test writers set aside for page-level evaluations.













\subsection{Preprocessing Handwriting Lines}

We subjected each segmented handwriting line to a multi-stage signal-conditioning and geometric-normalization pipeline, as illustrated in Figure~\ref{fig:preprocessing}.


\begin{figure*}[t!]
\centering
\includegraphics[width=\textwidth]{fig/preprocessing.png}
\caption{Multi-Stage Signal Conditioning and Geometric Normalization Pipeline}
\label{fig:preprocessing}
\end{figure*}



Let the raw grayscale handwriting line be denoted as $I_{raw}$. First, to maximize ink visibility against degraded backgrounds, we applied Contrast Limited Adaptive Histogram Equalization (CLAHE) to produce an enhanced image, $I_{clahe}$. 

To eliminate arbitrary background margins, we generated a binary adaptive text mask $M$ from $I_{clahe}$. By computing the smoothed 1D pixel projections along the horizontal and vertical axes of $M$, we isolated the tightest bounding coordinates $[y_0, y_1]$ and $[x_0, x_1]$ that contained valid ink. We then cropped the enhanced image to these coordinates to yield the content-only image:

\begin{equation}
I_{crop} = I_{clahe}[y_0:y_1, x_0:x_1]
\label{eq:crop}
\end{equation}

Next, we normalized the physical scale of the handwriting. $I_{crop}$ was resized to a fixed target height $H_{target} = 128$ pixels while strictly preserving its original aspect ratio, resulting in a resized image $I_{res}$ of width $W_{new}$. To unify the tensor dimensions for batch processing without distorting the handwriting geometry, $I_{res}$ was placed onto a fixed-size canvas $C \in \mathbb{R}^{H_{target} \times W_{max}}$, where the maximum width $W_{max} = 1580$. During training, we actuated spatial data augmentation by placing $I_{res}$ at a random horizontal offset (\verb|place_train="random"|), whereas during evaluation, we centered it deterministically (\verb|place_eval="center"|). The canvas was then normalized to a continuous float range $[0, 1]$.

\textbf{Patch Extraction}

Because handwriting lines vary drastically in length, we modeled each line as a sequence of localized, overlapping visual patches. Let a single patch be mathematically designated as $p_k \in \mathbb{R}^{S_{patch} \times S_{patch} \times 1}$, where the patch size $S_{patch} = 128$. 

We extracted these patches using a sliding window approach along the horizontal axis of the valid content region $[x_{start}, x_{end}]$ of the canvas $C$. A patch $p_k$ at step $k$ was extracted starting at coordinate $x_k = x_{start} + k \cdot S$, where the stride length $S = 56$:

\begin{equation}
p_k = C[:, x_k : x_k + S_{patch}]
\label{eq:patch}
\end{equation}

To prevent the model from processing empty background space, we enforced a strict foreground density constraint. A patch $p_k$ was only appended to the final sequence if its ink ratio exceeded a minimum threshold $\tau$. By thresholding $p_k$ via Otsu's method, we defined the foreground indicator function $f(p_k)$, and strictly enforced:

\begin{equation}
f(p_k) \geq \tau_{min}
\label{eq:threshold}
\end{equation}

where $\tau_{min} = 0.04$. The final preprocessed output for a single handwriting line was the sequence of valid, highly dense patches $P_{line} = \{p_1, p_2, \dots, p_N\}$, which was subsequently passed to the patch encoder. 

\textit{(Note: An ablation study validating the impact of this preprocessing pipeline is provided in Section \ref{subsubsec:preprocessing}.}





\subsection{Patch Encoder Architecture}
\label{subsec:patch_encoder}

To extract distinct features from the processed handwriting patches, we aimed to build a biometric pipeline that achieves high accuracy while being computationally efficient. Practical document analysis requires fast, localized processing. Therefore, we created the \textbf{DPE-Net (Dual-Path Patch Encoder)}, a custom architecture designed for minimal computational load and quick inference without losing verification performance.

Alongside our custom network, we thoroughly tested several modern State-of-the-Art (SOTA) architectures to set a comparative baseline. Among them, the pretrained \textbf{FasterNet-T0} stood out as a strong alternative that also meets high-speed deployment needs.

To rigorously check these models’ practicality for large-scale analysis, we measured architectural complexity (Total Parameters, FLOPs) and processing speed (Inference Latency per Line) against Line-Level Accuracy and AUC to ensure dependable biometric verification. The detailed comparative results of these architectures can be found in Section \ref{subsec:architectural_benchmarking} (Table \ref{tab:table1}). The isolated study of our custom architecture's internal components is provided in Section \ref{subsubsec:architecture_pooling} (Table \ref{tab:table3}).





\subsubsection{DPE-Net (Dual-Path Patch Encoder) Model Architecture}
\label{subsubsec:dpenet}

To capture both the fine details of ink deposition and the overall flow of handwriting, we designed a custom Dual-Path Convolutional Neural Network, as shown in Figure~\ref{fig:architecture_integrated}. Instead of relying on deep, parameter-heavy sequential layers, this architecture employed two parallel branches that processed a shared initial feature map.



\begin{figure*}[t!]
\centering
\includegraphics[width=\textwidth, keepaspectratio]{fig/architecture_integrated.png}
\caption{DPE-Net (Dual-Path Patch Encoder) Architecture Diagram.}
\label{fig:architecture_integrated}
\end{figure*}






The network received an input patch tensor of dimensions $128 \times 128 \times 1$. The processing pipeline was set up as follows:

\begin{enumerate}
    \item \textbf{Shared Stem:} The input patch first went through a downsampling stem made of a $3 \times 3$ convolutional layer with a stride of 2 and padding of 1, followed by 2D Batch Normalization and an in-place ReLU activation. This cut down the spatial dimensions to $64 \times 64$ while projecting the features into 32 channels.
    \item \textbf{Standard Branch (Local Textures):} The first parallel pathway had two sequential blocks of $3 \times 3$ convolutions (stride 2, padding 1), Batch Normalization, and ReLU activations. This standard convolutional branch was tuned to capture localized micro-textures and intricate stroke variations, producing a 64-channel feature map.
    \item \textbf{Dilated Branch (Long-Range Strokes):} The second parallel pathway mirrored the structure of the first but used dilated convolutions. It applied two sequential blocks of $3 \times 3$ convolutions with a stride of 2, a padding of 2, and a dilation rate of 2. This dilation expanded the receptive field of the $3 \times 3$ kernel to an effective $5 \times 5$ area without increasing the number of trainable parameters. This branch was made to trace long-range stroke connections and continuous patterns, resulting in a secondary 64-channel feature map.
    \item \textbf{Feature Fusion and Patch Embedding:} The outputs from both parallel branches were joined along the channel dimension to form a 128-channel feature map. We applied 2D Global Average Pooling (GAP) to reduce the spatial dimensions, followed by a dense Linear layer that projected the features into the final $128$-dimensional patch embedding vector.
\end{enumerate}

\textbf{Line-Level Feature Aggregation:}
Since handwriting lines can vary in length, they were modeled as a sequence of $K$ valid patches, $P_{line} = \{p_1, p_2, \dots, p_K\}$. The DPE-Net, shown here as the embedding function $f_{\theta}$, processed these patches simultaneously to extract a corresponding sequence of patch embeddings $E = \{e_1, e_2, \dots, e_K\}$, where each $e_k = f_{\theta}(p_k) \in \mathbb{R}^{D}$ and the embedding dimension $D = 128$.

To combine these localized textual and geometric features into a single biometric descriptor, we used Mean Pooling across the $K$ dimension, calculating the raw line-level representation $v$:

\begin{equation}
v = \frac{1}{K} \sum_{k=1}^{K} e_k
\label{eq:mean_pooling}
\end{equation}

Finally, to stabilize the distance metric calculations needed for the Triplet Margin Loss optimization, we applied $L_2$ normalization to the aggregated descriptor $v$. This mathematically projected the final line embedding $\hat{v}$ onto a unit hypersphere $\mathbb{S}^{D-1}$:

\begin{equation}
\hat{v} = \frac{v}{\|v\|_2}
\label{eq:l2_norm}
\end{equation}

where $\|\cdot\|_2$ represents the standard Euclidean norm. This normalized vector $\hat{v}$ provided the final, distinct feature representation of the entire handwriting line.

\subsubsection{Pretrained FasterNet-T0}
\label{subsubsec:fasternet}

As a high-performance alternative to our custom DPE-Net, we included the FasterNet-T0\cite{b20} architecture as a main State-of-the-Art (SOTA) encoder in our pipeline. We chose FasterNet for its excellent balance of high accuracy and quick processing speeds. It relied on a high-speed CNN model designed to optimize floating-point operations per second (FLOPs) by using partial convolutions (PConv).

For our framework, we used the FasterNet-T0 variant initialized with weights pretrained on the ImageNet dataset to ensure strong, generalized foundational feature extraction. We adjusted the network for single-channel grayscale inputs by averaging the pretrained RGB weights across the first convolutional layer and substituting the standard classification head with a $128$-dimensional linear projection layer. Like our custom architecture, the FasterNet-T0 processed sequences of $K$ patches and combined them via Mean Pooling and $L_2$ Normalization to create the final line-level biometric embedding.

\subsubsection{Other SOTA Encoders Considered}
\label{subsubsec:other_encoders}

To thoroughly validate our choice of \textbf{DPE-Net} and \textbf{FasterNet-T0\cite{b20}} as our main operational models, we built a comparison group that reflects different architectural styles. This group includes several models that are commonly discussed in the literature. We examined classic deep convolutional networks (\textbf{Pretrained ResNet-18\cite{b21}} [Classic CNN]), mobile-optimized lightweight models (\textbf{Pretrained MobileNetV4 Conv-S\cite{b22}} [Mobile CNN]), and pure self-attention systems (\textbf{Pretrained ViT-Tiny\cite{b23}} [Pure Transformer]). We also looked into several cutting-edge hybrid models that combine CNNs and Transformers to balance local feature extraction with global context. These included \textbf{Pretrained EdgeNeXt-XXS\cite{b24}} [Hybrid Edge], \textbf{Pretrained EfficientViT-M0\cite{b25}} [Hybrid Speed], and custom hybrid arrangements linking these pretrained models with Single-Path aggregation heads (e.g., Pretrained EdgeNeXt-XXS + Single-Path (Strided + GAP) and Pretrained FasterNet-T0 + Single-Path (Strided + GAP).












\subsection{Patch Encoder Model Training}
\label{subsec:model_training}

To optimize the feature extraction capabilities of our patch encoder, we used a metric learning approach. This method aimed to clearly distinguish between different writers’ styles while grouping variations from the same writer. The entire data flow for generating the encoding of a line is shown in Figure \ref{fig:training_setup}.




\begin{figure*}[t!]
\centering
\includegraphics[width=\textwidth]{fig/training_setup.png}
\caption{End-to-End Data Flow for Generating Line-Level Biometric Encodings.}
\label{fig:training_setup}
\end{figure*}



% \begin{figure}[t!]
% \centering
% \includegraphics[width=\columnwidth]{fig/training_setup.png}
% \caption{Illustration of the data flow and Siamese architecture during the training phase.}
% \label{fig:training_setup}
% \end{figure}


To train the network, we utilized a \textbf{Triplet Siamese Architecture\cite{b26,b27}}. During each training step, the model processed three distinct handwriting samples simultaneously: an Anchor ($x_a$), a Positive ($x_p$), and a Negative ($x_n$). The Anchor and Positive samples were distinct lines drawn from the same writer, while the Negative sample was drawn from a different, randomly selected writer. 

Following the feature aggregation step described in Section \ref{subsubsec:dpenet}, the network output $L_2$-normalized line-level embeddings for each branch, denoted as $\hat{v}_a$, $\hat{v}_p$, and $\hat{v}_n$, respectively. 

\textbf{Training Objective (Loss Function):}
To optimize the embedding space, we utilized the Triplet Margin Loss based on squared Euclidean distances. The objective is to minimize the distance between the Anchor and Positive embeddings while maximizing the distance between the Anchor and Negative embeddings by at least a predefined margin $\alpha$. The loss function $\mathcal{L}_{triplet}$ is mathematically defined as:

\begin{equation}
\mathcal{L}_{triplet} = \frac{1}{B} \sum_{i=1}^{B} \max \left(0, \|\hat{v}_{a,i} - \hat{v}_{p,i}\|_2^2 - \|\hat{v}_{a,i} - \hat{v}_{n,i}\|_2^2 + \alpha \right)
\label{eq:triplet_loss}
\end{equation}

where $B$ represents the batch size, and $\alpha = 0.4$ enforces a strict margin of separation between identical and distinct writers. 

\subsubsection{Optimization and Hyperparameters}
\label{subsubsec:hyperparameters}

To avoid data leakage and ensure the model could generalize, we divided the training dataset into an 80/20 writer-disjoint setup (\verb|val_split_by_writers = 0.2|). During training, each handwriting line was dynamically represented by randomly sampling $K = 8$ spatial patches (\verb|patches_per_line = 8|). This random sampling served as a strong form of spatial data augmentation, preventing the network from memorizing fixed sequence locations.

The model was optimized using the \textbf{Adam optimizer} with an initial learning rate of $\eta = 0.0005$. To dynamically adjust the learning rate as the model converged, we implemented a \verb|ReduceLROnPlateau| learning rate scheduler monitoring the validation loss. The scheduler was configured to halve the learning rate (factor of $0.5$) if the validation loss plateaued for 3 consecutive epochs, down to a minimum bound of $10^{-6}$.

To maximize efficiency on our NVIDIA RTX 3050 GPU, we trained the network with a batch size of $64$ triplets. We also enabled Automatic Mixed Precision (AMP) through PyTorch's \verb|GradScaler|, which computes gradients in \verb|float16| while keeping \verb|float32| weight updates. This method significantly reduced VRAM usage without losing stability.

The network was trained for a maximum of $50$ epochs. To prevent overfitting, we enforced early stopping with a patience of $10$ epochs. To maintain complete reproducibility for all runs, we set the environment, data split, and model initializations to a global random seed of $42$.

\subsection{Page-Level Line Segmentation Pipeline}
\label{subsec:page_segmentation}

To assess our framework on entire unconstrained page documents, we needed a solid pipeline to extract individual handwriting lines systematically. Because our biometric encoder relied on stroke geometry rather than semantic meaning, we bypassed computationally heavy Optical Character Recognition (OCR) text decoding in favor of a high-speed, scale-normalized approach:

\begin{enumerate}
    \item \textbf{Scale-Normalized Detection:} The document was dynamically resized (preserving aspect ratio) to a maximum dimension of 1024 pixels. It was then passed through a CRAFT-based detection module. By applying a binary threshold to the resulting character and affinity heatmaps and extracting bounding coordinates via connected component analysis, we extracted raw horizontal bounding boxes, skipping the character recognition phase entirely.
    \item \textbf{Adaptive Tolerance Grouping:} Fragmented word boxes were grouped into continuous horizontal lines based on their vertical center coordinates. Two adjacent boxes were merged if their vertical distance was strictly less than an adaptive tolerance threshold ($\tau = 0.5$) scaled by the height of the preceding box.
    \item \textbf{Margin Extraction:} The aggregated line coordinates were projected back to the original high-resolution scale, padded with a 4-pixel margin to preserve extreme ascenders and descenders, and cropped.
\end{enumerate}

The choice of this particular detection and adaptive grouping method came from thorough testing. A detailed comparison of this pipeline against other segmentation methods is provided in the ablation study in Section \ref{subsubsec:segmentation_ablation} (Table \ref{tab:table4}).








\subsection{Hierarchical Local-to-Global Feature Aggregation}
\label{subsec:hierarchical_aggregation}

To represent a full page, we compress the extracted visual information into a single, highly discriminative biometric vector. We achieved this through a structured, hierarchical aggregation pipeline (Patch $\rightarrow$ Line $\rightarrow$ Page).

Mathematically, let a full document page $D$ be segmented into $L$ valid handwriting lines, $D = \{l_1, l_2, \dots, l_L\}$. As established in Section \ref{subsubsec:dpenet}, each line $l_i$ is represented by a sequence of $K$ patches, where the encoder mapped each patch to an embedding $e_{i,k}$.

First, we aggregated the localized patches to form the line-level representation $v_i$ via mean pooling:

\begin{equation}
v_i = \frac{1}{K} \sum_{k=1}^{K} e_{i,k}
\label{eq:line_pooling}
\end{equation}

Next, we integrated the sequential line vectors to form the global raw page representation $V$ by applying a second tier of mean pooling across the $L$ dimension:

\begin{equation}
V = \frac{1}{L} \sum_{i=1}^{L} v_i
\label{eq:page_pooling}
\end{equation}

Finally, to stabilize distance metric computations, the aggregated page descriptor was subjected to $L_2$ normalization, projecting the final biometric signature $\hat{V}$ onto a unit hypersphere:

\begin{equation}
\hat{V} = \frac{V}{\|V\|_2}
\label{eq:page_l2_norm}
\end{equation}

This hierarchical approach structurally preserved the horizontal stroke sequences inherent to human handwriting. By enforcing a Line-Level intermediary, the network forced the patches to maintain their sequential spatial context. This geometry-aware integration proved superior to "Flat" aggregation strategies (Patch $\rightarrow$ Page), which bypassed line segmentation and treated the document as an unordered "bag of patches." The complete experimental validation justifying this hierarchical selection over flat pooling networks is provided in the ablation study in Section \ref{subsubsec:feature_integration_ablation} (Table \ref{tab:table5}).

\subsection{Evaluation Metrics and Threshold Determination}
\label{subsec:evaluation_metrics}

To objectively evaluate our framework on writer verification tasks (determining whether two handwriting samples belong to the same author), we operated in a biometric distance space rather than a direct classification space.

Let $e_1$ and $e_2$ represent the $L_2$-normalized feature embeddings of two given handwriting samples (either at the line or page level), The dissimilarity between these samples was computed using Cosine Distance, defined mathematically as:

\begin{equation}
D_{cos}(e_1, e_2) = 1 - (e_1 \cdot e_2)
\label{eq:cosine_distance}
\end{equation}

Because all feature vectors are $L_2$-normalized prior to distance calculation, minimizing the Squared Euclidean distance during Triplet Loss optimization mathematically translates directly to maximizing Cosine similarity during evaluation. A binary prediction was made by comparing this distance against a decision threshold $t$. If $D_{cos} \le t$, the samples were classified as a positive pair (same writer); otherwise, they were classified as a negative pair (different writers).

To quantitatively assess the framework's verification performance, we defined the standard binary classification outcomes specifically in the context of writer pairing:
\begin{itemize}
    \item \textbf{True Positives ($TP$):} Same-writer pairs correctly classified as a match ($D_{cos} \le t$).
    \item \textbf{True Negatives ($TN$):} Different-writer pairs correctly classified as non-matches ($D_{cos} > t$).
    \item \textbf{False Positives ($FP$):} Different-writer pairs incorrectly classified as a match (False Acceptance).
    \item \textbf{False Negatives ($FN$):} Same-writer pairs incorrectly classified as non-matches (False Rejection).
\end{itemize}

Based on these defined biometric pairing outcomes, the primary performance metrics are mathematically formulated as follows:


\begin{subequations}\label{eq:metrics}

\begin{align}
Accuracy &= \frac{TP + TN}{TP + TN + FP + FN} \label{eq:accuracy} \\
Precision\ (P) &= \frac{TP}{TP + FP} \label{eq:precision} \\
Recall\ (R) &= \frac{TP}{TP + FN} \label{eq:recall} \\
F1\text{-}Score &= 2 \cdot \frac{P \cdot R}{P + R} \label{eq:f1}
\end{align}

\end{subequations}


\textbf{Dynamic Threshold Determination and Balanced Evaluation:}
A critical challenge in open-set verification is that relying on a statically predefined threshold is highly susceptible to dataset bias. Furthermore, skewed evaluation sets can artificially inflate performance metrics. To prevent this, our validation and testing protocols were strictly constructed using an exactly equal number of positive (same-writer) and negative (different-writer) pairs, guaranteeing a perfectly balanced evaluation devoid of class-imbalance artifacts.

To ensure our decision boundary was rigorously generalizable, we dynamically determined the optimal operating threshold $t^*$ using the disjoint validation cohort prior to final testing. We performed a fine-grained continuous threshold sweep across the absolute distance range $t \in [0.0, 2.0]$. For each discrete step, we computed standard evaluation metrics: Accuracy, Precision ($P$), and Recall ($R$). We defined the optimal threshold $t^*$ as the Break-Even Point—the exact piecewise-linear intersection where Precision equals Recall:

\begin{equation}
t^* = \{ t \mid P(t) = R(t) \}
\label{eq:threshold_opt}
\end{equation}

By locking the threshold at this point of equilibrium, we guaranteed the model was benchmarked at its most balanced operational state. This established threshold was then strictly applied to the entirely unseen test set to compute the final, reported metrics: Accuracy, Precision, Recall, F1-Score, and the threshold-independent Area Under the ROC Curve (AUC).







\subsection{Writer-Based Document Retrieval}
\label{subsec:document_retrieval}

To evaluate our framework’s performance in retrieval space, we implemented a Leave-One-Out (LOO) ranking protocol, independently assessing the retrieval pipelines driven by both our custom DPE-Net (Dual-Path Patch Encoder) and the Pretrained FasterNet-T0.

\subsubsection{Retrieval Protocol and Ranking Strategy}
\label{subsubsec:retrieval_protocol}

Let the evaluation dataset consist of a closed set of full document pages, where each page is preprocessed and passed through our hierarchical integration pipeline (utilizing either DPE-Net or FasterNet-T0 as the foundational feature extractor) to produce an $L_2$-normalized global page embedding, $\hat{V}$. The entire dataset can be mathematically defined as a pool of document-embedding pairs, $\mathcal{D} = \{(w_1, \hat{V}_1), (w_2, \hat{V}_2), \dots, (w_N, \hat{V}_N)\}$, where $w_i$ represents the ground-truth writer identity of the $i$-th document.

During the LOO evaluation, every single document in $\mathcal{D}$ was sequentially isolated and treated as a query, $q = (w_q, \hat{V}_q)$. The remaining documents formed the search gallery, $\mathcal{G} = \mathcal{D} \setminus \{q\}$. For each query, we calculated the dissimilarity between the query embedding $\hat{V}_q$ and every gallery embedding $\hat{V}_j \in \mathcal{G}$ using the Cosine Distance:

\begin{equation}
D_{cos}(\hat{V}_q, \hat{V}_j) = 1 - (\hat{V}_q \cdot \hat{V}_j)
\label{eq:retrieval_distance}
\end{equation}

The gallery documents were then sorted in ascending order of their distance to the query, generating a ranked retrieval list $R_q$. To ensure mathematical validity, any query belonging to a writer with only a single document in the entire dataset (meaning $0$ relevant matches exist in $\mathcal{G}$) was excluded from the final metric aggregation.

\subsubsection{Evaluation Metrics}
\label{subsubsec:retrieval_metrics}

To quantitatively assess the ranking quality of our network, we utilized three standard retrieval metrics: Top-1 Accuracy, Top-5 Accuracy, and Mean Average Precision (mAP). 

\textbf{Top-$k$ Accuracy:}
This metric measures the probability that at least one highly relevant document appears within the uppermost results. A query $q$ is considered a "hit" for Top-$k$ accuracy if at least one document in the first $k$ ranks of $R_q$ shares the exact same writer identity ($w_q$). We formally reported Top-1 (the absolute closest match) and Top-5 accuracy to evaluate the model's immediate precision.

\textbf{Mean Average Precision (mAP):}
While Top-$k$ accuracy indicates if \textit{any} match was found early, it does not evaluate the model's ability to cluster \textit{all} documents by the same writer together. To evaluate overall ranking quality, we calculated the Mean Average Precision.

First, we computed the Average Precision ($AP$) for a single query $q$. Let $N_q$ represent the total number of relevant documents (true matches) existing in the gallery $\mathcal{G}$. Let $P(r)$ denote the cumulative precision calculated at rank $r$, and let the indicator function $rel(r) \in \{0,1\}$ equal $1$ if the document at rank $r$ is a true match, and $0$ otherwise. The $AP$ for query $q$ is mathematically defined as:

\begin{equation}
AP_q = \frac{1}{N_q} \sum_{r=1}^{|\mathcal{G}|} P(r) \cdot rel(r)
\label{eq:average_precision}
\end{equation}

This formulation heavily penalizes models that rank true matches lower down the list, as the precision fraction $P(r)$ drops as $r$ increases. Finally, the mAP was computed by averaging the $AP$ scores across all valid queries in the evaluation set $\mathcal{Q}$:

\begin{equation}
mAP = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} AP_q
\label{eq:map}
\end{equation}






\subsection{Handwriting-Based Document Clustering}
\label{subsec:document_clustering}

In unsupervised document clustering, an investigator is presented with a large, unlabelled corpus of documents and must autonomously group them such that each distinct cluster corresponds to a unique, unknown author. To execute this, we leveraged the global page-level embeddings extracted by our hierarchical pipeline mentioned in section \ref{subsec:hierarchical_aggregation}, independently benchmarking the latent spaces generated by both our custom \textbf{DPE-Net} and the pretrained \textbf{FasterNet-T0}.

\subsubsection{Algorithm Selection and Distance Formulation}
\label{subsubsec:clustering_algorithm}

Let an unlabelled corpus consist of $N$ document pages, yielding a set of $L_2$-normalized embeddings $\mathcal{D} = \{\hat{V}_1, \hat{V}_2, \dots, \hat{V}_N\}$. Because traditional clustering algorithms operating in high-dimensional Euclidean space are susceptible to the curse of dimensionality, we explicitly utilized a precomputed Cosine Distance matrix. We constructed a symmetric $N \times N$ distance matrix $M_{dist}$, where each element represents the dissimilarity between two documents:

\begin{equation}
M_{dist}(i, j) = 1 - (\hat{V}_i \cdot \hat{V}_j)
\label{eq:clustering_distance}
\end{equation}

To partition this distance space, we evaluated multiple unsupervised clustering methodologies. As detailed in the ablation study in Section \ref{subsubsec:clustering_ablation} (Table \ref{tab:table6}), \textbf{Agglomerative Hierarchical Clustering} outperformed density-based algorithms such as DBSCAN. 

We applied Agglomerative Clustering using \textbf{average linkage}, which merges pairs of clusters based on the average cosine distance between all respective member embeddings. Rather than forcing the algorithm to find a predefined number of clusters ($k$), we controlled the cluster formation dynamically using a maximum distance threshold ($\tau_{cluster}$). If the average distance between two clusters exceeded $\tau_{cluster}$, the merging process halted. 

Pairwise accuracy was chosen as the main tuning metric because it ensures that the resulting clusters keep strong biometric purity for real-world use.

Because the geometric distribution of the latent space naturally varies between different neural architectures, this optimal stopping threshold was determined independently for each feature encoder. By executing a comprehensive threshold sweep on the disjoint tuning cohort, we selected the thresholds that maximized pairwise Accuracy—yielding $\tau_{cluster} = 0.150$ for the \textbf{DPE-Net} and $\tau_{cluster} = 0.110$ for the \textbf{FasterNet-T0.}

\subsubsection{Clustering Evaluation Metrics}
\label{subsubsec:clustering_metrics}

To thoroughly evaluate the quality of the unsupervised clustering, we applied two distinct classes of evaluation metrics: Global Partitioning Metrics and Pairwise Assignment Metrics.

Let $W = \{w_1, w_2, \dots, w_N\}$ denote the ground-truth writer identities for the corpus, and let $C = \{c_1, c_2, \dots, c_N\}$ denote the discrete cluster labels assigned by the algorithm. 

\textbf{Global Partitioning Metrics:}
To evaluate the structural integrity of the clusters against the ground-truth classes, we utilized the Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
\begin{itemize}
    \item \textbf{NMI} measures the mutual dependence between the ground-truth writer distributions and the predicted clusters, normalized by their combined entropy to account for varying cluster sizes. It is defined as:

    \begin{equation}
    NMI(W, C) = \frac{2 \cdot I(W; C)}{H(W) + H(C)}
    \label{eq:nmi}
    \end{equation}
  
    where $I$ is the mutual information and $H$ is the Shannon entropy.
    \item \textbf{ARI} measures the similarity between the two data clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters, mathematically adjusted for chance grouping:

    \begin{equation}
    ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}
    \label{eq:ari}
    \end{equation}
  
    where $RI$ is the raw Rand Index and $E[RI]$ is its expected value.
\end{itemize}

\textbf{Pairwise Assignment Metrics:}
While ARI and NMI evaluate global dataset structure, real-world application demands a granular understanding of pairwise matching accuracy. We constructed a binary ground-truth matrix where a true pair $y_{i,j} = 1$ if $w_i = w_j$. Similarly, we generated a binary prediction matrix where $\hat{y}_{i,j} = 1$ if $c_i = c_j$. 

By evaluating the upper triangular elements of these matrices (representing all $\frac{N(N-1)}{2}$ unique document combinations), we redefined the standard classification outcomes for the clustering domain:
\begin{itemize}
    \item \textbf{True Positives ($TP$):} Documents by the same writer correctly placed in the same cluster.
    \item \textbf{True Negatives ($TN$):} Documents by different writers correctly placed in different clusters.
    \item \textbf{False Positives ($FP$):} Documents by different writers incorrectly grouped into the same cluster.
    \item \textbf{False Negatives ($FN$):} Documents by the same writer incorrectly separated into different clusters.
\end{itemize}

Using these clustering-specific outcomes, we computed the pairwise metrics utilizing the standard formulas:



\begin{subequations}\label{eq:metrics}

\begin{align}
Accuracy &= \frac{TP + TN}{TP + TN + FP + FN} \label{eq:accuracy} \\
Precision\ (P) &= \frac{TP}{TP + FP} \label{eq:precision} \\
Recall\ (R) &= \frac{TP}{TP + FN} \label{eq:recall} \\
F1\text{-}Score &= 2 \cdot \frac{P \cdot R}{P + R} \label{eq:f1}
\end{align}

\end{subequations}









\subsection{Sequential Multi-Writer Segmentation}
\label{subsec:sequential_segmentation}

Most handwriting analysis research assumes that a single document page is written entirely by one person. While our clustering pipeline (Section \ref{subsec:document_clustering}) easily detects multiple writers across a stack of distinct pages, a more complex challenge is \textit{multi-author page composition}—when a single, continuous page contains multiple paragraphs written by different authors. To solve this, we developed a Sequential Multi-Writer Segmentation pipeline to accurately detect and map the top-to-bottom sequence of different writers on the same page. A visual demonstration of this pipeline, depicting the transition from raw multi-writer inputs to the final segmented outputs, is provided in Figure \ref{fig:multi_writer_segmentation}.



\begin{figure}[t!]
\centering
\includegraphics[width=\columnwidth]{fig/multi_writer_segmentation.png}
\caption{Example Input and Output of the Sequential Multi-Writer Segmentation Pipeline with Different Colors for Different Writers.}
\label{fig:multi_writer_segmentation}
\end{figure}



To the best of our knowledge, there is no standardized framework or dataset for this specific task. To benchmark our pipeline, we manually curated a highly realistic evaluation dataset. We extracted horizontal paragraph crops from distinct writers and vertically merged them to simulate single, continuous pages containing between one and four different authors. To ensure these synthesized images visually replicated authentic, untouched documents, we carefully color-matched the backgrounds across all merged crops to eliminate distinct visual seams. 

This dataset was designed to strictly test the algorithm's ability to track chronological order and re-identify previous writers. It includes both linear progressions (e.g., Writer A\_B\_C\_D) and alternating recurrences (e.g., Writer A\_B\_A\_C).

\subsubsection{Algorithm Selection and Sequence Reconstruction}
\label{subsubsec:algorithm_selection_sequence}

Let a synthesized multi-writer page yield a chronological, top-to-bottom sequence of valid handwriting lines, $L = \{l_1, l_2, \dots, l_K\}$. Passing these lines through our hierarchical pipeline (using either DPE-Net or FasterNet-T0) generates a corresponding sequence of $L_2$-normalized line embeddings, $E = \{\hat{v}_1, \hat{v}_2, \dots, \hat{v}_K\}$.

To cluster these sequential embeddings into contiguous writer blocks, we evaluated multiple unsupervised clustering algorithms. As detailed in the ablation study in Section \ref{subsubsec:sequential_ablation} (Table \ref{tab:table7}), we compared \textbf{DBSCAN}  against the Agglomerative Clustering. DBSCAN proved superior for sequential segmentation. While Agglomerative Clustering forces all embeddings into discrete groups regardless of transitional ambiguity, DBSCAN’s density-reachability logic inherently isolates ambiguous line boundaries as "noise" (label $-1$), naturally forming highly stable, contiguous spatial blocks for the core handwriting text.

The raw chronological cluster assignments generated by DBSCAN, denoted as $C_{raw} = \{c_1, c_2, \dots, c_K\}$, were subsequently processed through a two-stage sequential reconstruction pipeline:

\begin{enumerate}
    \item  \textbf{Temporal Smoothing:} Because short or heavily degraded handwriting lines can cause isolated clustering misclassifications, we applied a moving majority filter to $C_{raw}$. Using a temporal window of size $w_{size}$, the label at step $i$ was reassigned to the most frequent cluster ID (the statistical mode) within its local neighborhood. This effectively filters out single-line spatial anomalies. For example, a raw sequence containing an isolated error, such as $(c_1, c_1, c_1, c_2, c_1, c_1)$, is smoothed to $(c_1, c_1, c_1, c_1, c_1, c_1)$, yielding the cleaned sequence $C_{smooth}$.
    
    \item  \textbf{State Compression:} To reconstruct the final high-level author transitions, we compressed the redundant line-by-line labels in $C_{smooth}$ into discrete author blocks. Consecutive identical cluster labels were collapsed into a single state representation, and any unassigned noise points ($-1$) were entirely discarded. For instance, a long sequence of identical line clusters such as $(c_1, c_1, c_1, c_1, c_1, c_2, c_2, c_2, c_3, c_3, c_3, c_3, c_1, c_1, c_1, c_1)$ compresses down to the final predicted chronological transition sequence, $S_{pred} = (c_1, c_2, c_3, c_1)$.
\end{enumerate}

\subsubsection{Hyperparameter Tuning for Spatial Encoders}
\label{subsubsec:hyperparameter_tuning}

Because DBSCAN relies heavily on density thresholds, its performance is highly sensitive to the specific geometric distribution of the underlying latent space. Therefore, the clustering and smoothing hyperparameters—Maximum Neighborhood Distance ($\epsilon$), Minimum Samples ($\mu_{samples}$), and Smoothing Window Size ($w_{size}$)—were tuned independently for each patch encoder using our disjoint multi-writer tuning cohort. 

We performed an exhaustive multidimensional grid search to minimize the sequence error. The optimal parameters were determined as follows:
\begin{itemize}
    \item \textbf{For DPE-Net:} $\epsilon = 0.09$, $\mu_{samples} = 3$, and $w_{size} = 5$.
    \item \textbf{For FasterNet-T0:} $\epsilon = 0.10$, $\mu_{samples} = 2$, and $w_{size} = 5$.
\end{itemize}

The stabilization of both models at $w_{size} = 5$ confirms that DBSCAN inherently produces contiguous groupings that require only minimal local smoothing, circumventing the need for aggressive post-hoc corrections.

\subsubsection{Sequence Evaluation Metrics}
\label{subsubsec:sequence_evaluation}

To quantitatively evaluate the success of the sequential segmentation, we measured the deviation between the predicted chronological sequence $S_{pred}$ and the ground-truth sequence $S_{gt}$ using the \textbf{Levenshtein Distance ($LD$)}. 

The Levenshtein Distance calculates the minimum number of single-element edit operations (insertions, deletions, or substitutions) required to perfectly transform the predicted sequence into the ground-truth sequence. Based on this edit distance, we reported two primary metrics:

\textbf{1. Sequence Error Rate (SER):}
SER normalizes the Levenshtein Distance by the total number of true transitions on the page. It provides a granular measurement of the segmentation error per document and is mathematically defined as:

\begin{equation}
SER = \frac{LD(S_{pred}, S_{gt})}{|S_{gt}|}
\label{eq:ser}
\end{equation}

where $|S_{gt}|$ is the length of the ground-truth transition sequence. The reported SER is the average across all pages in the evaluation dataset.

\textbf{2. Absolute Sequence Accuracy:}
While SER measures partial success, practical document examination often requires flawless sequence reconstruction. We defined Absolute Sequence Accuracy as the percentage of total pages where the chronological sequence was reconstructed without a single error (i.e., $LD = 0$). This metric serves as the strictest benchmark of the pipeline's capability to untangle multi-author page composition.









\section{Architectural Benchmarking and Ablation Studies}
\label{sec:benchmarking_and_ablation}

To test our model components efficiently while ensuring fair and rigorous comparisons, all benchmarking and ablation experiments strictly adhered to the hardware configurations, preprocessing pipelines, and training hyperparameters established in Section \ref{sec:methodology}. For the ablation studies, specific modifications were made to the pipeline to empirically justify the selection of each core component.

These experiments were conducted on a subset of 100 writers from the primary dataset, partitioned into 80 writers for training and validation, 20 writers for line-level testing, and a 10-writer subset for page-level tests, in completely disjoint manner.

\subsection{Architectural Benchmarking}
\label{subsec:architectural_benchmarking}

Our primary goal was to build a feature encoder that is highly accurate, lightweight, and fast. To this end, we compared several State-of-the-Art (SOTA) networks using the line-level verification task. Performance was evaluated on a balanced test set of 1,000 line pairs (500 positive and 500 negative samples), with metrics reported at the threshold where Precision equals Recall. We measured model size (Total Parameters) and computational cost (FLOPs for a standard 128 $\times$ 128 input patch) using the PyTorch \verb|fvcore| library. We also recorded the average time per training epoch and the real-world inference speed per line, measured precisely using CUDA timers over 100 runs. 

As shown in Table~\ref{tab:table1}, this comparison highlights two clear solutions. The pretrained FasterNet-T0 emerged as the best SOTA option, achieving the highest accuracy (90.00\%) and AUC (0.9676) while remaining highly efficient during inference (3.27 million parameters and 3.44 ms latency). Furthermore, it demonstrated superior training efficiency compared to heavier models like ResNet-18 or ViT-Tiny, converging to its peak performance in just 10 epochs with an average epoch time of 6.85 minutes. 

When designing for strictly limited computational resources, such as edge devices, our custom DPE-Net is the ideal choice. It maintains a strong 83.40\% accuracy while drastically shrinking the model size to just 0.177 million parameters and processing each line in only 0.70 ms. This lightweight architecture also translates to exceptionally fast training, requiring only 2.94 minutes per epoch. Because they distinctly address the dual mandates of absolute accuracy and extreme efficiency, we selected both FasterNet-T0 and DPE-Net as the foundational encoders for our entire framework.




\begin{table*}[t!]
\caption{Architectural Benchmarking on Writer Verification (100-Writer Cohort, Line-Level).}
\label{tab:table1}
\centering
\renewcommand{\arraystretch}{1.05}
\scriptsize
\setlength{\tabcolsep}{1.5pt}

\begin{tabular}{|p{1.8cm}|c|c|c|c|c|c|c|c|c|c|c|}
\hline

\textbf{Metric} &
ResNet-18 &
\makecell{MobileNetV4\\(Conv-S)} &

\textbf{FasterNet-T0} &
ViT-Tiny &
\makecell{EdgeNeXt\\-XXS} &

EffiViT-M0 &
\makecell{Single-Path CNN\\(MaxPool + Dense)} &
\makecell{Single-Path CNN\\(Strided + GAP)} &
\makecell{EdgeNeXt-XXS\\Single-Path} &
\makecell{FasterNet-T0\\Single-Path} &
\textbf{DPE-Net} \\

\hline

\makecell[l]{Paradigm} &
CNN & Mobile CNN & \textbf{Low-Lat CNN} & Transformer & Hybrid & Hybrid & CNN & CNN & Hybrid & Hybrid & \textbf{Custom CNN} \\
\hline

\makecell[l]{Total Parameters} &
11.43M & 3.14M & \textbf{3.27M} & 5.49M & 1.24M & 2.25M & 8.61M & 0.15M & 1.42M & 3.02M & \textbf{0.1775M} \\
\hline

\makecell[l]{FLOPs (MACs) \\ per patch} &
568.39M & 60.74M & \textbf{109.58M} & 349.85M & 64.74M & 79.13M & 164.23M & 39.49M & 104.24M & 148.66M & \textbf{58.49M} \\
\hline

\makecell[l]{Epoch Time} &
10.16m & 8.68m & \textbf{6.85m} & 25.62m & 10.85m & 15.49m & 14.78m & 2.86m & 19.77m & 9.31m & \textbf{2.94m} \\
\hline

\makecell[l]{Best / Total \\ Epochs} &
17/20 & 12/20 & \textbf{10/20} & 20/20 & 17/20 & 17/20 & 18/20 & 18/20 & 18/20 & 17/20 & \textbf{18/20} \\
\hline

\makecell[l]{Inference / Line} &
3.45ms & 4.13ms & \textbf{3.44ms} & 5.82ms & 6.06ms & 21.90ms & 1.94ms & 0.53ms & 10.50ms & 5.82ms & \textbf{0.70ms} \\
\hline

\makecell[l]{Threshold} &
0.422 & 0.383 & \textbf{0.408} & 0.346 & 0.362 & 0.483 & 0.301 & 0.240 & 0.343 & 0.224 & \textbf{0.258} \\
\hline

\makecell[l]{Accuracy} &
89.00\% & 86.60\% & \textbf{90.00\%} & 89.80\% & 89.10\% & 87.80\% & 83.10\% & 82.60\% & 88.50\% & 89.40\% & \textbf{83.40\%} \\
\hline

\makecell[l]{AUC} &
96.66\% & 93.84\% & \textbf{96.76\%} & 96.19\% & 95.85\% & 95.22\% & 90.46\% & 92.08\% & 95.42\% & 96.26\% & \textbf{92.50\%} \\
\hline

\makecell[l]{Precision} &
89.00\% & 86.60\% & \textbf{90.00\%} & 89.80\% & 89.18\% & 87.80\% & 83.03\% & 82.60\% & 88.58\% & 89.40\% & \textbf{83.40\%} \\
\hline

\makecell[l]{Recall} &
89.00\% & 86.60\% & \textbf{90.00\%} & 89.80\% & 89.00\% & 87.80\% & 83.20\% & 82.60\% & 88.40\% & 89.40\% & \textbf{83.40\%} \\
\hline

\makecell[l]{F1 Score} &
0.8900 & 0.8660 & \textbf{0.9000} & 0.8980 & 0.8909 & 0.8780 & 0.8312 & 0.8260 & 0.8849 & 0.8940 & \textbf{0.8340} \\
\hline

\end{tabular}
\end{table*}
















\subsection{Ablation Studies}
\label{subsec:ablation_studies}

To clearly identify and confirm the impact of each architectural and algorithmic choice, we performed a series of ablation studies. Unless otherwise specified, the "Proposed" baseline framework utilizes: DPE-Net (Mean Pooling), Full Preprocessing (CLAHE + Adaptive Masking), Scale-Normalized Detection with Adaptive Tolerance ($\tau=0.5$), and Hierarchical Aggregation. All line-level experiments utilize 1,000-pair evaluation protocol, while page-level tests utilize 300 balanced pairs (150 positive, 150 negative) derived from the 10-writer subset.

\subsubsection{Preprocessing and Signal Conditioning}
\label{subsubsec:preprocessing}

We evaluated the multi-stage signal conditioning pipeline against raw, unconditioned grayscale inputs to determine its impact on feature extraction.





\begin{table*}[t!]
\caption{Ablation on Line-Level Preprocessing}
\label{tab:table2}
\centering
\renewcommand{\arraystretch}{1.2}

\begin{tabular}{|l|l|l|l|}
\hline
\textbf{Configuration Variant} & \textbf{Line-Level Accuracy} & \textbf{AUC} & \textbf{$\Delta$ Accuracy} \\
\hline
\textbf{Proposed: Full Pipeline (CLAHE + Adaptive Masking)} & \textbf{83.40\%} & \textbf{0.9250} & \textbf{Baseline} \\
\hline
Raw Grayscale Inputs (No Preprocessing) & 81.50\% & 0.9015 & -1.90\% \\
\hline
\end{tabular}
\end{table*}




As demonstrated in Table~\ref{tab:table2}, bypassing preprocessing forces the model to process dataset-specific artifacts (e.g., shadows, scanner noise), causing a significant performance drop. The proposed pipeline standardizes stroke geometry and eliminates background noise, yielding a definitive \textbf{+1.90\%} accuracy increase.

\subsubsection{Model Architecture and Feature Pooling}
\label{subsubsec:architecture_pooling}

Table~\ref{tab:table3} details the ablation of the spatial feature extractor, comparing dual-path design, pooling strategies, and spatial flattening methods.



\begin{table*}[t!]
\caption{Ablation on Model Architecture and Feature Pooling (Line-Level)}
\label{tab:table3}
\centering
\renewcommand{\arraystretch}{1.2}
\footnotesize

\begin{tabularx}{\textwidth}{|X|c|c|c|c|c|}
\hline
\textbf{Configuration Variant} & \textbf{Total Params} & \textbf{Time / Epoch} & \textbf{Inf. / Line} & \textbf{Line-Level Acc.} & \textbf{$\Delta$ Accuracy (vs Proposed)} \\
\hline
\textbf{Proposed: DPE-Net (Mean Pooling)} & \textbf{0.1775 M} & \textbf{2.94 min} & \textbf{0.70 ms} & \textbf{83.40\%} & \textbf{Baseline} \\
\hline
Single-Path CNN (Strided Convs + GAP) & 0.1500 M & 2.86 min & 0.53 ms & 82.60\% & -0.80\% \\
\hline
Single-Path CNN (MaxPool + Dense Flattening) & 8.6131 M & 14.78 min & 1.94 ms & 83.10\% & -0.30\% \\
\hline
Proposed DPE-Net utilizing Max-Pooling & 0.1775 M & 2.96 min & 0.83 ms & 81.30\% & -2.10\% \\
\hline
\end{tabularx}

\end{table*}




Integrating Mean Pooling across the patch sequence proved vastly superior to Max Pooling (+2.10\%), confirming that writer identity is best captured mathematically as a continuous average of stylistic traits rather than isolated structural extremes. While the Single-Path CNN (strided + GAP) offered a slightly lighter footprint, adding the parallel dilated branch (DPE-Net) costs only $\sim$27k parameters while boosting accuracy by +0.80\%. Furthermore, DPE-Net demonstrated smoother convergence with fewer erratic spikes, reaching a deeper validation minimum of 0.0812 compared to the Single-Path CNN (strided + GAP) (see Figure 
~\ref{fig:dpenet_appendix} and ~\ref{fig:single_path_appendix}.)








\begin{figure}[h!]
\centering
\includegraphics[width=\columnwidth]{fig/DPE-NET.png}
\caption{Training Validation Loss Trajectory: Proposed DPE-Net (100-Writer Cohort).}
\label{fig:dpenet_appendix}
\end{figure}



\begin{figure}[h!]
\centering
\includegraphics[width=\columnwidth]{fig/Single Path.png}
\caption{Training Validation Loss Trajectory: Single-Path CNN (100-Writer Cohort).}
\label{fig:single_path_appendix}
\end{figure}










\subsubsection{Page-Level Line Segmentation Ablation}
\label{subsubsec:segmentation_ablation}

Full-page preprocessing latency is often the primary bottleneck in real-world deployment. We evaluated our Scale-Normalized Detection against alternative OCR-based segmentation strategies, tracking both processing time and downstream biometric accuracy.




\begin{table*}[t!]
\caption{Ablation on Page-Level Line Segmentation (Latency vs. Accuracy)}
\label{tab:table4}
\centering
\renewcommand{\arraystretch}{1.2}
\footnotesize

\begin{tabularx}{\textwidth}{|X|c|c|c|c|c|}
\hline
\textbf{Segmentation Strategy} & \textbf{Preprocessing Latency} & \textbf{Total Page Time} & \textbf{Page Acc.} & \textbf{Page AUC} & \textbf{$\Delta$ Accuracy} \\
\hline
\textbf{Proposed: Detect + Adaptive Tol ($\tau=0.5$)} & \textbf{0.48 s} & \textbf{0.51 s} & \textbf{96.00\%} & \textbf{0.9958} & \textbf{Baseline} \\
\hline
Detect + Adaptive Tol ($\tau=0.1$) & 0.59 s & 0.64 s & 94.67\% & 0.9934 & -1.33\% \\
\hline
Detect + Adaptive Tol ($\tau=0.9$) & 0.46 s & 0.48 s & 94.00\% & 0.9891 & -2.00\% \\
\hline
Baseline OCR (Full CRNN Text Recognition) & 3.05 s & 3.14 s & 93.33\% & 0.9913 & -2.67\% \\
\hline
Native OCR Paragraph Mode (Block Merging) & 2.71 s & 2.72 s & 73.33\% & 0.8042 & -22.67\% \\
\hline
\end{tabularx}

\end{table*}




As detailed in Table~\ref{tab:table4} (where latency is averaged per page on a single NVIDIA RTX 3050, and total time encompasses both segmentation and downstream DPE-Net inference), the results highlight three critical deployment insights. First, our pure-detection method avoids the severe latency penalty of OCR, operating over 6x faster than full CRNN semantic decoding (0.51 s vs. 3.14 s) while actually yielding a +2.67\% increase in downstream accuracy. This empirically proves that semantic decoding is an unnecessary computational burden for purely biometric tasks. Second, relying on native OCR paragraph grouping resulted in a catastrophic 22.67\% accuracy drop. This native mode aggressively merges distinct lines into massive vertical blocks, destroying the horizontal spatial consistency required by the patch encoder. Finally, optimizing the vertical grouping tolerance ($\tau$) was crucial to segmentation performance; a strict $\tau = 0.1$ caused over-segmentation, while a loose $\tau = 0.9$ caused under-segmentation. The proposed $\tau = 0.5$ successfully preserved perfect single-line continuity, ultimately maximizing page-level accuracy.

\subsubsection{Local-to-Global Feature Integration Ablation}
\label{subsubsec:feature_integration_ablation}

Traditional page-level document analysis typically relies on "flat" architectures that directly integrate local features into a global representation (Patch $\rightarrow$ Page). To evaluate the necessity of explicit line segmentation while generating page-level encodings, we benchmarked our proposed hierarchical integration (Patch $\rightarrow$ Line $\rightarrow$ Page) against three such flat baselines using 300 balanced (150 positive, 150 negative) page-level pairs. The quantitative results of this comparison are detailed in Table~\ref{tab:table5}.

\begin{table*}[htbp]
\caption{Ablation on Hierarchical Aggregation (Page-Level)}
\label{tab:table5}
\centering
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline
\textbf{Aggregation Strategy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Accuracy} & \textbf{$\Delta$ Accuracy} & \textbf{PR=RC Threshold} \\
\hline
\textbf{Proposed: Hierarchical (Patch $\rightarrow$ Line $\rightarrow$ Page)} & \textbf{96.00\%} & \textbf{96.00\%} & \textbf{96.00\%} & \textbf{96.00\%} & \textbf{-} & \textbf{0.138500} \\
\hline
Flat (Patch $\rightarrow$ Page): CNN + Self-Attention + GeM & 92.67\% & 92.67\% & 92.67\% & 92.67\% & -3.33\% & 0.157500 \\
\hline
Flat (Patch $\rightarrow$ Page): CNN + Transformer ([CLS] Token) & 85.33\% & 85.33\% & 85.33\% & 85.33\% & -10.67\% & 0.114000 \\
\hline
Flat (Patch $\rightarrow$ Page): CNN + NetVLAD & 53.99\% & 94.67\% & 68.76\% & 57.00\% & -39.00\% & 0.000500* \\
\hline
\end{tabular}%
}
\par\medskip
\raggedright
\small \textit{*Note: For the NetVLAD architecture, Precision and Recall did not converge at any discrete point; the closest mathematical approximation is reported.}
\end{table*}

The flat baselines bypass segmentation entirely, treating the document as an unordered "bag of patches" by sampling 32 ink-rich patches via a sliding window. Because these randomly sampled patches lack a spatial sequence—whereas continuous lines inherently preserve spatial knowledge and better capture the writer's stylistic flow—they struggle to separate structural noise from true handwriting. CNN + NetVLAD fundamentally failed (57.00\% accuracy) at clustering these patches, while a Transformer utilizing a \verb|[CLS]| token gathered context more effectively (85.33\%). The most robust flat baseline, CNN + Self-Attention + GeM Pooling (92.67\%), successfully emphasized salient strokes and processed full pages rapidly.

However, the proposed hierarchical architecture proved strictly superior across all metrics, peaking at 96.00\% accuracy. This performance gap highlights the fundamental flaw of flat sampling: a global "bag of patches" inevitably captures macro-level layout noise and inter-line spacing artifacts. By segmenting the document into distinct lines first, the hierarchical approach introduces a critical structural prior. It forces the network to evaluate contiguous text, systematically filtering out formatting variations to extract a pure biometric representation of the writer's penmanship.

\subsubsection{Unsupervised Page Clustering Ablation}
\label{subsubsec:clustering_ablation}

For the document clustering pipeline, the foundational DPE-Net (Mean Pooling) was fully trained on the primary 371-writer training and validation cohort. We evaluated clustering algorithms using disjoint writer sets: writers 372–403 were used for hyperparameter tuning, and writers 404–415 were isolated for final evaluation. The quantitative results of this comparison
are detailed in Table~\ref{tab:table6}



\begin{table*}[htbp]
\caption{Ablation on Unsupervised Page Clustering Algorithms}
\label{tab:table6}
\centering
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|}
\hline
\textbf{Algorithm} & \textbf{Hyperparameters} & \textbf{Tuning ACC} & \textbf{Tuning ARI} & \textbf{Eval Clusters} & \textbf{Eval ARI} & \textbf{Eval NMI} & \textbf{Eval F1-Score} & \textbf{Eval Accuracy} \\
\hline
Agglomerative & Distance Thresh: 0.110 & \textbf{99.21\%} & \textbf{0.9532} & 23 & \textbf{0.7630} & \textbf{0.8658} & \textbf{0.7875} & \textbf{95.60\%} \\
\hline
DBSCAN & eps: 0.060, min\_samples: 2 & 98.36\% & 0.9069 & 9 (+17 Noise) & 0.5847 & 0.7791 & 0.6415 & 90.86\% \\
\hline
\end{tabular}%
}
\end{table*}

While both algorithms performed exceptionally well on the tuning set, Agglomerative Clustering demonstrated superior scaling to the larger, unseen evaluation set, maintaining a high Adjusted Rand Index (0.7630) and overall Accuracy (95.60\%). DBSCAN, while highly effective at identifying dense core groups, naturally isolates more ambiguous or heavily degraded samples as unassigned "noise" (labeling 17 out of 138 evaluation pages as such). Because real-world document sorting typically requires every page to be definitively assigned to an authorial group, the distance-based hierarchical merging of Agglomerative Clustering proved to be the more appropriate and comprehensive choice for this specific global partitioning task.

\subsubsection{Sequential Multi-Writer Segmentation Ablation}
\label{subsubsec:sequential_ablation}

To evaluate the sequential multi-writer segmentation pipeline, we again utilized the fully trained DPE-Net. Hyperparameter tuning was conducted on 196 synthesized multi-writer pages (derived from writers 395–420), with final evaluation performed on 97 synthesized multi-writer pages (from writers 421–435), as shown in Table~\ref{tab:table7}.


\begin{table*}[t!]
\caption{Ablation on Sequential Multi-Writer Segmentation Pipelines}
\label{tab:table7}
\centering
\renewcommand{\arraystretch}{1.2}
\footnotesize

\begin{tabularx}{\textwidth}{|X|c|c|c|c|c|}
\hline
\textbf{Segmentation Algorithm} & \textbf{Optimal Distance / Eps} & \textbf{Optimal Window} & \textbf{Tuning SER (196 Pages)} & \textbf{Eval SER (97 Pages)} & \textbf{Absolute Sequence Acc.} \\
\hline
DBSCAN & \textbf{0.090} & \textbf{5} & \textbf{0.0931} & \textbf{0.0868} & \textbf{75.26\%} \\
\hline
Agglomerative & 0.220 & 7 & 0.0906 & 0.1916 & 72.16\% \\
\hline
\end{tabularx}

\end{table*}



These results highlight the distinct advantage of the proposed DBSCAN model (optimized at \verb|eps = 0.090| and \verb|min_samples = 3|) for localized chronological tracking. While Agglomerative Clustering slightly edged out DBSCAN on the tuning set (0.0906 vs. 0.0931 Sequence Error Rate), DBSCAN demonstrated better generalization on the unseen evaluation set, improving to a remarkable 0.0868 SER. Because Agglomerative Clustering forces every sequential embedding into a discrete group, it can occasionally misassign ambiguous boundary lines between paragraphs. Conversely, DBSCAN’s density-reachability metric uses this to its advantage by isolating those ambiguous transitional lines as "noise," thereby naturally producing highly contiguous and stable line-level blocks. This inherent stability makes DBSCAN the superior framework for absolute chronological reconstruction, perfectly mapping the correct writer transitions on 75.26\% of the test pages with a tighter temporal smoothing window.







\section{Final System Results}
\label{sec:final_results}

To evaluate the proposed framework, both FasterNet-T0 and DPE-Net were trained from scratch on the primary line-level dataset comprising 371 writers, partitioned into 297 for training and 74 for validation (an 80/20 writer disjoint split). The training dynamics, including convergence points and computational costs, are summarized in Table~\ref{tab:table8}. The complete training loss trajectory for DPE-Net is provided in Figure~\ref{fig:Complete_New_CNN_DPE_NET_epoch_Graph}.



\begin{figure}[t!]
\centering
\includegraphics[width=\columnwidth]{fig/Complete_New_CNN_DPE_NET_epoch_Graph.png}
\caption{Complete Training Loss and Convergence Trajectory for DPE-Net.}
\label{fig:Complete_New_CNN_DPE_NET_epoch_Graph}
\end{figure}



\begin{table*}[t!]
\caption{Training Dynamics and Convergence Metrics}
\label{tab:table8}
\centering
\renewcommand{\arraystretch}{1.2}
\footnotesize
\setlength{\tabcolsep}{5pt}

\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Architecture} & \textbf{Best Epoch / Total} & \textbf{Best Validation Loss} & \textbf{Avg. Epoch Time} & \textbf{Storage Size} \\
\hline
\textbf{DPE-Net} & 36 / 46 & 0.0416 & 17.06 min & 707 KB \\
\hline
\textbf{FasterNet-T0} & 36 / 46 & 0.0227 & 45.35 min & 12.6 MB \\
\hline
\end{tabular}

\end{table*}

\subsection{Line and Page-Level Evaluation Results}
\label{subsec:line_page_eval}

The final evaluation was conducted on a completely disjoint cohort of 32 unseen writers (IDs 372–403). Following the protocol established in Section \ref{subsec:evaluation_metrics}, all metrics were computed at the Break-Even Point (where Precision equals Recall). The dynamic threshold sweep diagrams for these evaluations are provided in Figure~\ref{fig:Complete_New_CNN_DPE_NET_line}, ~\ref{fig:Complete_fastnet_t0_line} ~\ref{fig:Complete_New_CNN_DPE_NET_page_level} ~\ref{fig:Complete_fastnet_t0_Page_Level}.




\begin{figure}[t!]
\centering
\includegraphics[width=\columnwidth]{fig/Complete_New_CNN_DPE_NET_line.png}
\caption{Dynamic Threshold Sweep and Break-Even Evaluation: DPE-Net (Line-Level).}
\label{fig:Complete_New_CNN_DPE_NET_line}
\end{figure}

\begin{figure}[t!]
\centering
\includegraphics[width=\columnwidth]{fig/Complete_fastnet_t0_line.png}
\caption{Dynamic Threshold Sweep and Break-Even Evaluation: FasterNet-T0 (Line-Level).}
\label{fig:Complete_fastnet_t0_line}
\end{figure}


\begin{figure}[t!]
\centering
\includegraphics[width=\columnwidth]{fig/Complete_New_CNN_DPE_NET_page_level.png}
\caption{Dynamic Threshold Sweep and Break-Even Evaluation: DPE-Net (Page-Level).}
\label{fig:Complete_New_CNN_DPE_NET_page_level}
\end{figure}

\begin{figure}[t!]
\centering
\includegraphics[width=\columnwidth]{fig/Complete_fastnet_t0_Page_Level.png}
\caption{Dynamic Threshold Sweep and Break-Even Evaluation: FasterNet-T0 (Page-Level).}
\label{fig:Complete_fastnet_t0_Page_Level}
\end{figure}



To rigorously test both local stroke extraction and global document aggregation, the evaluation was performed sequentially at both the line level (utilizing a strictly balanced set of 1,000 line pairs) and the page level (utilizing a balanced set of 400 full-document pairs). The systemic performance metrics for both stages, including the end-to-end processing latency for full pages, are summarized in Table~\ref{tab:table9}.

\begin{table*}[htbp]
\caption{Final Line and Page-Level Writer Verification Results.}
\label{tab:table9}
\centering
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|}
\hline
\textbf{Architecture} & \textbf{Evaluation Scope} & \textbf{Accuracy} & \textbf{AUC} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Eval Threshold} & \textbf{Avg Total Time / Page} \\
\hline
\textbf{DPE-Net} & Line-Level (1000 Pairs) & 90.10\% & 0.9669 & 0.9018 & 0.9000 & 0.9009 & 0.3148 & - \\
\hline
\textbf{DPE-Net} & Page-Level (400 Pairs) & 95.00\% & 0.9921 & 0.9500 & 0.9500 & 0.9500 & 0.1905 & 0.5163 s \\
\hline
\textbf{FasterNet-T0} & Line-Level (1000 Pairs) & 93.80\% & 0.9861 & 0.9380 & 0.9380 & 0.9380 & 0.3680 & - \\
\hline
\textbf{FasterNet-T0} & Page-Level (400 Pairs) & 96.00\% & 0.9949 & 0.9600 & 0.9600 & 0.9600 & 0.2525 & 0.5969 s \\
\hline
\end{tabular}%
}
\end{table*}

\subsection{Writer-Based Document Retrieval Evaluation Results}
\label{subsec:retrieval_eval}

The writer-based document retrieval evaluation (1:N search) was conducted on a cohort of 138 full document pages authored by 12 distinct writers (404–415 writers). Following the Leave-One-Out (LOO) ranking protocol established in Section \ref{subsec:document_retrieval}, each page was iteratively isolated and utilized as a query against the remaining search gallery. 

The retrieval performance was assessed by calculating the Cosine Distance between the global page embeddings to sort the ranked lists. The final retrieval metrics are summarized in Table~\ref{tab:table10}.

\begin{table}[htbp]
\caption{Writer-Based Document Retrieval Results.}
\label{tab:table10}
\centering
\resizebox{\columnwidth}{!}{%
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{|l|l|l|l|}
\hline
\textbf{Architecture} & \textbf{Top-1 Accuracy} & \textbf{Top-5 Accuracy} & \textbf{Mean Average Precision (mAP)} \\
\hline
\textbf{DPE-Net} & 94.20\% & 98.55\% & 91.19\% \\
\hline
\textbf{FasterNet-T0} & 100.00\% & 100.00\% & 96.60\% \\
\hline
\end{tabular}%
}
\end{table}

\subsection{Handwriting-Based Document Clustering Evaluation Results}
\label{subsec:clustering_eval}

To evaluate the unsupervised clustering capabilities of the framework, Agglomerative Hierarchical Clustering (with average linkage) was applied to the global page-level embeddings. As established in Section \ref{subsec:document_clustering}, the optimal distance threshold ($\tau_{cluster}$) for each architecture was determined via a parameter sweep on a disjoint tuning cohort of 239 pages from 32 writers (372–403 writers). This sweep identified the optimal stopping criteria as $\tau_{cluster} = 0.150$ for DPE-Net and $\tau_{cluster} = 0.110$ for FasterNet-T0.

The final clustering evaluation was executed on the isolated test cohort comprising 138 pages across 12 ground-truth writers (404–415). The results, encompassing both global partitioning metrics (ARI, NMI) and pairwise assignment metrics, are presented in Table~\ref{tab:table11}.



\begin{table*}[htbp]
\caption{Unsupervised Document Clustering Results.}
\label{tab:table11}
\centering
\renewcommand{\arraystretch}{1.15}
\scriptsize
\setlength{\tabcolsep}{2.2pt}

\begin{tabular}{|l|c|c|c|c|c|c|c|c|}
\hline

\textbf{Architecture} &
\textbf{\makecell{Distance\\Threshold ($\tau_{cluster}$)}} &
\textbf{\makecell{Clusters\\Formed}} &
\textbf{\makecell{Adjusted Rand\\Index (ARI)}} &
\textbf{\makecell{Normalized Mutual\\Info (NMI)}} &
\textbf{\makecell{Pairwise\\Accuracy}} &
\textbf{\makecell{Pairwise\\Precision}} &
\textbf{\makecell{Pairwise\\Recall}} &
\textbf{\makecell{Pairwise\\F1-Score}} \\

\hline

\textbf{DPE-Net} & 0.150 & 19 & 0.7865 & 0.9000 & 96.03\% & 0.7857 & 0.8328 & 0.8086 \\
\hline

\textbf{FasterNet-T0} & 0.110 & 22 & 0.9198 & 0.9445 & 98.64\% & 1.0000 & 0.8644 & 0.9272 \\
\hline

\end{tabular}
\end{table*}



FasterNet-T0 partitioned the evaluation corpus into 22 distinct clusters, resulting in a 0.9198 ARI and a 98.64\% pairwise accuracy. DPE-Net partitioned the dataset into 19 clusters, resulting in a 0.7865 ARI and a 96.03\% pairwise accuracy.

\subsection{Sequential Multi-Writer Segmentation Evaluation Results}
\label{subsec:segmentation_eval}

To evaluate the sequential multi-writer segmentation pipeline, DBSCAN clustering was applied to the chronological line embeddings following the protocol established in Section \ref{subsec:sequential_segmentation}. 

Hyperparameter tuning was conducted on a tuning cohort of 196 synthesized multi-writer pages to determine the optimal Maximum Neighborhood Distance ($\epsilon$), Minimum Samples ($\mu_{samples}$), and Smoothing Window Size ($w_{size}$) for each spatial encoder. The final evaluation was subsequently executed on a completely disjoint test set of 97 synthesized multi-writer pages. The sequence metrics, including the Sequence Error Rate (SER) and Absolute Sequence Accuracy, are presented in Table~\ref{tab:table12}.

\begin{table*}[htbp]
\caption{Sequential Multi-Writer Segmentation Results (196 Tuning Pages, 97 Evaluation Pages)}
\label{tab:table12}
\centering
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline
\textbf{Architecture} & \textbf{Optimal $\epsilon$} & \textbf{Minimum Samples} & \textbf{Window Size} & \textbf{Tuning SER} & \textbf{Eval SER} & \textbf{Absolute Sequence Accuracy} \\
\hline
\textbf{DPE-Net} & 0.09 & 3 & 5 & 0.0931 & 0.0868 & 75.26\% \\
\hline
\textbf{FasterNet-T0} & 0.10 & 2 & 5 & 0.0825 & 0.1168 & 79.38\% \\
\hline
\end{tabular}%
}
\end{table*}

FasterNet-T0 achieved an Absolute Sequence Accuracy of 79.38\% and a final evaluation SER of 0.1168. DPE-Net recorded an evaluation SER of 0.0868 and an Absolute Sequence Accuracy of 75.26\%.




\section{Discussion and Conclusion}
\label{sec:discussion_conclusion}

In this study, we introduced a unified biometric framework that significantly advances the scope of offline, document-level handwriting analysis, specifically addressing a major void in open-set Bengali handwriting research. A core technical achievement of this work is the departure from traditional "flat" spatial pooling. By engineering a geometrically aware \textit{Patch} $\rightarrow$ \textit{Line} $\rightarrow$ \textit{Page} hierarchical integration pipeline, we demonstrated that preserving the natural horizontal sequence of human penmanship is essential for isolating true biometric style from macro-level layout noise. Evaluated on a massively scaled, rigorously curated dataset of 435 writers, this framework establishes the first comprehensive open-set baselines for the Bangla script across verification, retrieval, and unsupervised clustering.

To resolve the diverse computational bottlenecks of large-scale digitization, we integrated and evaluated two distinct architectural solutions within our framework. We provide the state-of-the-art FasterNet-T0 as a formidable, high-accuracy solution, as it achieved near-perfect metrics across page-level verification (96.00\%) and retrieval (100.00\% Top-1 Accuracy). Concurrently, we offer our custom Dual-Path Patch Encoder (DPE-Net) as an exceptionally competitive, edge-optimized alternative. Despite operating with only a fraction of the parameters (0.1775 M vs. 3.27 M), DPE-Net maintained robust biometric rigor (95.00\% verification, 94.20\% Top-1 retrieval) while requiring significantly less training time ($\sim$13 hours vs. $\sim$34.5 hours). During inference, DPE-Net processed full pages in just 0.5163 seconds with a minimal 707 KB storage footprint, confirming its viability for real-time, large-scale document screening on resource-constrained devices.

Analyzing task-specific behaviors revealed key nuances in how these models map stylistic variation. In unsupervised document clustering, tuning the agglomerative thresholds to prioritize absolute cluster purity resulted in high sensitivity to intra-writer variance. Both models over-segmented the evaluation writers but maintained pairwise accuracies above 96\%. This indicates a conservative algorithmic bias that enforces strict morphological consistency; natural stylistic shifts (e.g., varying pens or physical writing conditions) are safely isolated into distinct identities. Consequently, this strict partitioning maximizes True Negative rates, effectively safeguarding against the erroneous merging of different authors.

Furthermore, expanding the analysis to sequential multi-writer segmentation revealed a distinct performance trade-off in chronological reconstruction, allowing practitioners to choose a model based on their specific priorities. FasterNet-T0 achieved a higher Absolute Sequence Accuracy (79.38\% vs. 75.26\%), making it ideal for flawless sequence recovery, but it recorded a higher Sequence Error Rate (0.1168 vs. 0.0868) than DPE-Net. This metric inversion indicates that while FasterNet-T0 perfectly reconstructs a greater volume of pages, its occasional misclassifications yield severe edit distances, whereas DPE-Net provides a highly stable, tightly bounded error margin across complex layouts. Ultimately, by formalizing a reproducible pipeline to detect and map chronological author transitions on a single continuous page, this framework breaks entirely new ground for handwriting biometrics in any language, paving the way for the automated analysis of highly complex, multi-author historical and forensic documents.







\clearpage


\appendices




\section*{Acknowledgment}




\begin{thebibliography}{00}

\bibitem{b1} G. Piccinini, L. Guarnera, and S. Battiato, ``SignForensics: A robust framework for forensic offline signature verification with enhanced detection, noise removal, and multi-stage authentication,'' \emph{IEEE Access}, vol. 13, pp. 172456--172470, 2025, doi: 10.1109/ACCESS.2025.3617221.

\bibitem{b2} M. Ohi, J. Shin, M. Kabir, M. M. Monowar, and M. A. Hamid, ``A thresholded Gabor-CNN based writer identification system for Indic scripts,'' \emph{IEEE Access}, 2021, doi: 10.1109/ACCESS.2021.3114799.

\bibitem{b3} C. Adak, B. B. Chaudhuri, and M. Blumenstein, ``An empirical study on writer identification and verification from intra-variable individual handwriting,'' \emph{IEEE Access}, vol. 7, pp. 24738--24758, 2019, doi: 10.1109/ACCESS.2019.2899908.

\bibitem{b4} H. Gomez Adorno \emph{et al.}, ``Hierarchical clustering analysis: The best-performing approach at PAN 2017 author clustering task,'' in \emph{Proc. CLEF}, Avignon, France, 2018, pp. 216--223, doi: 10.1007/978-3-319-98932-7\_20.

\bibitem{b5} R. Tolosana, R. Vera-Rodriguez, J. Fierrez, and J. Ortega-Garcia, ``DeepSign: Deep on-line signature verification,'' \emph{IEEE Trans. Biometrics, Behavior, Identity Sci.}, 2021, doi: 10.1109/TBIOM.2021.3054533.

\bibitem{b6} N. Tahir \emph{et al.}, ``Offline handwritten signature verification system using artificial neural networks,'' \emph{Int. J. Intell. Syst. Appl.}, vol. 13, 2021, doi: 10.5815/ijisa.2021.01.04.

\bibitem{b7} H. Zhao and H. Li, ``Handwriting identification and verification using artificial intelligence-assisted textural features,'' \emph{Sci. Rep.}, vol. 13, 2023, doi: 10.1038/s41598-023-48789-9.

\bibitem{b8} M. Vamsikrishna \emph{et al.}, ``Investigating writer-independent deep learning techniques for offline handwritten signature verification,'' in \emph{Proc. ICAC2N}, 2024, pp. 1658--1663, doi: 10.1109/ICAC2N63387.2024.10895553.

\bibitem{b9} M. Javidi and M. Jampour, ``A deep learning framework for text-independent writer identification,'' \emph{Eng. Appl. Artif. Intell.}, vol. 95, Art. no. 103912, 2020, doi: 10.1016/j.engappai.2020.103912.

\bibitem{b10} P. Zhang, ``RSTC: A new residual Swin transformer for offline word-level writer identification,'' \emph{IEEE Access}, vol. 10, 2022, doi: 10.1109/ACCESS.2022.3178597.

\bibitem{b11} M. Qasim \emph{et al.}, ``Writer identification by processing hyperspectral images using singular value decomposition and linear discriminant analysis,'' \emph{IEEE Access}, 2026, doi: 10.1109/ACCESS.2026.3667159.

\bibitem{b12} K. Ngin \emph{et al.}, ``KhmerWriterID: Toward robust Khmer writer verification using deep learning,'' \emph{IEEE Access}, 2026, doi: 10.1109/ACCESS.2026.3666649.

\bibitem{b13} M. L. Bouibed, H. Nemmour, and Y. Chibani, ``SVM-based writer retrieval system in handwritten document images,'' \emph{Multimedia Tools Appl.}, vol. 81, pp. 1--23, 2022, doi: 10.1007/s11042-020-10162-7.

\bibitem{b14} S. Rasoulzadeh and B. BabaAli, ``Writer identification and retrieval based on NetVLAD with re-ranking,'' \emph{IET Biometrics}, 2021, doi: 10.1049/bme2.12039.

\bibitem{b15} M. Koepf, F. Kleber, and R. Sablatnig, ``Writer identification and writer retrieval using vision transformer for forensic documents,'' in \emph{Proc. Int. Conf.}, 2022, pp. 352--366, doi: 10.1007/978-3-031-06555-2\_24.

\bibitem{b16} V. Christlein \emph{et al.}, ``Writer retrieval and writer identification in Greek papyri,'' in \emph{Proc. Int. Conf.}, 2022, pp. 76--89, doi: 10.1007/978-3-031-19745-1\_6.

\bibitem{b17} T. Raven, A. Matei, and G. A. Fink, ``Self-supervised vision transformers for writer retrieval,'' in \emph{Proc. ICDAR}, 2024, pp. 380--396.

\bibitem{b18} M. A. Rahman \emph{et al.}, ``BN-HTRd: A benchmark dataset for document-level offline Bangla handwritten text recognition (HTR),'' \emph{Mendeley Data}, vol. 4, 2023, doi: 10.17632/743k6dm543.4.

\bibitem{b19} C. Halder, S. M. Obaidullah, K. C. Santosh, and K. Roy, ``Content-independent writer identification on Bangla script: A document-level approach,'' \emph{Int. J. Pattern Recognit. Artif. Intell.}, vol. 32, no. 9, Art. no. 1856011, 2018, doi: 10.1142/S0218001418560116.


\bibitem{b20} J. Chen \emph{et al.}, ``Run, don't walk: Chasing higher FLOPS for faster neural networks,'' in \emph{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2023, pp. 12021--12031, doi: 10.1109/CVPR52729.2023.01157.

\bibitem{b21} K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image recognition,'' in \emph{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2016, pp. 770--778, doi: 10.1109/CVPR.2016.90.

\bibitem{b22} D. Qin \emph{et al.}, ``MobileNetV4: Universal models for the mobile ecosystem,'' in \emph{Proc. Eur. Conf. Comput. Vis. (ECCV)}, 2024, pp. 78--96.

\bibitem{b23} H. Touvron \emph{et al.}, ``Training data-efficient image transformers \& distillation through attention,'' in \emph{Proc. Int. Conf. Mach. Learn. (ICML)}, 2020.

\bibitem{b24} M. Maaz \emph{et al.}, ``EdgeNeXt: Efficiently amalgamated CNN-transformer architecture for mobile vision applications,'' in \emph{Proc. ECCV Workshops}, 2022, pp. 3--20.

\bibitem{b25} X. Liu \emph{et al.}, ``EfficientViT: Memory efficient vision transformer with cascaded group attention,'' in \emph{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2023, pp. 14420--14430, doi: 10.1109/CVPR52729.2023.01386.

\bibitem{b26} J. Bromley \emph{et al.}, ``Signature verification using a Siamese time delay neural network,'' \emph{Int. J. Pattern Recognit. Artif. Intell.}, vol. 7, no. 4, p. 25, 1993, doi: 10.1142/S0218001493000339.

\bibitem{b27} F. Schroff, D. Kalenichenko, and J. Philbin, ``FaceNet: A unified embedding for face recognition and clustering,'' in \emph{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2015, pp. 815--823, doi: 10.1109/CVPR.2015.7298682.




\end{thebibliography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{fig/a1.jpeg}}]{First A. Author} (M'76\-\-SM'81\-\-F'87) and all authors may include 
biographies. Biographies are often not included in conference-related
papers. This author became a Member (M) of IEEE in 1976, a Senior
Member (SM) in 1981, and a Fellow (F) in 1987. The first paragraph may
contain a place and/or date of birth (list place, then date). Next,
the author's educational background is listed. The degrees should be
listed with type of degree in what field, which institution, city,
state, and country, and year the degree was earned. The author's major
field of study should be lower-cased. 

The second paragraph uses the pronoun of the person (he or she) and not the 
author's last name. It lists military and work experience, including summer 
and fellowship jobs. Job titles are capitalized. The current job must have a 
location; previous positions may be listed 
without one. Information concerning previous publications may be included. 
Try not to list more than three books or published articles. The format for 
listing publishers of a book within the biography is: title of book 
(publisher name, year) similar to a reference. Current and previous research 
interests end the paragraph. The third paragraph begins with the author's 
title and last name (e.g., Dr.\ Smith, Prof.\ Jones, Mr.\ Kajor, Ms.\ Hunter). 
List any memberships in professional societies other than the IEEE. Finally, 
list any awards and work for IEEE committees and publications. If a 
photograph is provided, it should be of good quality, and 
professional-looking. Following are two examples of an author's biography.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{a2.png}}]{Second B. Author} was born in Greenwich Village, New York, NY, USA in 
1977. He received the B.S. and M.S. degrees in aerospace engineering from 
the University of Virginia, Charlottesville, in 2001 and the Ph.D. degree in 
mechanical engineering from Drexel University, Philadelphia, PA, in 2008.

From 2001 to 2004, he was a Research Assistant with the Princeton Plasma 
Physics Laboratory. Since 2009, he has been an Assistant Professor with the 
Mechanical Engineering Department, Texas A{\&}M University, College Station. 
He is the author of three books, more than 150 articles, and more than 70 
inventions. His research interests include high-pressure and high-density 
nonthermal plasma discharge processes and applications, microscale plasma 
discharges, discharges in liquids, spectroscopic diagnostics, plasma 
propulsion, and innovation plasma applications. He is an Associate Editor of 
the journal \emph{Earth, Moon, Planets}, and holds two patents. 

Dr. Author was a recipient of the International Association of Geomagnetism 
and Aeronomy Young Scientist Award for Excellence in 2008, and the IEEE 
Electromagnetic Compatibility Society Best Symposium Paper Award in 2011. 
\end{IEEEbiography}



\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{fig/a3.jpg}}]{M. SHAHIDUR RAHMAN} (Senior Member,
IEEE) was born in Jamalpur, Bangladesh, in 1975.
He received the B.Sc. and M.Sc. degrees in electronics and computer science from the Shahjalal
University of Science and Technology, Sylhet,
Bangladesh, in 1995 and 1997, respectively, and
the Ph.D. degree in mathematical information systems from Saitama University, Saitama, Japan,
in 2006. He began teaching with the Department
of Computer Science and Engineering, Shahjalal
University of Science and Technology, in 1997, where he is currently a
Professor. He was a JSPS Postdoctoral Research Fellow at Saitama University, from 2009 to 2011. His research interests include voice analysis,
speech synthesis, speech recognition, bone-conducted speech augmentation,
and digital signal processing.



\end{IEEEbiography}

\EOD

\end{document}

# Domain: Any Language
## Subdomain: First mention of handwriting verification system (1980-2000) 

<br>



<br>


| Paper title | Year | Is training and testing set contain same writers (no if they are of different writers) | Topic relevance | DOI | Character /word /line /page level | Does it mention hand written page comparison pipeline | Identification/verfication accuracies |
|-------------|------|-------------------------------------------------------------------------------------------|-----------------|-----|--------------------------------|-----------------------------------------------------|---------------------------------------|
| 1.1 Pen pressure as an identifying characteristic of signatures: verification from the computer | 1998 | Yes | It directly explores handwriting/signature verification through computational analysis — specifically using pen pressure patterns as a biometric feature for access control and verification. | 10.69525/jasqde.7 | Signature level | No | ~99% success; Type I error 1–6%, Type II error <3% |
| 1.3 An on-line signature verification system using multi-template matching approaches | 1999 | Yes | It directly explores online handwriting/signature verification using multi-template matching approaches — specifically addressing the challenge of achieving accurate verification with only a few training samples, demonstrating one of the early computational methods for biometric access control. | 10.1109/CCST.1999.797957 | Signature level | No | FAR ~1%, FRR ~7.2%; larger test FAR ~11.6% |
| 1.4 Constructing a high performance signature verification system using a GA method | 1995 | Yes | It introduces GA-based selection of hard-to-imitate dynamic features (pen-up ‘virtual strokes’) for online signature verification—foundational to choosing writer-specific cues in modern deep metric learning for verification/retrieval. | 10.1109/ANNES.1995.499465 | Signature level | No | Type I ~6%, Type II ~0.8%; traced forgery Type II ~10% |






<br>

___

<br>



# Domain: Any Language
## Subdomain: Somewhat modern aproach to Handwriting verification system (2010-2015) 

<br>



<br>

| Paper title | Year | Is training and testing set contain same writers (no if they are of different writers) | Topic relevance | DOI | Character /word /line /page level | Does it mention hand written page comparison pipeline | Identification/verfication accuracies |
|-------------|------|-------------------------------------------------------------------------------------------|-----------------|-----|--------------------------------|-----------------------------------------------------|---------------------------------------|
| 2.1 Text Independent Writer Identification for Bengali Script | 2010 | Yes | text-independent Bengali writer identification using script-aware segmentation (matra/line/character) and directional + gradient (HOG-like) features with SVM, giving strong baselines. It’s directly useful to your Bangla pipeline as a script-specific, non-deep benchmark and a recipe for orientation/curvature-focused features to compare against your deep metric-learning models. | 10.1109/ICPR.2010.494 | Character-level (allographs) | Yes (line → word → character segmentation) | Top-1 ~95%, Top-3 ~99% |
| 2.3 Off-Line Writer Verification Using Shape and Pen Pressure Information | 2012 | No | Offline writer verification that fuses shape (Weighted Direction-code Histogram, WDH) with pen-pressure cues recovered from infrared (IR) scans. It shows how to extract pressure-like information in offline settings. | 10.1109/ICFHR.2012.246 | Line-level and page-level | Yes | 97–98% verification accuracy |
| 2.5 Text and script independent writer identification | 2014 | Yes | Text- and script-independent writer identification using GLCM texture features (correlation, homogeneity) across Roman, Kannada, Devanagari with k-NN, showing cross-script robustness | 10.1109/IC3I.2014.7019776 | Page level | Yes | ~82–85% (single/mixed scripts) |


<br>

___

<br>



# Domain: Any Language
## Subdomain: State of the art appraoch to Handwriting verification system (2017-2025)

<br>



<br>

| Paper title | Year | Is training and testing set contain same writers (no if they are of different writers) | Topic relevance | DOI | Character /word /line /page level | Does it mention hand written page comparison pipeline | Identification/verfication accuracies |
|-------------|------|-------------------------------------------------------------------------------------------|-----------------|-----|--------------------------------|-----------------------------------------------------|---------------------------------------|
| 3.1 Offline Text-Independent Writer Identification Based on Writer-Independent Model using Conditional AutoEncoder | 2018 | No | Text-independent writer identification via a writer-independent conditional autoencoder that conditions on character class to extract a character-agnostic “personal writing style” in latent space (ETL-1, NIST SD19). This directly informs your DML pipeline: learn content-invariant embeddings (multi-task/conditional cues), then compare embeddings for identification/verification. | 10.1109/ICFHR-2018.2018.00083 | Character-level | No | Top-25% ≈ 97% (ETL-1), ≈ 88% (NIST SD19) |
| 3.2 Length Independent Writer Identification Based on the Fusion of Deep and Hand-Crafted Descriptors | 2019 | Yes | Modern (2019) text/content/length-independent writer identification that fuses deep (CNN patch) + hand-crafted (LBP) features with VLAD encoding; shows middle-level fusion works best and proves strong cross-language robustness on IAM, CVL, Khatt. | 10.1109/ACCESS.2019.2927286 | Line level | No | Best on CVL/IAM with fusion; competitive on KHATT; very low average misclassifications (CVL ≈ 0.68/writer, IAM ≈ 0.49/writer, KHATT ≈ 0.01/writer). |
| 3.4 A Deep Learning Framework with Histogram Features for Online Writer Identification | 2020 | Yes | Online writer identification using sub-stroke–level histograms that fuse pressure + velocity (spatio-temporal cues), then a sequence LSTM autoencoder to compress descriptors and an SVM for classification (BIT CASIA). | 10.1109/ICFHR2020.2020.00032 | Line-level (online) | No | 81.75% (English), 80.97% (Chinese), 76.84% (Chinese–English) |
| 3.5 Writer identification using machine learning approaches: a comprehensive review | 2019 | Review paper | a comprehensive survey of writer identification (offline/online), covering datasets, features (hand-crafted & deep), classifiers, multi-script settings, and open challenges—perfect for background, taxonomy, and benchmarking your study. | 10.1007/s11042-018-6577-1 | Mixed | N/A | N/A |





<br>

___

<br>



# Domain: Bangla Specific
## Subdomain: State of the art approach for Bangla Handwriting verification system (2017-2025) 

<br>



<br>


| Paper title | Year | Is training and testing set contain same writers (no if they are of different writers) | Topic relevance | DOI | Character /word /line /page level | Does it mention hand written page comparison pipeline | Identification/verfication accuracies |
|-------------|------|-------------------------------------------------------------------------------------------|-----------------|-----|--------------------------------|-----------------------------------------------------|---------------------------------------|
| 4.2 Writer Identification of Bangla handwritings by Radon Transform Projection Profile | 2012 | Yes | This is offline, text-independent writer identification for Bangla using fragment-based features computed from Radon transform projection profiles, plus an open-set (“unknown writer”) rejection criterion—directly useful for your verification/ID survey and for feature baselines beyond gradients/HOG/LBP. | 10.1109/DAS.2012.98 | Page-level | Yes | Top-1 83.63% (90–110 words); Top-3 92.72%. For short text (20–30 words): Top-1 61.8%; Top-3 80%. |
| 4.3 Off-line Bangla signature verification: An empirical study | 2013 | Yes | Offline Bangla signature verification with handcrafted features (under-sampled bitmap, intersections/endpoints, directional chain code) and thresholded NN classifier—directly fits your signature verification survey (non-English, offline, handcrafted baselines + performance metric: AER 15.57%). | 10.1109/IJCNN.2013.6707123 | Signature-level (offline) | No | Best AER 15.75% (FAR=FRR=15.75%) with contour features; 20.16% (bitmap); 32.23% (intersection/end-points) |
| 4.4 Offline Bengali Writer Verification by PDF-CNN and Siamese Net | 2018 | Yes | text-independent Bengali writer verification using a hybrid pipeline: PDFs of handcrafted features → CNN (“PDF-CNN”) → Siamese network for same/different-writer decisions. Strong fit for your verification survey (Indic scripts, deep+handcrafted fusion, pairwise verification setup). | 10.1109/DAS.2018.33 | Page-level | Yes | 2.36% EER, 97.64% accuracy (Textural CNN-Siamese) |
| 4.5 Writer verification using feature selection based on geneticalgorithm: A case study on handwritten Bangla dataset | 2023 | Yes | Offline, text-independent Bengali writer verification using a hybrid pipeline: PDFs of handcrafted features → CNN (“PDF-CNN”) → Siamese network for same/different-writer decisions. Strong fitting for a verification survey (Indic scripts, deep+handcrafted fusion, pairwise verification setup). | 10.4218/etrij.2023-0188 | Page-level | Yes | Best 94.54% (SMO+GA, block); SMO no-GA: 89.76% (block), 81.02% (page); 5-fold CV 94.55%; AlexNet: 78.58% (block), 68.00% (page) |
| 4.6 Content Independent Writer Identification on Bangla Script: A Document Level Approach | 2018 | Yes | text-independent Bangla writer identification at the document level using local handwriting attributes with MLP and simple logistic classifiers; reports speedups vs. segmentation-based methods. Strong fit for a Bangla identification survey (behavioral biometrics, content-independent). | 10.1142/S0218001418560116 | Page level | Yes | Top-1 = 91.33%, Top-5 = 96.67%, EER = 6.67% |










<br>

___

<br>



# Domain: Signature verification
## Subdomain: State of the art approach for signature verification system {whether two signature are same or different or forged} {2018-2025}

<br>



<br>


| Paper title | Year | Is training and testing set contain same writers (no if they are of different writers) | Topic relevance | DOI | Character /word /line /page level | Does it mention hand written page comparison pipeline | Identification/verfication accuracies |
|-------------|------|-------------------------------------------------------------------------------------------|-----------------|-----|--------------------------------|-----------------------------------------------------|---------------------------------------|
| 5.1 SigVer - A Deep Learning Based Writer Independent Bangla Signature Verification System | 2021 | No | High. Writer-independent offline signature verification via Siamese CNN + contrastive loss. | 10.1007/978-981-16-1086-8_39 | Signature-level (offline) | No | 99.49% (2956 signatures, 57 writers) |
| 5.2 Inverse Discriminative Networks forHandwritten Signature Verification | 2019 | No | High. Multi-stream CNN with original/inverted branches + cross-attention for offline signature verification. | 10.1109/CVPR.2019.00591 | Signature-level (offline) | No | CEDAR EER 3.62% (FRR 2.17%, FAR 5.87%); BHsig-B Acc 95.32% (FRR 5.24%, FAR 4.12%); BHsig-H Acc 93.04% (FRR 4.93%, FAR 8.99%); Bengali 87.34% (FAR 19.89%, FRR 5.42%); Hindi 89.50% (FAR 12.01%, FRR 8.98%) |
| 5.3 SURDS: Self-Supervised Attention-guided Reconstruction and Dual Triplet Loss for Writer Independent Offline Signature Verification | 2022 | No | High. A writer-independent offline signature verification method that couples self-supervised encoder–decoder pretraining (attention-guided reconstruction) with metric learning (dual triplet loss). Directly targets genuine vs. forgery discrimination with modern representation learning—spot-on for your survey. | 10.48550/arXiv.2021.10138 | Signature-level (offline) | No |  |
| 5.4 Multi-scale CNN-CrossViT network for offline handwritten signature recognition and verification | 2025 | No | High. Offline signature verification/recognition using a multi-scale CNN front-end fused with a CrossViT (cross-attention Vision Transformer) back-end fits squarely under state-of-the-art verification. The setup is consistent with modern pipelines that learn multi-scale texture–stroke cues with CNNs and model long-range shape dependencies with transformer cross-attention, then make same/different (genuine vs. forgery) decisions. | 10.1007/s40747-025-02011-7 | Page-level (signature image-level) | Yes (pairwise Siamese setup) | Recognition: up to 100% (MCYT), 99.69% (CEDAR), Verification: 95.12% (Bengali), 92.33% (Hindi) |
| 5.5 Learning the micro deformations by max-pooling for offline signature verification | 2021 | No | High. Writer-independent OSV that learns subtle, local “micro-deformation” cues via a max-pooling scheme in a CNN and compares signatures in an embedding space. Strong fit for your “same/different/forged” methodology focus. | 10.1016/j.patcog.2021.108008 | Signature-level (offline) | No | GPDSsynthetic EER ≈ 2.8% (skilled), 1.5–2.0% (random); CEDAR EER 2.76%; UTSig EER 6.14%; BHsig260 EER 8.21% (Bengali), 9.01% (Hindi) |




<br>

___

<br>

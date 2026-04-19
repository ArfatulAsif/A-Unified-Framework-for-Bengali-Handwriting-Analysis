# 4. Architectural Benchmarking and Ablation Studies

To test our model components efficiently, we used a smaller subset of 100 writers from the entire dataset. This dedicated subset was used for all benchmarking and ablation experiments, partitioned into 80 writers for training and validation, 20 writers for line-level testing, and a 10-writer subset for page-level tests.

## 4.1 Architectural Benchmarking

Our primary goal was to build a feature encoder that is highly accurate, lightweight, and fast. To this end, we compared several State-of-the-Art (SOTA) networks using the line-level verification task. We measured model size (Total Parameters) and computational cost (FLOPs for a standard 128 × 128 input patch) using the PyTorch `fvcore` library. We also recorded the average time per training epoch and the real-world inference speed per line, measured precisely using CUDA timers over 100 runs. 

As shown in **Table I**, this comparison highlights two clear solutions. The pretrained FasterNet-T0 emerged as the best SOTA option, achieving the highest accuracy (90.00%) and AUC (0.9676) while remaining highly efficient during inference (3.27 million parameters and 3.44 ms latency). Furthermore, it demonstrated superior training efficiency compared to heavier models like ResNet-18 or ViT-Tiny, converging to its peak performance in just 10 epochs with an average epoch time of 6.85 minutes. 

When designing for strictly limited computational resources, such as edge devices, our custom DPE-Net is the ideal choice. It maintains a strong 83.40% accuracy while drastically shrinking the model size to just 0.177 million parameters and processing each line in only 0.70 ms. This lightweight architecture also translates to exceptionally fast training, requiring only 2.94 minutes per epoch. Because they distinctly address the dual mandates of absolute accuracy and extreme efficiency, we selected both FasterNet-T0 and DPE-Net as the foundational encoders for our entire framework.

**Table I: Architectural Benchmarking on Writer Verification (100-Writer Cohort, Line-Level)**

| Architecture | Paradigm | Total Params | FLOPs (MACs) per patch | Epoch Time | Best / Total Epochs | Inference / Line | Eval Threshold | Line Acc. | Line AUC | Line Prec. | Line Rec. | Line F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pretrained ResNet-18 | Classic CNN | 11.43 M | 568.39 M | 10.16 min | 17 / 20 | 3.45 ms | 0.422 | 89.00% | 96.66% | 89.00% | 89.00% | 0.8900 |
| Pretrained MobileNetV4 (Conv-S) | Mobile CNN | 3.14 M | 60.74 M | 8.68 min | 12 / 20 | 4.13 ms | 0.383 | 86.60% | 93.84% | 86.60% | 86.60% | 0.8660 |
| **Pretrained FasterNet-T0** | **Low-Latency CNN** | **3.27 M** | **109.58 M** | **6.85 min** | **10 / 20** | **3.44 ms** | **0.408** | **90.00%** | **96.76%** | **90.00%** | **90.00%** | **0.9000** |
| Pretrained ViT-Tiny | Pure Transformer | 5.49 M | 349.85 M | 25.62 min | 20 / 20 | 5.82 ms | 0.346 | 89.80% | 96.19% | 89.80% | 89.80% | 0.8980 |
| Pretrained EdgeNeXt-XXS | Hybrid (Edge) | 1.24 M | 64.74 M | 10.85 min | 17 / 20 | 6.06 ms | 0.362 | 89.10% | 95.85% | 89.18% | 89.00% | 0.8909 |
| Pretrained EfficientViT-M0 | Hybrid (Speed) | 2.25 M | 79.13 M | 15.49 min | 17 / 20 | 21.90 ms | 0.483 | 87.80% | 95.22% | 87.80% | 87.80% | 0.8780 |
| Single-Path CNN (MaxPool + Dense) | Custom CNN | 8.61 M | 164.23 M | 14.78 min | 18 / 20 | 1.94 ms | 0.301 | 83.10% | 90.46% | 83.03% | 83.20% | 0.8312 |
| Single-Path CNN (Strided + GAP) | Custom CNN | 0.15 M | 39.49 M | 2.86 min | 18 / 20 | 0.53 ms | 0.240 | 82.60% | 92.08% | 82.60% | 82.60% | 0.8260 |
| Pretrained EdgeNeXt-XXS + Single-Path | Hybrid | 1.42 M | 104.24 M | 19.77 min | 18 / 20 | 10.50 ms | 0.343 | 88.50% | 95.42% | 88.58% | 88.40% | 0.8849 |
| Pretrained FasterNet-T0 + Single-Path | Hybrid | 3.02 M | 148.66 M | 9.31 min | 17 / 20 | 5.82 ms | 0.224 | 89.40% | 96.26% | 89.40% | 89.40% | 0.8940 |
| **Proposed: DPE-Net** | **Custom CNN** | **0.1775 M** | **58.49 M** | **2.94 min** | **18 / 20** | **0.70 ms** | **0.258** | **83.40%** | **92.50%** | **83.40%** | **83.40%** | **0.8340** |

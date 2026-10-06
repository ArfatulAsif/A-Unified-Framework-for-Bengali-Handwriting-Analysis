## 1. Data

<br>

### Dataset Folder Structure
```text
./data/Train/<writer_id>/<doc_id>/*.jpg                --- line dataset
./data/Test/<writer_id>/<doc_id>/*.jpg                 --- line dataset
./data/Evaluate_For_Pages/<writer_id>/*.jpg            --- page dataset
./data/Test_For_Pages/<writer_id>/*.jpg                --- page dataset
./data/Multi_Writer/Tune_pages/page_writer_sequence*.jpg --- multi-writer page dataset
./data/Multi_Writer/Test_pages/page_writer_sequence*.jpg --- multi-writer page dataset
```
You can customize the dataset and use various supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff`.

The `data` folder in this repository contains only a few samples of each type to demonstrate the structure. The full processed dataset is available upon reasonable request.

---

### Data Sources
The entire dataset used was based on the following publicly available sources:

**BN-HTRd:**
M. A. Rahman et al., "BN-HTRd: A benchmark dataset for document-level offline Bangla handwritten text recognition (HTR)," *Mendeley Data*, vol. 4, 2023, doi: 10.17632/743k6dm543.4 .

**WBSUBNdb_text:**
C. Halder, S. M. Obaidullah, K. C. Santosh, and K. Roy, "Content-independent writer identification on Bangla script: A document-level approach," *Int. J. Pattern Recognit. Artif. Intell.*, vol. 32, no. 9, Art. no. 1856011, 2018, doi: 10.1142/S0218001418560116 .

The folders for the **line** and **page** datasets listed below contain only a few samples to demonstrate how the data was utilized. You can collect the full raw datasets from the cited sources above.

```text
./data/Train/<writer_id>/<doc_id>/*.jpg
./data/Test/<writer_id>/<doc_id>/*.jpg
./data/Evaluate_For_Pages/<writer_id>/*.jpg
./data/Test_For_Pages/<writer_id>/*.jpg
```

---

### Synthesized Multi-Writer Dataset
However, for the novel task of multi-writer segmentation, we have synthesized **293 multi-author pages** created specifically for sequential multi-writer segmentation. These contain 1–4 distinct sequential writer segments per page. These files are fully available at:

```text
./data/Multi_Writer/Tune_pages/page_writer_sequence*.jpg  --- multi-writer page dataset
./data/Multi_Writer/Test_pages/page_writer_sequence*.jpg  --- multi-writer page dataset
```

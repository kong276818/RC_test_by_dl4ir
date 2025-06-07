# 🔁 DL4IR Query Reformulator

A **PyTorch-based implementation** of a query reformulation framework for **deep learning-based information retrieval (DL4IR)**.  
This repository serves as a foundational component for an academic research project focused on **enhancing retrieval quality** through **query rewriting using neural representation models**.

---

### 📌 Purpose

This repository is **adapted and extended** to support a research paper currently in preparation.  
The main focus is on **query reformulation techniques** that leverage deep contextual embeddings to improve retrieval performance in **open-domain QA** or **document-level retrieval** tasks.

**Key goals include:**
- Implementing a **modular query rewriting pipeline** compatible with transformer-based encoders.
- Enabling **fine-grained token-level reformulation** using CLS-based signal extraction.
- Supporting **integration with downstream retrieval and ranking modules** (e.g., BM25, Dense Retriever).

---

### 🛠️ Features

- Query reformulation module with **attention-based transformer layers** (e.g., BERT, DistilBERT).
- Support for:
  - Encoding queries with **context awareness**
  - **Dynamic reformulation strategies** (e.g., appending, rewriting, masking)
- Easily integrable into retrieval pipelines (e.g., ColBERT, DPR, BM25).
- Scripted **training & evaluation** on MS MARCO / Natural Questions / custom datasets.

---

### 📂 Project Structure

| Path                        | Description                                      |
|----------------------------|--------------------------------------------------|
| `reformulator/`            | Core query reformulation module                 |
| ├── `encoder.py`           | Transformer encoder wrapper (e.g., BERT)        |
| ├── `reformulate.py`       | Query rewriting logic                           |
| └── `utils.py`             | Tokenization, padding, and attention masking    |
| `dataset/`                 | Dataset loader and preprocessing scripts        |
| `configs/`                 | Configuration files for model/training          |
| `experiments/`             | Jupyter notebooks and testing scripts           |
| `train.py`                 | Main training script                            |
| `README.md`                | Project documentation (this file)               |


---

### 🔧 Requirements

- Python 3.9+
- PyTorch ≥ 1.13
- Huggingface Transformers
- faiss-cpu / faiss-gpu *(optional, for retrieval evaluation)*

**Installation:**

```bash
pip install -r requirements.txt
📊 Example Usage
Training:

---
python train.py \
    --model_config configs/dual_cls.json \
    --dataset_path ./data/msmarco \
    --save_dir ./checkpoints/
Inference:
---
python

from reformulator.reformulate import QueryReformulator

qr = QueryReformulator(model_path="checkpoints/...")
reformulated = qr.rewrite("What are the effects of quantum tunneling?")
📚 Related Work
This project draws inspiration from prior works on query rewriting, including:

T5-based Query Rewriting (Nogueira et al., 2020)

COIL & ColBERT token-level retrieval

Dual-CLS token alignment frameworks for instruction-free QA retrieval (under review)

📌 Note
This repository is under active development and tailored to support experiments for a dual-token alignment framework currently under academic peer review.











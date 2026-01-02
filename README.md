# CSIH
## CSIH: Contrastive signed-intent hypergraph learning for gene-drug interaction sign prediction

### This repository contains the official PyTorch implementation of the CSIH model (based on the framework originally titled DiSign).CSIH is a robust graph neural network framework designed for Signed Bipartite Graphs. It is particularly effective for tasks like Gene-Drug interaction prediction, where it disentangles latent intents (e.g., biological pathways or mechanisms) behind interactions.


![image](https://github.com/wangxiaosong96/CSIH/blob/main/src/Figure%202.png)


## 📖 Introduction

Accurately predicting the sign of gene–drug interactions (Activation vs. Inhibition) is crucial for drug repurposing and mechanistic interpretation. **CSIH** addresses the limitations of existing methods (which often ignore edge signs or rely on simple balance theory) by introducing:

1.  **Signed Hypergraph Learning**: Models complex high-order correlations between genes and drugs.
2.  **Disentangled Intent Encoder**: Decomposes interactions into $K$ latent biological intents.
3.  **Contrastive Regularization**: Enhances representation quality at both node and graph levels, specifically designed for sparse datasets.


## 🛠️ Prerequisites

### To run this project, you need Python 3.8+ and the following libraries: 

### pip install torch torch-geometric pandas numpy scikit-learn networkx

Note: Ensure your torch-geometric version is compatible with your torch and CUDA version.




## 📊 Dataset

The model is evaluated on several benchmark signed gene-drug interaction datasets. The statistics of these datasets are summarized below:

| Dataset | Gene | Drug | Positive Edge | Negative Edge | Association |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DGIdb** | 3,019 | 11,187 | 9,074 | 19,008 | 28,082 |
| **DrugBank** | 11,284 | 425 | 40,926 | 39,998 | 80,924 |
| **LINCS L1000** | 3,769 | 187 | 9,876 | 10,734 | 20,610 |




## 🛠️ Prerequisites

* Python 3.8+
* PyTorch 1.12+
* PyTorch Geometric
* Pandas, NumPy, Scikit-learn, NetworkX

Install dependencies: pip install torch torch-geometric pandas numpy scikit-learn networkx


## 🚀 Usage

### 1. Basic Training
Run with the default sample dataset: python run_CSIH.py

### 2. Custom Training
To train on a specific benchmark (e.g., DrugBank), ensure your CSV is formatted correctly and run:

python run_CSIH.py --dataset_path ./data/DrugBank.csv --n_intents 16 --batch 128


## 🤝 Citation

If you use this code or datasets, please cite our paper:

@article{wang2026CSIH,
  title={CSIH: Contrastive signed-intent hypergraph learning for gene-drug interaction sign prediction},
  author={Wang, Xiaosong and Chen, Guojun and Lv, GuoHao and Xue, Wei and Yue, Zhenyu and Wang, Qingyong and Gu, Lichuan},
  journal={Manuscript},
  year={2026},
  institution={Anhui Agricultural University}
}


## 📧 Contact
For any questions, please contact: Xiaosong Wang: xiaosongwang@stu.ahau.edu.cn






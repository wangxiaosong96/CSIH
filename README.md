# CSIH
## CSIH: Contrastive signed-intent hypergraph learning for gene-drug interaction sign prediction

### This repository contains the official PyTorch implementation of the CSIH model (based on the framework originally titled DiSign).CSIH is a robust graph neural network framework designed for Signed Bipartite Graphs. It is particularly effective for tasks like Gene-Drug interaction prediction, where it disentangles latent intents (e.g., biological pathways or mechanisms) behind interactions.

### 📖 IntroductionTraditional methods often treat interactions as binary (exist/not exist) or rely on rigid sociological theories (like Balance Theory). CSIH overcomes these limitations by introducing:Global Disentangled Intent Encoder: Decomposes global graph embeddings into $K$ latent intents.Local Sign-Aware Subgraph Extractor: Captures fine-grained topological patterns using a novel Two-Anchor Distance Labeling mechanism.Contrastive Learning: Utilizes multi-granularity self-supervision to improve robustness, especially on sparse datasets.

![image](https://github.com/wangxiaosong96/CSIH/blob/main/src/Figure%202.png)


### 📂 Project StructureThe repository is organized as follows:

CSIH_Project/

├── CSIH.py             # Core Model Architecture (Global GNN + Local Attention)

├── DataHandler.py      # Data Loading, Subgraph Extraction, and ID Mapping

├── Params.py           # Hyperparameters and Configuration

├── run_CSIH.py         # Main Training and Evaluation Script

├── interactions.csv    # Dataset (Gene-Drug pairs)

### 🛠️ Prerequisites

### To run this project, you need Python 3.8+ and the following libraries:

Bash

pip install torch torch-geometric pandas numpy scikit-learn networkx

Note: Ensure your torch-geometric version is compatible with your torch and CUDA version.


# CSIH
## CSIH: Contrastive signed-intent hypergraph learning for gene-drug interaction sign prediction

### This repository contains the official PyTorch implementation of the CSIH model (based on the framework originally titled DiSign).CSIH is a robust graph neural network framework designed for Signed Bipartite Graphs. It is particularly effective for tasks like Gene-Drug interaction prediction, where it disentangles latent intents (e.g., biological pathways or mechanisms) behind interactions.


![image](https://github.com/wangxiaosong96/CSIH/blob/main/src/Figure%202.png)


## 📂 Project StructureThe repository is organized as follows:

CSIH_Project/

├── CSIH.py             # Core Model Architecture (Global GNN + Local Attention)

├── DataHandler.py      # Data Loading, Subgraph Extraction, and ID Mapping

├── Params.py           # Hyperparameters and Configuration

├── run_CSIH.py         # Main Training and Evaluation Script

├── interactions.csv    # Dataset (Gene-Drug pairs)

## 🛠️ Prerequisites

### To run this project, you need Python 3.8+ and the following libraries:

### pip install torch torch-geometric pandas numpy scikit-learn networkx

### Note: Ensure your torch-geometric version is compatible with your torch and CUDA version.

## 📊 Dataset
##The model is evaluated on several benchmark signed gene-drug interaction datasets. The statistics of these datasets are summarized below:

Dataset,Gene,Drug,Positive Edge,Negative Edge,Association
DGIdb,"3,019","11,187","9,074","19,008","28,082"
DrugBank,"11,284",425,"40,926","39,998","80,924"
LINCS L1000,"3,769",187,"9,876","10,734","20,610"



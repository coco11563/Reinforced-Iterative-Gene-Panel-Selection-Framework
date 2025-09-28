# Reinforced Iterative Gene Panel Selection Framework (RiGPS)

[![IEEE TCBB](https://img.shields.io/badge/IEEE%20TCBB-2025-blue)](https://ieeexplore.ieee.org/abstract/document/11164312)
[![PubMed](https://img.shields.io/badge/PubMed-40953430-green)](https://pubmed.ncbi.nlm.nih.gov/40953430/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTCBBIO.2025.3609721-orange)](https://doi.org/10.1109/TCBBIO.2025.3609721)

## 📖 Overview

This repository contains the implementation of **RiGPS (Reinforced Iterative Gene Panel Selection Framework)**, a novel approach for identifying informative genomic biomarkers in label-free single-cell RNA-seq data using reinforcement learning.

Our method addresses the challenges of traditional gene panel selection approaches by:
- Leveraging ensemble knowledge from existing gene selection algorithms
- Incorporating reinforcement learning for dynamic refinement
- Mitigating biases through stochastic adaptability
- Improving precision and efficiency in biomarker discovery

  
### Installation
#### Clone the repository
git clone https://github.com/[username]/Reinforced-Iterative-Gene-Panel-Selection-Framework.git
cd Reinforced-Iterative-Gene-Panel-Selection-Framework

#### Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

#### Install dependencies
pip install -r requirements.txt


### Demo

1. Place the dataset in `./data`. The dataset is a two-dimensional matrix where each row represents a cell, each column represents a gene, and the last column indicates the cell type. Save the dataset in `HDF` format.
2. Specify the dataset name (please keep it consistent with the dataset name in the `./data`) and hyperparameters.

```python
from RIGPS import RIGPS

# hyperparameters
params = {
    "LEARN_STEPS": 100,
    "EXPLORE_STEPS": 100,
    "LEARN_EPSILON": 0.999,
    "EXPLORE_EPSILON": 0.6,
    "N_STATES": 64,
    "N_ACTIONS": 2,
    "TARGET_REPLACE_ITER": 100,
    "MEMORY_CAPACITY": 800,
    "seed": 1,
    "filter_model": ["RandomForest", "SVM", "KBest"],
    "prior_model": ["mRMR", "KBest", "cellbrf"],
    "INJECTION_NUMBER": 800,
    "filter": True,
}

if __name__ == "__main__":
    # dataset name
    dataset='Dataset Name'

    # X is gene express matrix, y is cell label
    X, y = load(dataset)

    # Run RIGPS to get the optimal gene subset
    rigps = RIGPS(dataset, X, y, params)
    optimal_gene_set = rigps.run()
```


## 📑 Publication

**Knowledge-Guided Gene Panel Selection for Label-Free Single-Cell RNA-Seq Data: A Reinforcement Learning Perspective**

*Meng Xiao, Weiliang Zhang, Xiaohan Huang, Hengshu Zhu, Min Wu, Xiaoli Li, Yuanchun Zhou*

IEEE Transactions on Computational Biology and Bioinformatics, 2025

### Citation

If you find this work useful, please cite our paper:

```bibtex
@article{xiao2025knowledge,
  title={Knowledge-Guided Gene Panel Selection for Label-Free Single-Cell RNA-Seq Data: A Reinforcement Learning Perspective},
  author={Xiao, Meng and Zhang, Weiliang and Huang, Xiaohan and Zhu, Hengshu and Wu, Min and Li, Xiaoli and Zhou, Yuanchun},
  journal={IEEE Transactions on Computational Biology and Bioinformatics},
  year={2025},
  publisher={IEEE},
  doi={10.1109/TCBBIO.2025.3609721}
}```

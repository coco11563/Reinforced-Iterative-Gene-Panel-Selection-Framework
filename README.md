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

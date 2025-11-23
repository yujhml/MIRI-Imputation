# MIRI: Missing Data Imputation by Reducing Mutual Information



[![arXiv](https://img.shields.io/badge/arXiv-2505.11749-b31b1b.svg)](https://arxiv.org/abs/2505.11749) [![OpenReview](https://img.shields.io/badge/OpenReview-NeurIPS%202025-8c1b13.svg)](https://openreview.net/pdf/b51848c5f9ad724e3a2e5f89c68946ee78f8eff8.pdf)

Official implementation of **MIRI**.

MIRI imputes missing data by reducing mutual information between missing entries and imputed values, conditioned on observed data.

## Installation

```bash
git clone https://github.com/yujhml/MIRI-Imputation.git
cd MIRI-Imputation
conda create -n miri python=3.10 && conda activate miri
pip install -r requirements.txt
```

## Usage

Run the demos in `examples/`:
*   `demo_toy.ipynb`: Visualization on toy data.
*   `demo_UCI.ipynb`: UCI dataset experiments.
*   `demo_imgs.ipynb`: Image imputation.

## Citation

```bibtex
@inproceedings{yu2025missing,
  title={Missing Data Imputation by Reducing Mutual Information with Rectified Flows},
  author={Yu, Jiahao and Ying, Qizhen and Wang, Leyang and Jiang, Ziyue and Liu, Song},
  booktitle={Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
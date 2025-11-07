# 🧩 Tight Bounds for Jensen’s Gap with Applications to Variational Inference

Official repository for the paper:  
**Tight Bounds for Jensen’s Gap with Applications to Variational Inference**  
Accepted for presentation at **ACM CIKM 2025**  
📄 [arXiv:2502.03988](https://arxiv.org/abs/2502.03988)

---

### 🔗 Related Work

This research builds upon our previous project:  
➡️ [**Bounding Evidence and Estimating Log-Likelihood in VAE (AISTATS 2023)**](https://github.com/gmum/Bounding_Evidence_Estimating_LL)

---

## 📁 Repository Structure

| File | Description |
|------|--------------|
| `bounds.py` | Computes tight bounds using pre-generated samples from VAE models (5/10-IWAE). |
| `fig_2.py` | Reproduces **Figure 2** from the paper. |
| `fig_3-4.py` | Reproduces **Figures 3 and 4** from the paper. |
| `gamma_20000.csv`, `lognormal_20000.csv` | Sample datasets used in the **Figure 3** experiment. |

---

## ⚙️ Usage

### 1. Generate Samples and Compute Components

First, generate samples from VAE models (5/10-IWAE) and compute  
$R(x, z) = \frac{p(x|z)p(z)}{q(z|x)}$ — Equation (41) from [Struski et al., accepted for AISTATS 2023](https://arxiv.org/pdf/2206.09453).

```bash
# You can reuse the helper script from our previous repository:
bash calculate_components.sh  # available at:
# https://github.com/gmum/Bounding_Evidence_Estimating_LL/blob/main/calculate_components.sh
````

### 2. Compute the Bounds

```bash
python bounds.py
```

### 3. Reproduce Figures

```bash
# Figure 2
python fig_2.py

# Figures 3 and 4 (requires gamma_20000.csv and lognormal_20000.csv)
python fig_3-4.py
```

---

## 📚 Citation

If you use this code or refer to our results, please cite:

```bibtex
@inproceedings{mazur2025tight,
  title     = {Tight Bounds for Jensen’s Gap with Applications to Variational Inference},
  author    = {Mazur, Marcin and Dziarmaga, Tadeusz and Ko{\'s}cielniak, Piotr and Struski, {\L}ukasz},
  booktitle = {Proceedings of the 34rd ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```

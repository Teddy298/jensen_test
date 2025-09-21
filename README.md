# Tight Bounds for Jensen’s Gap with Applications to Variational Inference  

This repository contains the official code for the paper:  

**Tight Bounds for Jensen’s Gap with Applications to Variational Inference**  
Accepted for presentation at **CIKM 2025**.  

This work builds upon our previous project:  
[Bounding Evidence and Estimating Log-Likelihood in VAE (AISTATS 2023)](https://github.com/gmum/Bounding_Evidence_Estimating_LL).  

---

## 📂 Contents  

- `bounds.py` — script for calculating the proposed bounds.  
- `fig_2.py` — reproduces Figure 2 from the paper.  
- `fig_3-4.py` — reproduces Figures 3 and 4 from the paper.  
- `gamma_20000.csv`, `lognormal_20000.csv` — datasets for experiments in Figures 3 and 4.  

---

## 🚀 Usage  

```bash
# Calculate bounds
python bounds.py

# Reproduce Figure 2
python fig_2.py

# Reproduce Figures 3 and 4 (requires gamma_20000.csv and lognormal_20000.csv)
python fig_3-4.py

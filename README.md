# Surrogate Models for Agent-Based Models: Challenges, Methods, and Applications

[![Paper](https://img.shields.io/badge/Paper-JFSMA_2025-blue.svg)](https://hal.science/hal-05035806)

This repository contains the code, data, and explainability artifacts for the research paper **"Modèles de Substitution pour les Modèles à base d'Agents : Enjeux, Méthodes et Applications"** presented at the **JFSMA 2025** conference.

## 📖 Overview

Agent-Based Models (ABMs) are powerful for simulating complex systems, but they often suffer from high computational costs. This project explores the use of **Machine Learning Surrogate Models** to approximate the behavior of ABMs, drastically reducing simulation time.

Using the classic **Schelling Segregation Model** (implemented in the GAMA platform) as a case study, this research demonstrates:
1.  **Dual-Objective Approximation:** Predicting both a continuous outcome (Sparsity - via Regression) and a binary outcome (Convergence - via Classification).
2.  **Model Comparison:** Evaluating models from Scikit-Learn (Random Forest, SVM, etc.) against specialized surrogate models from the SMT library (Gaussian Processes, RBF).
3.  **Explainable AI (XAI):** Applying SHAP, Partial Dependence Plots (PDP), and Individual Conditional Expectation (ICE) to understand *how* input parameters (e.g., density, intolerance threshold) drive the emergent behaviors of the ABM, even when trained on sparse datasets.

## 📂 Project Structure

The repository is structured to support Data Generation, Modeling, and Explainability:

*   **Simulation & Data Generation**
    *   `gama_jfsma.py` / `gama_jfsma_simple.py`: Python scripts integrating with the GAMA platform via `gama-client` to run headless simulations.
    *   `database_jfsma.py`: Utilities for parsing and structuring simulation outputs.
    *   `doe_5_200/`: Contains the Design of Experiments (DoE) dataset. Includes CSV results of GAMA simulations sampled using Latin Hypercube Sampling (200 points x 5 repetitions).
*   **Machine Learning & Surrogate Modeling**
    *   `models_jfsma.py`: The core machine learning pipeline. Implements training and evaluation of various models using `scikit-learn` and `smt`.
*   **Explainability & Analysis**
    *   `Explainability_tools2.ipynb`: A Jupyter Notebook containing the code for generating SHAP values, PDPs, and other interpretability metrics.
    *   `shap_values_*.npy`: Cached numpy arrays containing pre-computed SHAP values for fast visualization. `0` prefix indicates classification (convergence), `1` indicates regression (sparsity).
    *   `*.png` / `*.jpg`: Plots and figures generated for the paper.

## 🛠️ Key Technologies

*   **[GAMA Platform](https://gama-platform.org/):** The Agent-Based Modeling environment.
*   **[gama-client](https://pypi.org/project/gama-client/):** Python bridge to execute and control GAMA simulations.
*   **[Scikit-Learn](https://scikit-learn.org/):** Standard machine learning models (Random Forest, SVM, kNN).
*   **[SMT (Surrogate Modeling Toolbox)](https://smt.readthedocs.io/):** Specialized surrogate models (Gaussian Processes, RBF, GENN).
*   **[SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/):** For model interpretability and feature importance.

## 🚀 Getting Started

*(Note: Ensure you have Python 3.8+ installed, and ideally the GAMA platform if you intend to run new simulations).*

1.  **Clone the repository**
2.  **Install dependencies:**
    *(A `requirements.txt` is not provided, but you will need standard data science libraries and the SMT toolbox).*
    ```bash
    pip install numpy pandas scikit-learn smt shap matplotlib jupyter gama-client
    ```
3.  **Explore the Explainability:**
    Open the `Explainability_tools2.ipynb` notebook to see how SHAP values are computed and visualized based on the provided DoE dataset.
4.  **Train Models:**
    Run `models_jfsma.py` to re-train the Scikit-Learn and SMT models on the data in `doe_5_200/`.

## 📝 Citation

If you use this code or data in your research, please cite the JFSMA 2025 paper:

```bibtex
@inproceedings{saves2025modeles,
  title={Mod{\`e}les de Substitution pour les Mod{\`e}les {\`a} base d'Agents : Enjeux, M{\'e}thodes et Applications},
  author={Saves, Paul and Verstaevel, Nicolas and Gaudou, Benoit},
  booktitle={Journ{\'e}es Francophones sur les Syst{\`e}mes Multi-Agents (JFSMA)},
  year={2025}
}
```

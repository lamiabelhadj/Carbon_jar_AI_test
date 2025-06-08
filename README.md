# Carbon_jar_Data_test
# Internship Technical Challenge — AI for Emissions & Compliance

This repository contains 10 Jupyter Notebook solutions to a full-stack AI technical challenge. The goal is to apply machine learning, NLP, reinforcement learning, and MLOps best practices to emission tracking, anomaly detection, compliance parsing, and reporting.

Each notebook is self-contained and located in the `Challenge_tasks/` directory.

---

## Folder Structure

```bash
Challenge_tasks/
├── report.html              # Auto-generated emissions report (from Task 9)
├── Task1.ipynb              # LSTM Autoencoder for anomaly detection
├── Task2.ipynb              # Advanced data imputation + GAN sketch
├── Task3.ipynb              # Feature engineering + SHAP validation
├── Task4.ipynb              # Modular LCA with uncertainty simulation
├── Task5.ipynb              # RL for decarbonization policy optimization
├── Task6.ipynb              # Explainability via SHAP (audit readiness)
├── Task7.ipynb              # Domain constraints: emission cap logic
├── Task8.ipynb              # NLP extraction from regulation (offline-safe)
├── Task9.ipynb              # Auto-reporting using Jinja2 + plots
├── Task10.ipynb             # Seldon Core deployment + monitoring
README.md                   # Project overview and task map
```

---

## Task Highlights

| Task        | Description |
|-------------|-------------|
| **Task 1**  | Design and train an LSTM Autoencoder (PyTorch) to detect anomalies in emissions data, including reconstruction loss logic. |
| **Task 2**  | Use `IterativeImputer` for multivariate imputation, validate with RMSE and plot distribution similarity, and sketch a GAN for tabular data. |
| **Task 3**  | Create rolling mean, intensity ratios, date features with pandas. Apply `SelectKBest` for selection and validate importance using SHAP. |
| **Task 4**  | Refactor a monolithic LCA model into 5 modules. Add Monte Carlo simulation to estimate uncertainty in carbon impact. |
| **Task 5**  | Build a custom `gym.Env` for emission policy actions. Train a PPO agent using `Stable-Baselines3` to learn cost-effective decarbonization. |
| **Task 6**  | Generate audit-ready explanations using SHAP for anomaly detection outputs. Includes force plot and summary plot handling. |
| **Task 7**  | Add a post-processing layer that either caps or flags ML predictions violating emission regulations. Compare it with other constraint enforcement strategies. |
| **Task 8**  | Parse compliance text using regex-based NER (offline-safe), extract entities like `ORG`, `DATE`, and `LIMIT`, and provide a pathway to use Hugging Face when online. |
| **Task 9**  | Build an auto-report generator using `Jinja2`, embedded `matplotlib` plots, and dynamic insertion of anomalies, stats, and legal references. |
| **Task 10** | Fix a Seldon Core K8s manifest for production, add resource limits and Prometheus annotations, and explain how to monitor data drift using statistical tests like KS or PSI. |

---

## How to Use This Repository

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/emissions-ai-challenge.git
   cd emissions-ai-challenge
   ```

2. Launch Jupyter or VS Code:
   ```bash
   jupyter lab  # or open with VS Code
   ```

3. Navigate to `Challenge_tasks/` and open any notebook (`Task1.ipynb`, `Task5.ipynb`, etc.)

4. Run all cells in order to explore the solution logic, explanations, visualizations, and comments.

---

## Optional Setup

If you're starting from scratch, install the needed Python packages with:

```bash
pip install -r requirements.txt
```

To generate the HTML report from Task 9:

```bash
python Challenge_tasks/Task9.ipynb
```

---

## Tools & Libraries Used

- PyTorch, scikit-learn, XGBoost
- pandas, matplotlib, seaborn
- SHAP, Jinja2, transformers (offline-safe)
- Gymnasium, Stable-Baselines3
- Seldon Core YAML + Prometheus
- Monte Carlo simulations, KS test, PSI

---

## 📬 Author

This challenge was solved as part of a technical internship evaluation.


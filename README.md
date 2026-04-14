# Predicting Clothing Purchase Behaviour Using Transaction and Product Data in Online Fashion Retail

This project builds machine learning models to predict whether a customer will purchase a specific fashion item using transactional, product, and customer data from the **H&M Personalized Fashion Recommendation Dataset** (Kaggle).  
A balanced dataset of **4 million records** was created using:

- 2 million real purchase transactions  
- 2 million realistic synthetic non‑purchase samples (to avoid data leakage)

Two models were implemented and compared:

- **XGBoost** — fast, tree‑based ensemble model  
- **TabNet** — deep learning model for tabular data with attention-based feature selection  

The project evaluates accuracy, ROC‑AUC, interpretability, customer‑segment performance, and business value impact.

A live demo is deployed on Hugging Face Spaces.

---

## Dataset

This project uses the **H&M Personalized Fashion Recommendation Dataset** from Kaggle:

Dataset link:  
https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

### Required files:
Download and place these three CSV files in your working directory:

- `transactions_train.csv`
- `articles.csv`
- `customers.csv`

These files are **not included in this repository** due to size and Kaggle licensing restrictions.

---

## Requirements

Install all required Python packages:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
pytorch-tabnet

Code

Or install them using pip:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost pytorch-tabnet

Code

---

## Models Used

### **1. XGBoost**
- Model: `XGBClassifier`
- Training time: < 60 seconds (CPU)
- Metrics:
  - Accuracy: **76.65%**
  - ROC‑AUC: **0.8455**

### **2. TabNet**
- Model: `TabNetClassifier`
- Training: 10 epochs on GPU
- Metrics:
  - Accuracy: **77.74%**
  - ROC‑AUC: **0.8584**

TabNet also provides **attention masks** for interpretability.

---

## Customer Segmentation

K‑means clustering (k=5) was applied using:

- Age  
- Club membership  
- Fashion‑news frequency  

Both models were evaluated across segments to check for bias.  
TabNet showed consistent performance across all groups.

---

## Business Value Impact

A business ROI model estimated that TabNet’s improvement over XGBoost could generate:

**€5.1 million additional annual profit**  
for a retailer of H&M’s scale.

---

## Demo

A live interactive demo is available on Hugging Face Spaces:

https://huggingface.co/spaces/AIINFASHION/hm-purchase-prediction

---

## How to Run

1. Download the dataset from Kaggle  
2. Place the three CSV files in your working directory  
3. Open the notebook:  
   `hm_purchase_prediction.ipynb`
4. Run all cells to reproduce preprocessing, training, evaluation, and visualizations  

---

## Documentation

This repository includes:

- **Project Report (PDF)**
- **Configuration Manual (PDF)**
- **Source Code Notebook**

---

## Author

**Priyanka Bellamkonda**  
MSc Artificial Intelligence  
National College of Ireland 

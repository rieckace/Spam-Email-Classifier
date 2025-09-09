# Spam-Email-Classifier
# 📧 Naive Bayes Email Spam Classifier (Streamlit App)

This project is a **Naive Bayes-based spam email classifier** built using Python, scikit-learn, and Streamlit.  
It classifies emails into **Spam** or **Ham** based on word frequency features from a bag-of-words dataset.

---

## 🚀 Features
- Trains a **Multinomial Naive Bayes model** on a bag-of-words dataset.
- Interactive **Streamlit web app**:
  - Paste an email → instantly get Spam/Ham prediction.
  - Displays model accuracy.
- Lightweight and fast (suitable for deployment).

---

## 📂 Dataset
The dataset is **not included in this repo** due to size constraints.  
Please download it from Kaggle:

👉 [Kaggle Dataset: Spam Email Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

Once downloaded:
1. Rename the file to `emails.csv`
2. Place it in the project root folder (same as `app.py`).

---

## 🛠️ Installation & Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier

pip install -r requirements.txt




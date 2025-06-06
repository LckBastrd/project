# Depression Detection from Reddit Posts using Machine Learning

This code is my Machine Learning course final project.  
The goal is to automatically detect signs of depression in Reddit user posts using a machine learning model.

---

## 📚 Dataset

- **Source:** [Reddit Depression Dataset on Kaggle](https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset)  
- **Size:** ~2.47 million posts  
- **Structure:**  
    - `title` + `body`: text of the post  
    - `upvotes`: number of upvotes  
    - `num_comments`: number of comments  
    - `label`: target variable (0 = non-depressed, 1 = depressed)

---

## 🧠 Project Overview

1️⃣ **Data preprocessing:**  
- Text cleaning (remove URLs, punctuation, convert to lowercase).  
- Combining `title` + `body` into a single text field.  
- TF-IDF vectorization (30,000 features).  
- Class imbalance handling (`class_weight='balanced'`).

2️⃣ **Exploratory Data Analysis (EDA):**  
- Class distribution  
- Correlations of features  
- Upvotes and comments analysis

3️⃣ **Model training:**  
- **Baseline model:** Logistic Regression + TF-IDF  
- **Improvement:** XGBoost tested, Logistic Regression performed better  
- **Hyperparameter tuning:** GridSearchCV on Logistic Regression  
- **Cross-validation:** 3-fold CV used in GridSearchCV

4️⃣ **Model saving and inference:**  
- Final Logistic Regression model saved as `logreg_best_model.joblib`  
- TF-IDF vectorizer saved as `tfidf_vectorizer.joblib`

5️⃣ **Streamlit app:**  
- A simple app where users can enter any text and get an instant prediction.

---

## 🚀 How to run the Streamlit app

1️⃣ **Start streamlit app:**
- streamlit run app.py

2️⃣ **Open your browser:**
- It will be started on localhost 

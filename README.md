# 🎓 Student Performance Prediction using Logistic Regression

## 📌 Project Title
**Student Performance Prediction using Logistic Regression**

## 📅 Week 2 – Mini Project (AI & ML)

This project aims to build a simple machine learning model that can predict whether a student will **pass or fail** based on how many hours they studied and their attendance percentage. The model is created using **Logistic Regression**, which is a supervised classification algorithm.

---

## 🎯 Objective

To understand and implement a basic machine learning classification algorithm — **Logistic Regression** — using a small dataset. The model should be able to:
- Learn from given data
- Predict pass/fail status of a student
- Evaluate its prediction accuracy

This is a beginner-level project focused on understanding core ML concepts like data handling, model training, testing, and prediction.

---

## 📊 Dataset Description

The dataset was created manually based on the example provided in the assignment PDF. It contains only three columns:

| Feature Name     | Description                        |
|------------------|------------------------------------|
| `Hours_Studied`  | Number of hours a student studied  |
| `Attendance`     | Student's attendance percentage    |
| `Pass_Fail`      | Target label: 1 = Pass, 0 = Fail   |

### 🔢 Sample Data (From Assignment)
| Hours_Studied | Attendance | Pass_Fail |
|---------------|------------|-----------|
| 5             | 85         | 1         |
| 2             | 60         | 0         |
| 4             | 75         | 1         |
| 1             | 50         | 0         |

This is a very small dataset and is only intended for learning and experimentation.

---

## 🛠️ Tools and Libraries Used

This project was implemented in **Google Colab**, using the following Python libraries:

| Library       | Purpose                               |
|---------------|----------------------------------------|
| `pandas`      | Create and handle the dataset          |
| `sklearn`     | Build, train, and evaluate the model   |

### Libraries:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## 📈 Machine Learning Workflow

### 1. **Dataset Creation**
The dataset is hardcoded using Python dictionaries and converted into a DataFrame using `pandas`.

### 2. **Data Splitting**
The dataset is divided into:
- **Features (`X`)**: `Hours_Studied` and `Attendance`
- **Label (`y`)**: `Pass_Fail`

The data is split into **training and testing sets** using `train_test_split()` with the `stratify=y` option to ensure balanced class distribution (important because we only have 4 samples).

### 3. **Model Training**
The `LogisticRegression()` model is trained using the training data (`X_train`, `y_train`).

### 4. **Model Testing**
The model makes predictions on the test data (`X_test`) and the accuracy is evaluated using `accuracy_score()`.

### 5. **Prediction on New Data**
The trained model is used to predict the result for a **new student** who studied 3 hours and had 70% attendance.

---

## 🧪 Results

### ✅ Model Accuracy
The model accurately predicts the test set (depending on random split).

### ✅ Example Prediction
```python
new_data = pd.DataFrame([[3, 70]], columns=['Hours_Studied', 'Attendance'])
result = model.predict(new_data)
print("Pass" if result[0] == 1 else "Fail")
```

**Output:**
```
Prediction for new student: Pass
```

---

## 📁 Project Files

| File Name                                   | Description                                |
|--------------------------------------------|--------------------------------------------|
| `student_performance_logistic_regression.ipynb` | Main notebook with complete working code   |
| `README.md`                                 | Project documentation                      |

---

## 🚀 How to Run This Project

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the `.ipynb` file
3. Run the cells step-by-step
4. Optionally modify the new data and test more predictions

---

## 📌 Important Notes

- This project is based on a **very small dataset** (4 rows). It is not meant for real-world usage but for **learning** how machine learning models work.
- The use of `stratify=y` in `train_test_split()` is important to prevent training set from containing only one class.
- Logistic Regression is a good starting point for understanding classification tasks in ML.

---

## 👤 Author

- **Name:** *Mohamed Shadhil*
- **Institution:** Crescent Institute of Science & Technology
- **Course:** AI & ML – Week 2 Project

---

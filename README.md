# 🏠 Home Price Prediction Using Gradient Descent

> 📅 Academic Year: 2024\
> 🧑‍🏫 Instructor: Assistant Professor Dr. Tinnaphop Dindam\
> 🏫 Institution: Panyapiwat Institute of Management\
> 💻 Course: Mathematics for AI 1

---

This repository presents a machine learning project designed to predict housing prices using a **Linear Regression model** optimized via **Gradient Descent**. It was developed as part of the coursework for **1321203 Mathematics for Artificial Intelligence 1** at **Panyapiwat Institute of Management**.

---

## 📌 Project Overview

Predicting the price of a house is a fundamental problem in the real estate domain. The objective of this project is to demonstrate how **Gradient Descent** can be used to minimize the loss function of a linear regression model and produce accurate predictions for housing prices.
The following features are considered:

* House Area *(sq.m.)*
* Number of Rooms
* Land Size *(sq.wa)*
* Price per sq.m.

---

## 🧩 Introduction

Buying a home is one of the most significant financial decisions a person can make. With numerous factors influencing house prices (e.g., size, location, economic conditions), it is often difficult to estimate an accurate value.
This project applies **supervised learning** and the **Gradient Descent** algorithm to solve this problem by building a predictive model that can learn from historical data.

---

## 📖 Related Concepts

### 🔹 Linear Regression

Models the relationship between the dependent variable *(house price)* and multiple independent variables:

```math
y = θ_0 + θ_1x_1 + θ_2x_2 + θ_3x_3 + θ_4x_4
```

### 🔹 Gradient Descent

An iterative optimization algorithm used to minimize the loss function by updating model parameters:

```math
θ_i^{new} = θ_i - α \cdot \frac{∂SSE}{∂θ_i}
```

### 🔹 Learning Rate (α)

Controls the step size during each iteration of Gradient Descent.

### 🔹 Loss Functions

* **MAE** – Mean Absolute Error
* **MSE** – Mean Squared Error
* **SSE** – Sum of Squared Errors

### 🔹 Normalization

Scaling the dataset to a common range (0–1) before training to improve convergence.

---

## ⚙️ Methodology

1. **Data Collection**
   Housing records with relevant features were collected.
2. **Data Preprocessing**
   Applied normalization to ensure all features are on a similar scale.
3. **Model Building**
   Constructed a linear regression model and implemented Gradient Descent as the optimization technique.
4. **Parameter Optimization**
   Updated model parameters for **800 epochs**.
5. **Denormalization**
   Converted the normalized outputs back to real-world values for interpretation and evaluation.

---

## 📊 Results

After 800 iterations, the final **denormalized** linear regression equation obtained was:

```math
y = -5,992,800 + 44,393.77x_1 - 367,345x_2 - 19,659x_3 + 222.61x_4
```

**Accuracy**:

* Mean error: **0.0658%**
* Real-world testing error: **< 9%**

These results demonstrate the effectiveness of Gradient Descent in improving model performance.

---

## 🧠 Conclusion

This project confirms that **Gradient Descent** is a powerful and reliable optimization method when applied to linear regression problems such as house price prediction.

**✅ Pros**

* High prediction accuracy
* Simple and interpretable model

**⚠️ Cons**

* Sensitive to learning rate selection
* Affected by outliers in the data

---

## 📚 References

* ResearchGate
* BorntoDev
* Medium.com
* Bangkok Asset: *บ้านมือสองที่ใช้สำหรับการเก็บข้อมูล* ([https://www.bangkokassets.com](https://www.bangkokassets.com))

---

## 👨‍💻 Authors

* **Kritkhachon Jirawongrungreung** (6752300216)
* **Phanuphong Onlamul** (6752300305)
* **Phuriphon Hemkul** (6752300313)

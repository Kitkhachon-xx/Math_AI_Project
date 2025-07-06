# 🏠 Home Price Prediction using Gradient Descent

This project presents a machine learning model that applies the Gradient Descent algorithm to predict house prices based on various independent variables. It is part of the coursework for **1321203 Mathematics for Artificial Intelligence 1**, at Panyapiwat Institute of Management.

## 📌 Project Overview

House price prediction is a significant problem in the field of real estate analytics. In this project, we explore how the **Gradient Descent** algorithm can be used to optimize the parameters in a **Linear Regression** model for accurate price forecasting. The factors considered include:

- House Area (sq.m.)
- Number of Rooms
- Land Size (sq.wa)
- Price per sq.m.

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Related Concepts](#related-concepts)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [References](#references)

---

## 🧩 Introduction

In the current era, purchasing a house is one of the most important investments. Given the complexity of factors affecting housing prices (e.g., area, location, economic conditions), we aim to build a model that can predict prices accurately. This project leverages **Gradient Descent** as an optimization algorithm in the context of **supervised learning**.

---

## 📖 Related Concepts

### 🔹 Linear Regression

We use linear regression to model the relationship between independent variables and house price.

\[
y = θ0 + θ1 + θ2 + θ3 + θ4
\]

### 🔹 Gradient Descent

An optimization technique to minimize the **loss function** by iteratively updating parameters.

### 🔹 Learning Rate (α)

Controls the size of steps taken toward the minimum of the loss function.

### 🔹 Loss Functions

- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **SSE (Sum of Squared Errors)**

### 🔹 Normalization

Preprocessing step to scale input features to the same range (0 to 1) before training.

---

## ⚙️ Methodology

### 1. Data Collection

Collected housing data including area, room count, land size, and price per sq.m.

### 2. Data Preprocessing

Applied **Normalization** to scale features.

### 3. Model Building

Constructed a linear regression model and used **Gradient Descent** to optimize parameters \( \theta_0 \) to \( \theta_4 \).

### 4. Parameter Tuning

Updated the parameters across 800 epochs with:

\[
\theta_i^{new} = \theta_i - \alpha \cdot \frac{\partial SSE}{\partial \theta_i}
\]

### 5. Denormalization

Reverted normalized predictions back to real-world scale using:

\[
y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \theta_4 x_4
\]

---

## 📊 Results

After 800 optimization rounds, the final denormalized model:

\[
y = -5,992,800 + 44,393.77x_1 - 367,345x_2 - 19,659x_3 + 222.61x_4
\]

### Accuracy

- Mean error: **0.0658%**
- Tested on multiple real-world cases with less than **9% error**.

---

## 🧠 Conclusion

The model demonstrates that Gradient Descent is an effective method for optimizing linear regression parameters in house price prediction. With proper feature selection and normalization, the model can generalize well to unseen data.

### ✅ Pros

- High accuracy
- Interpretable model

### ⚠️ Cons

- Requires careful tuning of learning rate
- Sensitive to outliers

---

## 📚 References

- ResearchGate
- BorntoDev
- Medium.com
- Bangkok Asset: [บ้านมือสองที่ใช้สำหรับการเก็บข้อมูล](https://www.bangkokassets.com)

---

## 👨‍💻 Authors

- Kritkhachon Jirawongrungreung (6752300216)
- Phanuphong Onlamul (6752300305)
- Phuriphon Hemkul (6752300313)

Course: Mathematics for AI 1  
Instructor: Asst. Prof. Tinnapop Dindam  
Institution: Panyapiwat Institute of Management  
Academic Year: 2024 (Semester 2.1)

---

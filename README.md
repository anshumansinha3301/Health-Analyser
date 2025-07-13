# Health-Analyser
# 🩺 Health Risk Prediction Platform

This project is a **Health Risk Screening Tool** built with **Streamlit**. It uses **Logistic Regression** to predict an individual’s general health risk based on vitals and lifestyle parameters such as BMI, blood pressure, cholesterol, sleep, and more.

> 🚀 Built with ❤️ by [Anshuman Sinha](mailto:anshumansinhadto@gmail.com)

---

## 🌐 Live Demo

🔗 **[Click here to try the live app](https://healthanalyser.streamlit.app/)**

---

## 📌 Features

- 📊 **Health Risk Prediction** using Logistic Regression
- 🧮 Real-time **BMI Calculation**
- 📉 Visualized **Feature Importance** with Seaborn
- 📄 **Generate PDF Health Report** with all user inputs and model prediction
- 🔍 Interactive expanders for **ideal health guidelines**
- 📱 Responsive, simple UI using **Streamlit**

---

## 🧠 Technologies Used

- `Python`
- `Streamlit`
- `Pandas`, `NumPy`
- `scikit-learn`
- `Seaborn`, `Matplotlib`
- `fpdf`
- `Warnings`, `BytesIO`, `datetime`

---

## 🧪 Model Details

A logistic regression model is trained on a synthetically generated dataset (500 samples) with features like:

- Age, Gender, Weight, Height, BMI
- Glucose, Blood Pressure (Systolic & Diastolic)
- Cholesterol, Heart Rate
- Physical Activity, Smoking, Alcohol Consumption, Sleep Hours

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/health-risk-predictor.git
cd health-risk-predictor

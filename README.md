# 🏦 Loan Approval Prediction Web App

## 📌 Overview
This project is a Machine Learning web application that predicts whether a bank loan will be **Approved** or **Rejected** based on an applicant's financial and personal details. 

The core prediction engine is powered by a **K-Nearest Neighbors (KNN)** classifier. The model was trained on historical loan data and is served through a user-friendly web interface built with **Flask**.

## ✨ Features
* **Machine Learning Model:** K-Nearest Neighbors (KNN) algorithm with data scaling (`StandardScaler`) for accurate predictions.
* **Interactive Web Interface:** A clean, responsive HTML/CSS form where users can input their financial details.
* **Real-Time Predictions:** Instant feedback showing "🎉 Loan Approved!" or "❌ Loan Rejected."
* **Cloud Ready:** Fully containerized with Docker, making it easy to deploy to platforms like Hugging Face Spaces.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Data Processing & ML:** `pandas`, `scikit-learn`, `joblib`
* **Web Framework:** `Flask`
* **Frontend:** HTML, CSS
* **Deployment:** Docker

## 📂 Project Structure
* `train.py`: The script used to clean the dataset, train the KNN model, and export the `.joblib` files.
* `app.py`: The Flask web server that handles the frontend form and runs user data through the trained model.
* `requirements.txt`: List of Python dependencies.
* `Dockerfile`: Container instructions for deployment.
* `model.joblib`, `scaler.joblib`, `features.joblib`: The exported machine learning model and preprocessing tools.

## 📊 Dataset Features Used
The model makes its predictions based on the following applicant details:
* Number of Dependents
* Education Level (Graduate / Not Graduate)
* Employment Status (Self-Employed: Yes / No)
* Annual Income
* Loan Amount & Loan Term
* CIBIL Score
* Value of Assets (Residential, Commercial, Luxury, Bank)

---

## 🚀 How to Run Locally

### 1. Prerequisites
Make sure you have Python installed. Clone this repository and install the required libraries:
```bash
pip install -r requirements.txt

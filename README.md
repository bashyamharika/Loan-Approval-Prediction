# 🏦 LoanAI — Intelligent Loan Approval Prediction System

> **An end-to-end Machine Learning application that predicts loan approval eligibility from applicant, financial, credit, and asset information.**

**Live Demo:** [🚀 Try LoanAI Live]   :   https://loan-approval-prediction-1-xzkj.onrender.com 
*(Hosted on Render — the free deployment may temporarily spin down after periods of inactivity, so the first request may take a little longer to respond.)*

Source Code: https://github.com/bashyamharika/Loan-Approval-Prediction


 What is LoanAI?

LoanAI is a machine-learning-powered loan approval prediction system designed to demonstrate how applicant information can be transformed into an instant prediction through a complete ML-to-production workflow.

Instead of stopping at model training and accuracy metrics, this project connects the entire pipeline:

**Applicant Data → Data Processing → ML Model → Prediction API → Interactive React UI → Live Deployment**

The user enters relevant applicant information through a web interface, the frontend sends the information to a FastAPI backend, and the trained machine learning model returns:

* ✅ Loan Approval / Rejection
* 📊 Prediction Confidence

The project combines **Machine Learning + FastAPI + React + REST API + Deployment** into one working application.

🎯 Project Objective

Traditional ML projects often end after:

```text
Dataset
   ↓
Model Training
   ↓
Accuracy
```

LoanAI extends this workflow into an actual usable application:

```text
                ┌──────────────────────┐
                │     Applicant        │
                │      Details         │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    React Frontend    │
                │   Interactive Form   │
                └──────────┬───────────┘
                           │
                           │ JSON Request
                           ▼
                ┌──────────────────────┐
                │     FastAPI REST     │
                │         API          │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Trained ML Model   │
                │    Random Forest     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Prediction +         │
                │ Confidence Score     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   React Result UI    │
                └──────────────────────┘
```

---

# 🚀 Key Features

### 🤖 Machine Learning Prediction

The system uses a trained machine learning model to predict loan approval based on applicant information.

### 🌲 Random Forest Model

Multiple machine learning approaches were evaluated during model development, with **Random Forest** selected as the model used by the application.

### 📊 High Model Performance

The project reports:

| Metric       |            Result |
| ------------ | ----------------: |
| Model        |     Random Forest |
| Accuracy     |        **98.13%** |
| F1 Score     |        **98.49%** |
| Dataset Size | **4,269 records** |

> These metrics describe performance on the evaluation used during model development and should not be interpreted as a guarantee of real-world lending outcomes.

### ⚡ Instant Prediction

The trained model is loaded by the FastAPI backend and produces a prediction through an API request.

### 🌐 Interactive Web Application

The React frontend provides a clean interface where users can enter applicant information without interacting directly with Python or the ML model.

### 🔌 REST API Architecture

The frontend and machine learning model are separated using a FastAPI REST API.

### 📈 Prediction Confidence

The backend uses the model's probability output to return a confidence percentage along with the predicted status.

### 📱 Responsive UI

The frontend is designed as a modern web interface for interacting with the prediction system.

---

# 🧠 Machine Learning Workflow

The ML pipeline follows these major stages:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Preparation
     ↓
Categorical Feature Handling
     ↓
Train / Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Selection
     ↓
Model Serialization
     ↓
FastAPI Integration
     ↓
Real-Time Prediction
```

---

# 📋 Input Features

LoanAI uses applicant and financial information including:

### Personal Information

* Number of Dependents
* Education
* Self Employment Status
* CIBIL Score

### Financial Information

* Annual Income
* Loan Amount
* Loan Term

### Asset Information

* Residential Asset Value
* Commercial Asset Value
* Luxury Asset Value
* Bank Asset Value

These features are submitted from the React application to the backend as JSON.

---

# 🔄 How Prediction Works

When a user submits the form:

### Step 1 — User Input

The applicant enters their information through the React interface.

### Step 2 — Data Conversion

The frontend converts numerical fields into numeric values and prepares a JSON request.

Example structure:

```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 5000000,
  "loan_amount": 15000000,
  "loan_term": 20,
  "cibil_score": 750,
  "residential_assets_value": 8000000,
  "commercial_assets_value": 2000000,
  "luxury_assets_value": 5000000,
  "bank_asset_value": 4000000
}
```

### Step 3 — API Request

React sends the data to:

```text
POST /predict
```

### Step 4 — Backend Processing

FastAPI receives the request and creates a pandas DataFrame containing the applicant information.

### Step 5 — ML Prediction

The serialized Random Forest model processes the applicant data.

### Step 6 — Probability Calculation

The model generates class probabilities using:

```python
model.predict_proba()
```

### Step 7 — API Response

The backend returns:

```json
{
  "status": "Approved",
  "confidence": 97.0
}
```

### Step 8 — UI Result

The React frontend displays the prediction to the user.

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────┐
│                 USER                         │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│             REACT FRONTEND                   │
│                                              │
│  • Applicant Form                            │
│  • Input Validation                          │
│  • API Request                               │
│  • Prediction Result                         │
└─────────────────────┬────────────────────────┘
                      │
                 HTTP / JSON
                      │
                      ▼
┌──────────────────────────────────────────────┐
│               FASTAPI                        │
│                                              │
│  POST /predict                               │
│  CORS                                        │
│  Request Processing                           │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│          MACHINE LEARNING MODEL              │
│                                              │
│             Random Forest                   │
│                                              │
│  predict() + predict_proba()                 │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│              PREDICTION                     │
│                                              │
│       Approved / Rejected                    │
│       Confidence Percentage                  │
└──────────────────────────────────────────────┘
```

---

# 💡 What Makes This Project Different?

Many beginner loan prediction projects focus primarily on model training and displaying an accuracy score.

LoanAI focuses on the **complete application lifecycle**.

### Conventional ML Project

```text
Dataset
   ↓
Training
   ↓
Accuracy
   ↓
Done
```

### LoanAI

```text
Dataset
   ↓
Feature Preparation
   ↓
Model Comparison
   ↓
Model Selection
   ↓
Model Serialization
   ↓
FastAPI Backend
   ↓
REST API
   ↓
React Frontend
   ↓
User Input
   ↓
Real-Time Prediction
   ↓
Live Deployment
```

The main distinction is therefore not simply the choice of algorithm.

It is the integration of the **ML model into a usable software system**.

---

# 🔬 Why Random Forest?

Random Forest was selected as the model used for prediction after evaluating machine learning approaches during development.

Random Forest is particularly useful for this type of structured tabular data because it can capture nonlinear relationships between applicant characteristics and the target variable.

The final application therefore uses the trained Random Forest model stored as:

```text
loan_approval_model.pkl
```

---

# 🛠️ Technology Stack

## Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Backend

* FastAPI
* Uvicorn
* REST API
* CORS

## Frontend

* React
* Vite
* JavaScript
* CSS

## Deployment

* Render
* GitHub

---

# 📂 Project Structure

```text
Loan-Approval-Prediction/
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── loan_approval_dataset.csv
├── loan_approval_model.pkl
├── train_model.py
├── train_models.py
├── test_prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧪 Model Development

The project contains dedicated scripts for model development and testing.

### `train_model.py`

Used for the model training workflow.

### `train_models.py`

Used for training/evaluating the model approaches used during development.

### `test_prediction.py`

Used to test the trained model independently from the web application.

Example prediction output:

```text
==============================
     LOAN PREDICTION
==============================
Loan Status : APPROVED
Confidence  : 97.00%
==============================
```

---

# ⚙️ Local Installation

## 1. Clone the repository

```bash
git clone https://github.com/bashyamharika/Loan-Approval-Prediction.git
```

```bash
cd Loan-Approval-Prediction
```

---

# 🐍 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

---

# 📦 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 4. Start FastAPI Backend

From the project root:

```bash
python -m uvicorn backend.app:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

# ⚛️ 5. Start React Frontend

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

---

# 🔗 API Endpoint

## Health / Root Endpoint

```text
GET /
```

Response:

```json
{
  "message": "Loan Approval Prediction API is running"
}
```

## Prediction Endpoint

```text
POST /predict
```

### Request

```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 5000000,
  "loan_amount": 15000000,
  "loan_term": 20,
  "cibil_score": 750,
  "residential_assets_value": 8000000,
  "commercial_assets_value": 2000000,
  "luxury_assets_value": 5000000,
  "bank_asset_value": 4000000
}
```

### Response

```json
{
  "status": "Approved",
  "confidence": 97.0
}
```

---

# 🌍 Deployment

The application can be deployed as separate frontend and backend services.

The backend is suitable for deployment as a FastAPI web service, while the React application can be deployed as a static frontend.

Render supports FastAPI web services and static sites, and deployed web services receive a public `onrender.com` address.

### Live Application

🚀 **[Open LoanAI Live Demo](YOUR_RENDER_URL_HERE)**

> **Deployment note:** The live demo is hosted using Render's free infrastructure. Free web services can automatically spin down after 15 minutes without incoming traffic and may take some time to start again when the next request arrives. Therefore, availability of the demo may be temporary or subject to the hosting provider's free-tier limitations.

---

# 🔐 Important Disclaimer

LoanAI is an **educational machine-learning project** created to demonstrate an end-to-end ML application.

The prediction should **not** be treated as an actual banking or financial approval decision.

A real lending system would require additional factors such as:

* Regulatory compliance
* Applicant verification
* Credit history
* Debt-to-income analysis
* Fraud detection
* Financial institution policies
* Human/underwriter review
* Fairness and bias evaluation
* Production-grade security
* Privacy and data protection

The model's confidence score represents the model's prediction probability, **not a guarantee of loan approval**.

---

# 📊 Project Highlights

| Component     | Implementation            |
| ------------- | ------------------------- |
| Problem       | Loan approval prediction  |
| ML Type       | Supervised Classification |
| Model         | Random Forest             |
| Accuracy      | 98.13%                    |
| F1 Score      | 98.49%                    |
| Dataset       | 4,269 records             |
| Backend       | FastAPI                   |
| Frontend      | React + Vite              |
| Communication | REST API / JSON           |
| Model Format  | `.pkl`                    |
| Deployment    | Render                    |
| Repository    | GitHub                    |

---

# 🎓 Learning Outcomes

This project demonstrates practical experience with:

* Machine Learning classification
* Dataset handling
* Feature preparation
* Model evaluation
* Random Forest
* Probability-based prediction
* Model serialization
* Python backend development
* FastAPI
* REST API design
* CORS configuration
* React frontend development
* Frontend-backend integration
* JSON data exchange
* Git and GitHub
* Cloud deployment
* End-to-end ML application architecture

---

# 🔮 Future Improvements

Potential improvements include:

### 🧠 Explainable AI

Add SHAP or similar explainability techniques to show which applicant features contributed most to a prediction.

### 📊 Prediction Analytics

Add visual analytics for prediction distributions and applicant profiles.

### 🔐 Authentication

Introduce secure authentication and role-based access.

### 🗄️ Database Integration

Store prediction requests and results using a production database.

### 📈 Model Monitoring

Track model performance and data drift after deployment.

### ⚖️ Fairness Analysis

Evaluate model performance across relevant applicant groups and investigate potential bias.

### 🔄 Automated Retraining

Create a pipeline for periodically retraining the model using newly available data.

### 🐳 Containerization

Package the application using Docker for reproducible deployment.

---

# 👩‍💻 Author

**B. Harika**

B.Tech — Computer Science & Engineering
Data Science Specialization

GitHub:
https://github.com/bashyamharika

---

# ⭐ Support

If you found this project interesting, consider giving the repository a ⭐ on GitHub.

---

## 📌 Project Summary

**LoanAI transforms a traditional loan-prediction ML experiment into a complete deployable application by connecting a trained Random Forest model with a FastAPI prediction service and an interactive React frontend.**

```text
        DATA
         │
         ▼
   MACHINE LEARNING
         │
         ▼
    RANDOM FOREST
         │
         ▼
      FASTAPI
         │
         ▼
    REST API
         │
         ▼
      REACT
         │
         ▼
   LIVE APPLICATION
```

**From dataset → model → API → interface → deployment. 🚀**

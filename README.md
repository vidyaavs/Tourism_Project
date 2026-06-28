Tourism Package Prediction using MLOps

Project Overview

This project implements an end-to-end MLOps pipeline to predict whether a customer is likely to purchase a tourism package. The solution covers data registration, data preparation, model training, experiment tracking with MLflow, model deployment using Streamlit on Hugging Face Spaces, containerization with Docker, and CI/CD automation using GitHub Actions.

Project Workflow
Dataset Registration
        │
        ▼
Data Preparation
        │
        ▼
Model Training & Hyperparameter Tuning
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
Model Registration (Hugging Face)
        │
        ▼
Streamlit Deployment
        │
        ▼
Docker Containerization
        │
        ▼
GitHub Actions CI/CD

Technologies Used
| Category             | Technology                     |
| -------------------- | ------------------------------ |
| Programming Language | Python                         |
| Machine Learning     | XGBoost                        |
| Data Processing      | Pandas, Scikit-learn           |
| Experiment Tracking  | MLflow                         |
| Model Hub            | Hugging Face Model Hub         |
| Dataset Storage      | Hugging Face Dataset Hub       |
| Deployment           | Streamlit, Hugging Face Spaces |
| Containerization     | Docker                         |
| CI/CD                | GitHub Actions                 |

## Repository Structure

```text
Tourism_Project/
├── .github/
│   └── workflows/
│       └── pipeline.yml              # GitHub Actions workflow
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv               # Dataset
│   ├── deployment/
│   │   ├── app.py                    # Streamlit application
│   │   ├── Dockerfile                # Docker configuration
│   │   └── requirements.txt          # Deployment dependencies
│   ├── hosting/
│   │   └── hosting.py                # Hugging Face deployment script
│   ├── model_building/
│   │   ├── data_register.py          # Dataset registration
│   │   ├── prep.py                   # Data preparation
│   │   └── train.py                  # Model training and registration
│   ├── notebook/
│   │   └── Tourism_Project.ipynb     # Project notebook
│   ├── requirements.txt              # GitHub Actions dependencies
│   └── README.md                     # Project documentation
├── .dockerignore
├── .gitignore
└── README.md
```

Model Performance
Accuracy
Precision
Recall
F1-score
ROC-AUC

Deployment
Hugging Face Space

https://huggingface.co/spaces/vidyaa2026/Tourism_Package_Prediction

Hugging Face Model

https://huggingface.co/vidyaa2026/tourism-package-model

Hugging Face Dataset

https://huggingface.co/datasets/vidyaa2026/tourism_project

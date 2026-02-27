# ML Concept Drift Monitoring System

An end-to-end Machine Learning system for customer churn prediction with concept drift detection, automated retraining, and a monitoring dashboard.

This project demonstrates how real-world ML systems are designed beyond basic model training by incorporating monitoring, retraining logic, and a user-facing interface.

---

## Key Features

- Customer churn prediction using supervised machine learning  
- Concept drift detection using statistical hypothesis testing  
- Automated retraining when data distribution changes  
- Performance-based model replacement  
- Interactive dashboard built using Streamlit  

---

## System Workflow

Train initial model  
→ Generate predictions on new data  
→ Monitor feature distributions  
→ Detect concept drift using KS Test  
→ Retrain model if performance improves  
→ Update deployed model  

---

## Project Structure

ml-concept-drift-system  
├── app.py                 # Streamlit dashboard  
├── model.pkl              # Trained model artifact  
├── requirements.txt       # Dependencies  
├── README.md  
├── data  
│   └── churn.csv          # Dataset  
└── src  
    ├── train.py           # Training & retraining pipeline  
    ├── predict.py         # Prediction logic  
    └── drift_detection.py # Concept drift detection  

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- SciPy  
- Streamlit  

---

## Machine Learning Approach

- Model: Logistic Regression  
- Evaluation Metric: ROC-AUC  
- Drift Detection Method: Kolmogorov–Smirnov Test  
- Retraining Strategy:
  - Retrain only when drift is detected  
  - Replace model only if new performance is better  

---

## How to Run the Project

1. Install dependencies  
pip install -r requirements.txt  

2. Train the model  
py src/train.py  

3. Launch the dashboard  
streamlit run app.py  

---

## Dashboard Capabilities

- Displays churn prediction probabilities  
- Monitors concept drift in data  
- Allows retraining from the UI  
- Shows system and model status  

---

## Why Concept Drift Is Important

In real-world systems, customer behavior and data distributions change over time.  
This project ensures model reliability by continuously monitoring data shifts and retraining when necessary.

---

## Future Enhancements

- Real-time data ingestion  
- Drift alerting and logging  
- Cloud deployment  
- Extension to fraud detection or anomaly detection use cases  

---

## Author

Built as a practical Machine Learning engineering project focusing on monitoring, automation, and system design.

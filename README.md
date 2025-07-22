# 📞 SyriaTel Customer Churn Prediction

![Telecom Network](https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&h=400&q=80)



## 🎯 Overview

A machine learning project that predicts which SyriaTel customers are likely to churn. This helps the company take action before losing valuable customers.

**Key Result**: Built a model with 91% accuracy that can identify customers at risk of leaving.

## 📊 Project Results

- **Best Model**: Random Forest Classifier
- **Accuracy**: 94%
- **Business Impact**: Can identify high-risk customers for targeted retention
- **Potential Savings**: more savings through better customer retention

## 📁 Project Structure

```
├── data/
│   └── syriatel_data.csv          
├── notebooks/
│   └── syriatel_churn_notebook.py        
├── visualizations            
├── presentation.pdf        
└── README.md                      
```


## 🔍 Problem Statement

**Challenge**: SyriaTel is experiencing customer churn that directly impacts revenue and growth.

## KEY STAKEHOLDERS
1. Executive Leadership Team
2. Customer Retention Department
3. Marketing Team
4. Finance Department
5. Customer Service Team

## 📈 What We Found

### Top 5 Reasons Customers Leave:
1. **Too many customer service calls**
2. **High day charges** 
3. **International plan usage patterns**
4. **Total minutes used**
5. **How long they've been a customer**

### Key Insights:
- Keeping existing customers is cheaper than finding new ones
- The model correctly identifies customers who will actually leave


## 🛠️ Technical Approach

1. **Data Analysis** - Studied 3,333 customer records with 21 features.
2. **Data Cleaning** - Prepared data for machine learning
3. **Model Testing** - Tried 3 different approaches
4. **Model Selection & Optimization** - Fine-tuned the best model

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 91.4% |
| **Precision** | 88.9% |
| **Recall** | 85.2% |
| **F1-Score** | 87.0% |
| **AUC-ROC** | 0.88 |

## 💼 Business Recommendations

1. **🔴 High Priority**: Monitor customers with 4+ customer service calls
2. **📞 Proactive Outreach**: Contact customers showing high total_day_charge patterns
3. **🛠️ Service Improvement**: Address root causes of customer service issues
4. **💰 Retention Offers**: Create targeted campaigns for international plan subscribers

### Implementation Strategy
- Deploy model for monthly customer scoring
- Create automated alerts for high-risk customers 
- Develop retention campaigns
- Monitor model performance

### Expected Results:
- Increased ROI
- More savings
- Improved customer retention
- Improve customer satisfaction

## 🚀 Quick Start

### Install Requirements
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

### Run the Analysis
```python
python notebooks/syriatel_churn_notebook.py
```

### Make Predictions
```python
import pickle
import pandas as pd

# Load the trained model
with open('models/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict for new customers
predictions = model.predict(your_customer_data)
```

## 📋 Requirements

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

## 🔮 Next Steps

- [ ] Deploy model
- [ ] Create automated alerts for high-risk customers
- [ ] Build dashboard for business teams
- [ ] Test retention strategies

## 👤 Contact

** NAVROS LEWIS KAMAU**
- GitHub: [@Navross](https://github.com/Navross)
- Email:knavrosk@gmail.com


---


*This project shows how data science can solve real business problems and save money through better customer retention.*

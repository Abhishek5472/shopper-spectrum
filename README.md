# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendation Platform

An AI-powered retail analytics platform that performs customer segmentation using RFM analysis and KMeans clustering while generating product recommendations through collaborative filtering.

---

## 🚀 Project Overview

Shopper Spectrum is an intelligent retail analytics system developed to analyze customer purchasing behavior and provide business insights through customer segmentation and product recommendations.

The project helps businesses:

* Identify valuable customers.
* Detect at-risk customers.
* Understand purchasing behavior.
* Recommend similar products.
* Improve customer retention.
* Increase sales through targeted marketing.

---

## 📊 Dataset Description

The project uses the **Online Retail Dataset**, containing over **541,909 transaction records** from a UK-based online retail company.

### Dataset Features:

* Invoice Number
* Stock Code
* Product Description
* Quantity
* Invoice Date
* Unit Price (£)
* Customer ID
* Country

After preprocessing:

* Original Records: 541,909
* Cleaned Records: 392,692
* Unique Customers: 4,338
* Unique Products: 3,866

---

## 🧹 Data Preprocessing

The following cleaning operations were performed:

* Removed missing Customer IDs.
* Removed cancelled invoices.
* Removed negative quantities.
* Removed invalid prices.
* Removed duplicate records.
* Converted invoice dates.
* Created TotalAmount feature.

---

## 📈 Exploratory Data Analysis

The project includes multiple visualizations:

* Monthly Sales Trends
* Transaction Volume Analysis
* Country Distribution
* Product Revenue Analysis
* Customer Spending Distribution
* Customer Segment Distribution

---

## 👥 Customer Segmentation

RFM analysis was performed using:

### Recency (R)

Days since customer's last purchase.

### Frequency (F)

Number of invoices generated.

### Monetary (M)

Total amount spent.

These features were standardized and used for KMeans clustering.

---

## 🤖 KMeans Clustering

Optimal cluster selection was performed using:

* Elbow Method
* Silhouette Score

Customers were segmented into:

* 👑 High Value Customers
* ⭐ Regular Customers
* 🛒 Occasional Customers
* ⚠️ At Risk Customers

---

## 📦 Product Recommendation System

Item-based collaborative filtering was implemented.

### Steps:

* Customer-product matrix creation.
* Cosine similarity calculation.
* Similar product identification.
* Top-5 recommendation generation.

Example:

WHITE HANGING HEART T-LIGHT HOLDER

Recommended:

* GIN + TONIC DIET METAL SIGN
* RED HANGING HEART T-LIGHT HOLDER
* WASHROOM METAL SIGN
* LAUNDRY 15C METAL SIGN
* GREEN VINTAGE SPOT BEAKER

---

## 📊 Business Insights

* High Value customers contribute over 63% of revenue.
* Only 13% of customers generate the majority of sales.
* At-risk customers can be targeted through retention campaigns.
* Regular customers offer upselling opportunities.

---

## 🖥️ Streamlit Dashboard Features

### 🏠 Home Dashboard

* Revenue KPIs
* Customer Metrics
* Sales Trends
* Revenue Contribution Analysis

### 🛍️ Product Recommendation

* Product Search
* Similar Product Recommendations
* Match Score Visualization

### 👥 Customer Segmentation

* RFM Input Interface
* Real-Time Customer Prediction
* Marketing Recommendations

### 📈 Business Insights

* Segment Summary
* Revenue Contribution
* Strategic Recommendations

---

## 📂 Project Structure

```text
shopper_spectrum/
│
├── data/
│   └── online_retail.csv
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── product_similarity.pkl
│   └── product_names.pkl
│
├── notebooks/
│   └── shopper_spectrum.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* KMeans Clustering
* Collaborative Filtering
* Streamlit
* Plotly
* Matplotlib
* Seaborn
* Joblib

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Abhishek5472/shopper-spectrum.git
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run notebook:

```bash
jupyter notebook notebooks/shopper_spectrum.ipynb
```

Run dashboard:

```bash
streamlit run app.py
```

Application runs at:

```text
http://localhost:8501
```

---

## 💡 Key Findings

* 571 High Value customers generate over 63% of total revenue.
* Customer segmentation improves marketing efficiency.
* Product recommendations support cross-selling.
* RFM analysis provides valuable business insights.

---

## 🔮 Future Enhancements

* Customer login system.
* Personalized customer recommendations.
* Real-time sales forecasting.
* Customer churn prediction.
* Deep learning recommendation systems.

---

## 👨‍💻 Developed By

**Abhishek Kulkarni**

VIT Pune – CSE (AI & ML)

Labmentix Internship Project

---

## 📜 License

This project was developed for educational and internship purposes.

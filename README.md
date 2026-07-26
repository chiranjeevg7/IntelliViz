# 📊 IntelliViz v1 – AI-Powered Data Analytics Dashboard

> Transform raw data into meaningful insights through interactive visualizations, intelligent analytics, and machine learning—all within a modern Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-black?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-blue?logo=plotly)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

**IntelliViz v1** is an interactive data analytics platform built using **Python** and **Streamlit**. It enables users to upload datasets, perform data preprocessing, generate interactive visualizations, detect anomalies, build basic machine learning models, monitor threshold-based alerts, simulate real-time data streams, and export professional analytical reports.

The project was developed as a complete end-to-end analytics dashboard that combines data science, visualization, and intelligent reporting within a single web application.

---

## 🚀 Live Demo

🔗 **Application:** *https://intelliviz.streamlit.app/*

Example:

```
https://intelliviz.streamlit.app
```

---

## 📸 Screenshots

> Add screenshots inside a `screenshots/` folder.

Example:

```
screenshots/
│
├── dashboard.png
├── upload.png
├── visualization.png
├── ai_insights.png
├── alerts.png
└── reports.png
```

Then embed them:

```md
![Dashboard](screenshots/dashboard.png)
```

---

# ✨ Key Features

## 📂 Data Upload

Supports importing datasets in:

- CSV (.csv)
- Microsoft Excel (.xlsx, .xls)

Displays:

- Dataset Preview
- Number of Rows
- Number of Columns
- Data Types
- Missing Values
- Duplicate Records
- Statistical Summary

---

## 🧹 Data Cleaning

Comprehensive preprocessing tools including:

- Missing Value Handling
- Duplicate Record Removal
- Data Normalization
- Outlier Detection
- Cleaned Dataset Preview
- Data Cleaning Summary

### Missing Value Handling

- Mean Imputation
- Median Imputation
- Mode Imputation (Categorical Data)

### Outlier Detection

- Interquartile Range (IQR) Method

### Data Normalization

- StandardScaler

---

## 📊 Interactive Visualizations

Create interactive charts using Plotly.

Supported Charts:

- 📈 Line Chart
- 📊 Bar Chart
- 🥧 Pie Chart
- 🔵 Scatter Plot
- 📉 Histogram
- 🔥 Correlation Heatmap

Features include:

- Dynamic Axis Selection
- Interactive Hover Information
- Zoom & Pan
- Chart Customization

---

## 🧠 Intelligent Data Insights

The application automatically analyzes uploaded datasets and generates rule-based analytical insights.

Includes:

- Dataset Overview
- Missing Value Analysis
- Duplicate Analysis
- Numerical Feature Detection
- Categorical Feature Detection
- Correlation Analysis
- Data Quality Summary

> **Note:** The insights are generated using rule-based analytics and statistical methods rather than Large Language Models (LLMs).

---

## 🚨 Anomaly Detection

Detect unusual observations using statistical methods.

Supported Techniques:

- Z-Score Analysis
- Interquartile Range (IQR)

Highlights:

- Outlier Count
- Distribution Visualization
- Histogram Highlighting

---

## 🤖 Machine Learning

### Linear Regression

Build predictive models by selecting:

- Target Variable
- Feature Variables

Outputs:

- R² Score
- Feature Importance
- Actual vs Predicted Visualization

---

### K-Means Clustering

Cluster datasets into meaningful groups.

Features:

- User-defined Cluster Count
- Cluster Visualization
- Cluster Statistics

---

## 🚦 Threshold Alert System

Monitor numerical columns using custom thresholds.

Supports:

- Upper Threshold
- Lower Threshold

Automatically identifies:

- Threshold Violations
- Maximum Values
- Minimum Values

Maintains an alert history for analysis.

---

## 📡 Simulated Real-Time Analytics

Includes two demonstration dashboards.

### 📈 Stock Market Simulation

Displays simulated:

- Stock Price
- Volume
- High
- Low

---

### 🌦 Weather Simulation

Displays simulated:

- Temperature
- Humidity
- Atmospheric Pressure

Includes gauge-style visualizations.

> These datasets are simulated and are not connected to live APIs.

---

## 📄 Report Generation

Generate downloadable reports.

Supported Formats:

- HTML Report
- CSV Report
- JSON Summary

Reports include:

- Dataset Summary
- Statistical Analysis
- AI Insights
- Correlation Results
- Visual Summaries

---

# 🖥 Application Modules

The dashboard includes the following sections:

- 📂 Data Upload & Analysis
- 📊 Visualizations
- 🧠 AI Insights
- 🚨 Alert System
- 📡 Real-Time Data
- 📄 Export Reports

---

# 🏗 Software Architecture

The application follows an object-oriented architecture with dedicated classes for each major responsibility.

Core Components:

- DataProcessor
- AIInsights
- VisualizationEngine
- AlertSystem
- RealTimeData

The Streamlit application controls page routing, user interaction, and analytical workflows.

---

# 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Report Generation | HTML, CSV, JSON |
| Utilities | Requests, Base64, Datetime |

---

# 📦 Python Libraries

- streamlit
- pandas
- numpy
- plotly
- matplotlib
- seaborn
- scikit-learn
- openpyxl
- requests

---

# 📁 Project Structure

```
IntelliViz-v1/

│
├── Data_analysis.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙ Installation

Clone the repository.

```bash
git clone https://github.com/chiranjeevg7/IntelliViz.git
```

Move into the project directory.

```bash
cd IntelliViz
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the application.

```bash
streamlit run Data_analysis.py
```

---

# 🎯 Project Highlights

- Interactive Streamlit dashboard
- Complete data preprocessing workflow
- Multiple visualization techniques
- Rule-based intelligent insights
- Statistical anomaly detection
- Machine learning integration
- Threshold monitoring system
- Simulated real-time dashboards
- Professional report generation
- Modular object-oriented architecture

---

# ⚠ Current Limitations

- AI insights are rule-based rather than powered by generative AI.
- Real-time dashboards use simulated datasets instead of live APIs.
- No user authentication or persistent data storage.
- Analyses are processed in memory, making extremely large datasets less efficient.
- Theme switching is limited to basic visual customization.

---

# 🚀 Future Roadmap (IntelliViz Pro v2.0)

Planned enhancements include:

- Full-stack architecture (HTML, CSS, JavaScript, Node.js, Express.js)
- MongoDB integration
- User Authentication (JWT)
- Persistent user workspace
- Dataset history
- Saved reports
- Saved visualizations
- Improved dashboard UI
- Responsive SaaS interface
- Enhanced analytics engine

---

# 👨‍💻 Developer

**Chiranjeev Radheshyam Gupta**

Final Year B.Sc. Information Technology Student

Mumbai University

---

# 📜 License

This project is intended for educational, learning, and portfolio purposes.

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.

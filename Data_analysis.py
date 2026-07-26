# IntelliViz - AI-Powered Data Analytics Dashboard
# A comprehensive data analytics platform with AI-powered insights

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import requests
import io
import base64
from datetime import datetime, timedelta
import warnings
import json
import time
import sys

warnings.filterwarnings('ignore')


# Check if running in Streamlit context
def check_streamlit_context():
    """Check if code is running in Streamlit context"""
    try:
        st.session_state
        return True
    except:
        return False


# If not in Streamlit context, provide instructions
if not check_streamlit_context():
    print("=" * 60)
    print("🚀 INTELLIVIZ - AI-POWERED DATA ANALYTICS DASHBOARD")
    print("=" * 60)
    print("\n⚠️  This is a Streamlit application!")
    print("\n📋 To run this application properly:")
    print("1. Make sure you have Streamlit installed:")
    print("   pip install streamlit pandas numpy plotly matplotlib seaborn scikit-learn requests")
    print("\n2. Run the application using:")
    print(f"   streamlit run {__file__}")
    print("\n3. The dashboard will open in your web browser automatically.")
    print("\n" + "=" * 60)
    sys.exit(0)

# Page configuration
st.set_page_config(
    page_title="IntelliViz Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for modern UI
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }

    .insight-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(78, 205, 196, 0.1));
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }

    .alert-box {
        background: rgba(255, 107, 107, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }

    .success-box {
        background: rgba(78, 205, 196, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stSelectbox > div > div {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


class DataProcessor:
    """Handles all data processing and cleaning operations"""

    @staticmethod
    def load_data(uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return None

            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    @staticmethod
    def clean_data(df, options):
        """Comprehensive data cleaning"""
        if df is None or df.empty:
            return df, []

        cleaned_df = df.copy()
        cleaning_report = []

        try:
            # Handle missing values
            if options.get('handle_missing', False):
                missing_before = cleaned_df.isnull().sum().sum()

                # Numeric columns: fill with mean/median
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if cleaned_df[col].isnull().sum() > 0:
                        if options.get('missing_strategy') == 'mean':
                            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                        else:  # median
                            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)

                # Categorical columns: fill with mode
                categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if cleaned_df[col].isnull().sum() > 0:
                        mode_val = cleaned_df[col].mode()
                        if len(mode_val) > 0:
                            cleaned_df[col].fillna(mode_val[0], inplace=True)

                missing_after = cleaned_df.isnull().sum().sum()
                cleaning_report.append(f"Missing values: {missing_before} → {missing_after}")

            # Remove duplicates
            if options.get('remove_duplicates', False):
                duplicates_before = cleaned_df.duplicated().sum()
                cleaned_df.drop_duplicates(inplace=True)
                cleaning_report.append(f"Duplicates removed: {duplicates_before}")

            # Normalize numeric data
            if options.get('normalize_data', False):
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    scaler = StandardScaler()
                    cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])
                    cleaning_report.append("Numeric data normalized using StandardScaler")

            # Remove outliers using IQR method
            if options.get('remove_outliers', False):
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                outliers_removed = 0
                original_len = len(cleaned_df)

                for col in numeric_cols:
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]

                outliers_removed = original_len - len(cleaned_df)
                if outliers_removed > 0:
                    cleaning_report.append(f"Outliers removed: {outliers_removed} rows")

        except Exception as e:
            st.error(f"Error during data cleaning: {str(e)}")
            return df, ["Error during cleaning process"]

        return cleaned_df, cleaning_report


class AIInsights:
    """AI-powered insights and analysis"""

    @staticmethod
    def generate_summary_insights(df):
        """Generate comprehensive data insights"""
        if df is None or df.empty:
            return ["No data available for analysis"]

        insights = []

        try:
            # Basic statistics
            insights.append(f"📊 **Dataset Overview**: {df.shape[0]:,} rows and {df.shape[1]} columns")

            # Data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            if numeric_cols:
                insights.append(
                    f"🔢 **Numeric columns**: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})")
            if categorical_cols:
                insights.append(
                    f"🏷️ **Categorical columns**: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})")

            # Missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                worst_missing = missing_data.idxmax()
                insights.append(
                    f"⚠️ **Missing data**: {missing_data.sum():,} total missing values. '{worst_missing}' has the most ({missing_data[worst_missing]} missing)")
            else:
                insights.append("✅ **No missing values** detected in the dataset")

            # Duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                insights.append(
                    f"🔁 **Duplicate rows**: {duplicate_count} duplicate rows found ({duplicate_count / len(df) * 100:.1f}%)")
            else:
                insights.append("✅ **No duplicate rows** found in the dataset")

        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")

        return insights

    @staticmethod
    def detect_anomalies(df, column):
        """Detect anomalies in numeric columns"""
        if df is None or column not in df.columns:
            return [], None

        if column not in df.select_dtypes(include=[np.number]).columns:
            return ["Selected column is not numeric"], None

        try:
            data = df[column].dropna()
            if len(data) < 3:
                return ["Insufficient data for anomaly detection"], None

            # Z-score method
            z_scores = np.abs((data - data.mean()) / data.std())
            anomalies_zscore = data[z_scores > 3]

            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies_iqr = data[(data < lower_bound) | (data > upper_bound)]

            insights = []
            if len(anomalies_zscore) > 0:
                insights.append(f"🚨 Z-score anomalies: {len(anomalies_zscore)} values (>3 standard deviations)")

            if len(anomalies_iqr) > 0:
                insights.append(f"📊 IQR anomalies: {len(anomalies_iqr)} values outside 1.5×IQR range")

            if not insights:
                insights.append("✅ No significant anomalies detected")

            return insights, anomalies_iqr.index.tolist() if len(anomalies_iqr) > 0 else []

        except Exception as e:
            return [f"Error in anomaly detection: {str(e)}"], None

    @staticmethod
    def correlation_insights(df):
        """Generate correlation insights"""
        if df is None:
            return ["No data available"]

        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return ["Need at least 2 numeric columns for correlation analysis"]

        try:
            corr_matrix = numeric_df.corr()

            # Find strong correlations
            insights = []
            strong_correlations = []

            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.7:
                            strength = "very strong" if abs(corr_val) > 0.9 else "strong"
                            direction = "positive" if corr_val > 0 else "negative"
                            strong_correlations.append(
                                f"**{col1}** ↔ **{col2}**: {strength} {direction} correlation ({corr_val:.3f})")

            if strong_correlations:
                insights.extend(strong_correlations[:5])  # Show top 5
            else:
                insights.append("No strong correlations (>0.7) found between numeric variables")

        except Exception as e:
            insights = [f"Error in correlation analysis: {str(e)}"]

        return insights


class VisualizationEngine:
    """Handles all visualization generation"""

    @staticmethod
    def create_bar_chart(df, x_col, y_col, title="Bar Chart"):
        """Create interactive bar chart"""
        try:
            fig = px.bar(df, x=x_col, y=y_col, title=title,
                         color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(template='plotly_dark')
            return fig
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return None

    @staticmethod
    def create_line_chart(df, x_col, y_col, title="Line Chart"):
        """Create interactive line chart"""
        try:
            fig = px.line(df, x=x_col, y=y_col, title=title,
                          line_shape='spline', color_discrete_sequence=['#FF6B6B'])
            fig.update_layout(template='plotly_dark')
            return fig
        except Exception as e:
            st.error(f"Error creating line chart: {str(e)}")
            return None

    @staticmethod
    def create_scatter_plot(df, x_col, y_col, color_col=None, size_col=None, title="Scatter Plot"):
        """Create interactive scatter plot"""
        try:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                             title=title, hover_data=df.columns[:5].tolist())
            fig.update_layout(template='plotly_dark')
            return fig
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
            return None

    @staticmethod
    def create_pie_chart(df, values_col, names_col, title="Pie Chart"):
        """Create interactive pie chart"""
        try:
            fig = px.pie(df, values=values_col, names=names_col, title=title,
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(template='plotly_dark')
            return fig
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
            return None

    @staticmethod
    def create_heatmap(df, title="Correlation Heatmap"):
        """Create correlation heatmap"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                return None

            corr_matrix = numeric_df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=corr_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            fig.update_layout(title=title, template='plotly_dark')
            return fig
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            return None

    @staticmethod
    def create_histogram(df, column, title="Histogram"):
        """Create histogram with distribution"""
        try:
            fig = px.histogram(df, x=column, title=title, nbins=30,
                               color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(template='plotly_dark')
            return fig
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
            return None


class AlertSystem:
    """Handle threshold alerts and notifications"""

    @staticmethod
    def check_thresholds(df, column, threshold_config):
        """Check if values breach defined thresholds"""
        if df is None or column not in df.columns:
            return []

        alerts = []

        if column not in df.select_dtypes(include=[np.number]).columns:
            return alerts

        try:
            data = df[column].dropna()

            # Check upper threshold
            if 'upper' in threshold_config and threshold_config['upper'] is not None:
                violations = data[data > threshold_config['upper']]
                if len(violations) > 0:
                    alerts.append({
                        'type': 'upper_breach',
                        'column': column,
                        'threshold': threshold_config['upper'],
                        'violations': len(violations),
                        'max_value': violations.max(),
                        'message': f"🔴 **Upper threshold breach**: {len(violations)} values in '{column}' exceed {threshold_config['upper']:.2f}. Max value: {violations.max():.2f}"
                    })

            # Check lower threshold
            if 'lower' in threshold_config and threshold_config['lower'] is not None:
                violations = data[data < threshold_config['lower']]
                if len(violations) > 0:
                    alerts.append({
                        'type': 'lower_breach',
                        'column': column,
                        'threshold': threshold_config['lower'],
                        'violations': len(violations),
                        'min_value': violations.min(),
                        'message': f"🔵 **Lower threshold breach**: {len(violations)} values in '{column}' below {threshold_config['lower']:.2f}. Min value: {violations.min():.2f}"
                    })

        except Exception as e:
            st.error(f"Error checking thresholds: {str(e)}")

        return alerts


class RealTimeData:
    """Fetch real-time data from APIs"""

    @staticmethod
    def get_stock_data(symbol="AAPL"):
        """Fetch stock data (simulation for demo)"""
        try:
            # Generate realistic simulated stock data
            np.random.seed(42)  # For reproducible results
            dates = pd.date_range(start=datetime.now() - timedelta(days=30),
                                  end=datetime.now(), freq='D')

            # Generate more realistic stock price movement
            base_price = 150
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [base_price]

            for return_rate in returns[1:]:
                prices.append(prices[-1] * (1 + return_rate))

            stock_data = pd.DataFrame({
                'Date': dates,
                'Price': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates)),
                'Symbol': symbol
            })
            return stock_data
        except Exception as e:
            st.error(f"Error generating stock data: {str(e)}")
            return None

    @staticmethod
    def get_weather_data(city="New York"):
        """Fetch weather data (simulation for demo)"""
        try:
            dates = pd.date_range(start=datetime.now() - timedelta(days=7),
                                  end=datetime.now(), freq='D')

            # Generate seasonal temperature patterns
            base_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
            weather_data = pd.DataFrame({
                'Date': dates,
                'Temperature': base_temp + np.random.normal(0, 3, len(dates)),
                'Humidity': np.random.normal(60, 15, len(dates)).clip(0, 100),
                'Pressure': np.random.normal(1013, 10, len(dates)),
                'City': city
            })
            return weather_data
        except Exception as e:
            st.error(f"Error generating weather data: {str(e)}")
            return None


# Initialize session state safely
def init_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []


# Main Application
def main():
    """Main application function"""
    load_css()
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">🚀 IntelliViz Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")

        # Theme toggle
        theme = st.selectbox("🎨 Theme", ["Dark", "Light"])

        # Navigation
        page = st.selectbox("📄 Navigate", [
            "📊 Data Upload & Analysis",
            "📈 Visualizations",
            "🤖 AI Insights",
            "🔔 Alert System",
            "📡 Real-time Data",
            "📋 Export Reports"
        ])

        # Quick stats in sidebar
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
            st.metric("Rows", f"{df.shape[0]:,}")
            st.metric("Columns", df.shape[1])
            st.metric("Missing", f"{df.isnull().sum().sum():,}")

    # PAGE: Data Upload & Analysis
    if page == "📊 Data Upload & Analysis":
        st.header("📊 Data Upload & Processing")

        # File upload
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])

        if uploaded_file is not None:
            # Load data
            df = DataProcessor.load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df

                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                with col4:
                    st.metric("Duplicates", f"{df.duplicated().sum():,}")

                # Data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(100), use_container_width=True)

                # Data cleaning options
                st.subheader("🧹 Data Cleaning Options")
                col1, col2 = st.columns(2)

                with col1:
                    handle_missing = st.checkbox("Handle Missing Values", value=True)
                    missing_strategy = st.selectbox("Missing Value Strategy", ["mean", "median"])
                    remove_duplicates = st.checkbox("Remove Duplicates", value=True)

                with col2:
                    normalize_data = st.checkbox("Normalize Numeric Data", value=False)
                    remove_outliers = st.checkbox("Remove Outliers (IQR method)", value=False)

                cleaning_options = {
                    'handle_missing': handle_missing,
                    'missing_strategy': missing_strategy,
                    'remove_duplicates': remove_duplicates,
                    'normalize_data': normalize_data,
                    'remove_outliers': remove_outliers
                }

                if st.button("🔄 Clean Data", type="primary"):
                    with st.spinner("Cleaning data..."):
                        cleaned_df, cleaning_report = DataProcessor.clean_data(df, cleaning_options)
                        st.session_state.cleaned_df = cleaned_df

                        st.success("✅ Data cleaning completed!")

                        # Show cleaning report
                        if cleaning_report:
                            st.subheader("📝 Cleaning Report")
                            for report in cleaning_report:
                                st.info(report)

                        # Show cleaned data preview
                        st.subheader("✨ Cleaned Data Preview")
                        st.dataframe(cleaned_df.head(), use_container_width=True)

    # PAGE: Visualizations
    elif page == "📈 Visualizations":
        st.header("📈 Interactive Visualizations")

        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first in the 'Data Upload & Analysis' section.")
            return

        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df

        # Chart type selection
        chart_type = st.selectbox("📊 Select Chart Type", [
            "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap", "Histogram"
        ])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        all_cols = df.columns.tolist()

        if not all_cols:
            st.error("No columns found in the dataset.")
            return

        if chart_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", all_cols)
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols if numeric_cols else all_cols)

            if x_col and y_col and x_col != y_col:
                # Group by x_col and aggregate y_col for better visualization
                try:
                    if x_col in categorical_cols and len(df[x_col].unique()) < 50:
                        grouped_df = df.groupby(x_col)[y_col].mean().reset_index()
                        fig = VisualizationEngine.create_bar_chart(grouped_df, x_col, y_col,
                                                                   f"Average {y_col} by {x_col}")
                    else:
                        fig = VisualizationEngine.create_bar_chart(df.head(50), x_col, y_col, f"{y_col} vs {x_col}")

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")

        elif chart_type == "Line Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", all_cols)
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols if numeric_cols else all_cols)

            if x_col and y_col:
                fig = VisualizationEngine.create_line_chart(df.head(1000), x_col, y_col, f"{y_col} over {x_col}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for scatter plot")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
                with col3:
                    color_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
                    size_col = st.selectbox("Size by (optional)", [None] + numeric_cols)

                if x_col and y_col:
                    fig = VisualizationEngine.create_scatter_plot(df.head(1000), x_col, y_col, color_col, size_col,
                                                                  f"{y_col} vs {x_col}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie Chart":
            if not categorical_cols or not numeric_cols:
                st.warning("Need both categorical and numeric columns for pie chart")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    names_col = st.selectbox("Categories", categorical_cols)
                with col2:
                    values_col = st.selectbox("Values", numeric_cols)

                if names_col and values_col:
                    try:
                        # Aggregate data for pie chart
                        pie_data = df.groupby(names_col)[values_col].sum().reset_index()
                        # Limit to top 10 categories for readability
                        pie_data = pie_data.nlargest(10, values_col)
                        fig = VisualizationEngine.create_pie_chart(pie_data, values_col, names_col,
                                                                   f"Distribution of {values_col} by {names_col}")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating pie chart: {str(e)}")

        elif chart_type == "Heatmap":
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation heatmap")
            else:
                fig = VisualizationEngine.create_heatmap(df, "Correlation Matrix")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram":
            if not numeric_cols:
                st.warning("No numeric columns available for histogram")
            else:
                column = st.selectbox("Select Column", numeric_cols)
                if column:
                    fig = VisualizationEngine.create_histogram(df, column, f"Distribution of {column}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

        # Data filtering and sorting
        st.subheader("🔍 Data Filtering & Sorting")

        with st.expander("Filter Data"):
            if all_cols:
                filter_column = st.selectbox("Filter by column", all_cols, key="filter_col")

                if filter_column in numeric_cols:
                    try:
                        min_val = float(df[filter_column].min())
                        max_val = float(df[filter_column].max())
                        if min_val != max_val:
                            filter_range = st.slider(f"Select {filter_column} range", min_val, max_val,
                                                     (min_val, max_val))
                            filtered_df = df[
                                (df[filter_column] >= filter_range[0]) & (df[filter_column] <= filter_range[1])]
                        else:
                            filtered_df = df
                            st.info(f"All values in {filter_column} are the same ({min_val})")
                    except Exception as e:
                        st.error(f"Error filtering numeric data: {str(e)}")
                        filtered_df = df
                else:
                    try:
                        unique_values = df[filter_column].unique().tolist()
                        if len(unique_values) > 50:
                            st.warning(f"Too many unique values ({len(unique_values)}). Showing first 50.")
                            unique_values = unique_values[:50]

                        selected_values = st.multiselect(f"Select {filter_column} values",
                                                         unique_values,
                                                         default=unique_values[:min(5, len(unique_values))])
                        filtered_df = df[df[filter_column].isin(selected_values)] if selected_values else df
                    except Exception as e:
                        st.error(f"Error filtering categorical data: {str(e)}")
                        filtered_df = df

                st.subheader("📊 Filtered Data")
                st.dataframe(filtered_df.head(500), use_container_width=True)
                st.info(f"Showing {len(filtered_df):,} rows after filtering")

    # PAGE: AI Insights
    elif page == "🤖 AI Insights":
        st.header("🤖 AI-Powered Insights")

        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first in the 'Data Upload & Analysis' section.")
            return

        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df

        # Generate comprehensive insights
        with st.spinner("🧠 Generating AI insights..."):
            insights = AIInsights.generate_summary_insights(df)
            correlation_insights = AIInsights.correlation_insights(df)

        # Display summary insights
        st.subheader("📊 Dataset Summary Insights")
        insights_container = st.container()
        with insights_container:
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        # Correlation insights
        if correlation_insights and len(correlation_insights) > 0:
            st.subheader("🔗 Correlation Insights")
            for insight in correlation_insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        # Anomaly detection
        st.subheader("🚨 Anomaly Detection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_column = st.selectbox("Select column for anomaly detection", numeric_cols)

            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    anomaly_insights, anomaly_indices = AIInsights.detect_anomalies(df, selected_column)

                    if anomaly_insights:
                        for insight in anomaly_insights:
                            if "Error" in insight:
                                st.error(insight)
                            else:
                                st.markdown(f'<div class="alert-box">{insight}</div>', unsafe_allow_html=True)

                        # Show anomalies in chart
                        fig = VisualizationEngine.create_histogram(df, selected_column,
                                                                   f"Distribution of {selected_column} with Anomalies")

                        # Highlight anomalies
                        if fig and anomaly_indices:
                            try:
                                anomaly_data = df.loc[anomaly_indices, selected_column]
                                fig.add_scatter(x=anomaly_data, y=[0] * len(anomaly_data),
                                                mode='markers', marker=dict(color='red', size=10),
                                                name='Anomalies')
                            except Exception as e:
                                st.warning(f"Could not highlight anomalies on chart: {str(e)}")

                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for anomaly detection")

        # Machine Learning Insights
        st.subheader("🤖 Machine Learning Insights")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox("Target variable (Y)", numeric_cols)
            with col2:
                available_features = [col for col in numeric_cols if col != target_col]
                feature_cols = st.multiselect("Feature variables (X)",
                                              available_features,
                                              default=available_features[:min(3, len(available_features))])

            if target_col and feature_cols:
                if st.button("🎯 Run Linear Regression Analysis"):
                    with st.spinner("Running regression analysis..."):
                        try:
                            # Prepare data
                            X = df[feature_cols].dropna()
                            y = df.loc[X.index, target_col]

                            if len(X) < 3:
                                st.error("Insufficient data for regression analysis")
                                return

                            # Fit model
                            model = LinearRegression()
                            model.fit(X, y)

                            # Predictions
                            y_pred = model.predict(X)
                            r2 = r2_score(y, y_pred)

                            # Display results
                            st.success(f"✅ Model R² Score: {r2:.4f}")

                            # Feature importance
                            feature_importance = pd.DataFrame({
                                'Feature': feature_cols,
                                'Coefficient': model.coef_,
                                'Abs_Coefficient': np.abs(model.coef_)
                            }).sort_values('Abs_Coefficient', ascending=False)

                            st.subheader("📊 Feature Importance")
                            fig = px.bar(feature_importance, x='Feature', y='Coefficient',
                                         title="Linear Regression Coefficients",
                                         color='Coefficient', color_continuous_scale='RdBu_r')
                            fig.update_layout(template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True)

                            # Prediction vs Actual
                            comparison_df = pd.DataFrame({
                                'Actual': y,
                                'Predicted': y_pred
                            })

                            fig_scatter = px.scatter(comparison_df, x='Actual', y='Predicted',
                                                     title=f"Actual vs Predicted {target_col}",
                                                     color_discrete_sequence=['#4ECDC4'])
                            fig_scatter.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()],
                                                             mode='lines', name='Perfect Prediction',
                                                             line=dict(color='red', dash='dash')))
                            fig_scatter.update_layout(template='plotly_dark')
                            st.plotly_chart(fig_scatter, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error in regression analysis: {str(e)}")
        else:
            st.info("Need at least 2 numeric columns for machine learning analysis")

        # Clustering Analysis
        if len(numeric_cols) >= 2:
            st.subheader("🎯 Clustering Analysis")
            cluster_features = st.multiselect("Select features for clustering", numeric_cols,
                                              default=numeric_cols[:2])
            n_clusters = st.slider("Number of clusters", 2, 10, 3)

            if cluster_features and len(cluster_features) >= 2:
                if st.button("🔍 Perform K-Means Clustering"):
                    with st.spinner("Performing clustering..."):
                        try:
                            # Prepare data
                            cluster_data = df[cluster_features].dropna()

                            if len(cluster_data) < n_clusters:
                                st.error(f"Insufficient data points ({len(cluster_data)}) for {n_clusters} clusters")
                                return

                            # Standardize data
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(cluster_data)

                            # K-means clustering
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(scaled_data)

                            # Add clusters to dataframe
                            cluster_df = cluster_data.copy()
                            cluster_df['Cluster'] = clusters.astype(str)

                            # Visualize clusters
                            fig = px.scatter(cluster_df, x=cluster_features[0], y=cluster_features[1],
                                             color='Cluster', title=f"K-Means Clustering ({n_clusters} clusters)",
                                             color_discrete_sequence=px.colors.qualitative.Set1)
                            fig.update_layout(template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True)

                            # Cluster statistics
                            st.subheader("📊 Cluster Statistics")
                            cluster_stats = cluster_df.groupby('Cluster')[cluster_features].agg(
                                ['mean', 'std', 'count']).round(3)
                            st.dataframe(cluster_stats, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error in clustering analysis: {str(e)}")
            else:
                st.info("Please select at least 2 features for clustering")

    # PAGE: Alert System
    elif page == "🔔 Alert System":
        st.header("🔔 Alert System")

        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first in the 'Data Upload & Analysis' section.")
            return

        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        st.subheader("⚙️ Configure Threshold Alerts")

        if numeric_cols:
            selected_column = st.selectbox("Select column for alerts", numeric_cols)

            # Display column statistics
            col_stats = df[selected_column].describe()
            st.subheader("📊 Column Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{col_stats['mean']:.2f}")
            with col2:
                st.metric("Std Dev", f"{col_stats['std']:.2f}")
            with col3:
                st.metric("Min", f"{col_stats['min']:.2f}")
            with col4:
                st.metric("Max", f"{col_stats['max']:.2f}")

            col1, col2 = st.columns(2)

            with col1:
                enable_upper = st.checkbox("Enable upper threshold")
                if enable_upper:
                    upper_threshold = st.number_input("Upper threshold",
                                                      value=float(col_stats['75%']),
                                                      min_value=float(col_stats['min']),
                                                      max_value=float(col_stats['max']))
                else:
                    upper_threshold = None

            with col2:
                enable_lower = st.checkbox("Enable lower threshold")
                if enable_lower:
                    lower_threshold = st.number_input("Lower threshold",
                                                      value=float(col_stats['25%']),
                                                      min_value=float(col_stats['min']),
                                                      max_value=float(col_stats['max']))
                else:
                    lower_threshold = None

            threshold_config = {}
            if enable_upper:
                threshold_config['upper'] = upper_threshold
            if enable_lower:
                threshold_config['lower'] = lower_threshold

            if st.button("🔍 Check Thresholds"):
                alerts = AlertSystem.check_thresholds(df, selected_column, threshold_config)

                if alerts:
                    # Add to session state
                    for alert in alerts:
                        alert['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.alerts.extend(alerts)

                    st.subheader("🚨 Active Alerts")
                    for alert in alerts:
                        st.markdown(f'<div class="alert-box">{alert["message"]}</div>', unsafe_allow_html=True)

                    # Show visualization of violations
                    fig = VisualizationEngine.create_histogram(df, selected_column,
                                                               f"{selected_column} Distribution with Thresholds")

                    if fig:
                        # Add threshold lines
                        if 'upper' in threshold_config and threshold_config['upper'] is not None:
                            fig.add_vline(x=threshold_config['upper'], line_dash="dash",
                                          line_color="red", annotation_text="Upper Threshold")
                        if 'lower' in threshold_config and threshold_config['lower'] is not None:
                            fig.add_vline(x=threshold_config['lower'], line_dash="dash",
                                          line_color="blue", annotation_text="Lower Threshold")

                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown('<div class="success-box">✅ No threshold violations detected!</div>',
                                unsafe_allow_html=True)
        else:
            st.info("No numeric columns available for threshold monitoring")

        # Display alert history
        if st.session_state.alerts:
            st.subheader("📋 Alert History")

            # Convert alerts to DataFrame for better display
            alert_df = pd.DataFrame([
                {
                    'Timestamp': alert.get('timestamp', 'N/A'),
                    'Column': alert['column'],
                    'Type': alert['type'],
                    'Threshold': alert['threshold'],
                    'Violations': alert['violations']
                }
                for alert in st.session_state.alerts
            ])

            st.dataframe(alert_df, use_container_width=True)

            if st.button("🗑️ Clear Alert History"):
                st.session_state.alerts = []
                st.success("Alert history cleared!")
                st.rerun()

    # PAGE: Real-time Data
    elif page == "📡 Real-time Data":
        st.header("📡 Real-time Data Integration")

        # Data source selection
        data_source = st.selectbox("Select Data Source", ["Stock Market", "Weather Data"])

        if data_source == "Stock Market":
            st.subheader("📈 Live Stock Data (Demo)")

            col1, col2 = st.columns(2)
            with col1:
                stock_symbol = st.text_input("Stock Symbol", value="AAPL")
            with col2:
                refresh_interval = st.selectbox("Refresh Interval", ["Manual", "30 seconds", "1 minute"])

            # Create placeholders for real-time updates
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()

            if st.button("📊 Load Stock Data"):
                with st.spinner("Loading stock data..."):
                    stock_data = RealTimeData.get_stock_data(stock_symbol)

                    if stock_data is not None:
                        # Display metrics
                        with metrics_placeholder.container():
                            col1, col2, col3, col4 = st.columns(4)

                            current_price = stock_data['Price'].iloc[-1]
                            prev_price = stock_data['Price'].iloc[-2] if len(stock_data) > 1 else current_price
                            price_change = current_price - prev_price

                            with col1:
                                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}")
                            with col2:
                                st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,}")
                            with col3:
                                st.metric("30-Day High", f"${stock_data['Price'].max():.2f}")
                            with col4:
                                st.metric("30-Day Low", f"${stock_data['Price'].min():.2f}")

                        # Create stock chart
                        with chart_placeholder.container():
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Price'],
                                                     mode='lines+markers', name='Stock Price',
                                                     line=dict(color='#4ECDC4', width=2)))
                            fig.update_layout(
                                title=f"{stock_symbol} Stock Price (Last 30 Days)",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Store data in session state for further analysis
                        st.session_state.df = stock_data
                        st.info(
                            "💡 Stock data has been loaded as your current dataset. You can now analyze it in other sections!")

        elif data_source == "Weather Data":
            st.subheader("🌤️ Live Weather Data (Demo)")

            col1, col2 = st.columns(2)
            with col1:
                city = st.text_input("City", value="New York")
            with col2:
                refresh_interval = st.selectbox("Refresh Interval", ["Manual", "1 minute", "5 minutes"],
                                                key="weather_refresh")

            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()

            if st.button("🌡️ Load Weather Data"):
                with st.spinner("Loading weather data..."):
                    weather_data = RealTimeData.get_weather_data(city)

                    if weather_data is not None:
                        # Display metrics
                        with metrics_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Temp", f"{weather_data['Temperature'].iloc[-1]:.1f}°C")
                            with col2:
                                st.metric("Humidity", f"{weather_data['Humidity'].iloc[-1]:.1f}%")
                            with col3:
                                st.metric("Pressure", f"{weather_data['Pressure'].iloc[-1]:.1f} hPa")

                        # Create weather charts
                        with chart_placeholder.container():
                            fig = make_subplots(rows=2, cols=2,
                                                subplot_titles=(
                                                'Temperature', 'Humidity', 'Pressure', 'Current Temperature'),
                                                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                                       [{"secondary_y": False}, {"type": "indicator"}]])

                            fig.add_trace(go.Scatter(x=weather_data['Date'], y=weather_data['Temperature'],
                                                     name='Temperature', line=dict(color='#FF6B6B')), row=1, col=1)
                            fig.add_trace(go.Scatter(x=weather_data['Date'], y=weather_data['Humidity'],
                                                     name='Humidity', line=dict(color='#4ECDC4')), row=1, col=2)
                            fig.add_trace(go.Scatter(x=weather_data['Date'], y=weather_data['Pressure'],
                                                     name='Pressure', line=dict(color='#45B7D1')), row=2, col=1)

                            # Add gauge for current temperature
                            current_temp = weather_data['Temperature'].iloc[-1]
                            fig.add_trace(go.Indicator(
                                mode="gauge+number",
                                value=current_temp,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Current Temp (°C)"},
                                gauge={'axis': {'range': [-10, 40]},
                                       'bar': {'color': "#FF6B6B"},
                                       'steps': [{'range': [-10, 10], 'color': "lightblue"},
                                                 {'range': [10, 25], 'color': "lightgreen"},
                                                 {'range': [25, 40], 'color': "lightcoral"}]}
                            ), row=2, col=2)

                            fig.update_layout(height=600, showlegend=False, template='plotly_dark',
                                              title=f"Weather Data for {city} (Last 7 Days)")
                            st.plotly_chart(fig, use_container_width=True)

                        # Store data in session state for further analysis
                        st.session_state.df = weather_data
                        st.info(
                            "💡 Weather data has been loaded as your current dataset. You can now analyze it in other sections!")

    # PAGE: Export Reports
    elif page == "📋 Export Reports":
        st.header("📋 Export Analytics Reports")

        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first in the 'Data Upload & Analysis' section.")
            return

        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df

        st.subheader("📄 Report Configuration")

        # Report options
        col1, col2 = st.columns(2)
        with col1:
            report_title = st.text_input("Report Title", value="IntelliViz Analytics Report")
            include_summary = st.checkbox("Include Data Summary", value=True)
            include_visualizations = st.checkbox("Include Visualizations", value=True)

        with col2:
            include_insights = st.checkbox("Include AI Insights", value=True)
            include_correlations = st.checkbox("Include Correlation Analysis", value=True)
            report_format = st.selectbox("Export Format", ["HTML", "CSV"])

        if st.button("📊 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                try:
                    # Generate insights
                    insights = AIInsights.generate_summary_insights(df) if include_insights else []
                    correlation_insights = AIInsights.correlation_insights(df) if include_correlations else []

                    # HTML Report Content
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>{report_title}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: #ffffff; }}
                            .header {{ text-align: center; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); 
                                      padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
                            .section {{ margin: 20px 0; padding: 15px; background-color: #2d2d2d; border-radius: 8px; }}
                            .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                                     background-color: #3d3d3d; border-radius: 5px; min-width: 120px; text-align: center; }}
                            .insight {{ background-color: #2a4a4a; padding: 10px; margin: 10px 0; 
                                       border-left: 4px solid #4ECDC4; border-radius: 5px; }}
                            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                            th, td {{ border: 1px solid #555; padding: 8px; text-align: left; }}
                            th {{ background-color: #4ECDC4; color: #000; }}
                            tr:nth-child(even) {{ background-color: #2a2a2a; }}
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <h1>{report_title}</h1>
                            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        </div>
                    """

                    if include_summary:
                        html_content += f"""
                        <div class="section">
                            <h2>📊 Data Summary</h2>
                            <div style="text-align: center;">
                                <div class="metric">
                                    <h3>Rows</h3>
                                    <p>{df.shape[0]:,}</p>
                                </div>
                                <div class="metric">
                                    <h3>Columns</h3>
                                    <p>{df.shape[1]}</p>
                                </div>
                                <div class="metric">
                                    <h3>Missing Values</h3>
                                    <p>{df.isnull().sum().sum():,}</p>
                                </div>
                                <div class="metric">
                                    <h3>Duplicates</h3>
                                    <p>{df.duplicated().sum():,}</p>
                                </div>
                            </div>

                            <h3>Column Information</h3>
                            <table>
                                <tr><th>Column</th><th>Type</th><th>Non-Null Count</th><th>Null Count</th></tr>
                        """

                        for col in df.columns:
                            non_null = df[col].count()
                            null_count = df[col].isnull().sum()
                            col_type = str(df[col].dtype)
                            html_content += f"<tr><td>{col}</td><td>{col_type}</td><td>{non_null:,}</td><td>{null_count:,}</td></tr>"

                        html_content += "</table></div>"

                    if include_insights and insights:
                        html_content += """
                        <div class="section">
                            <h2>🤖 AI Insights</h2>
                        """
                        for insight in insights:
                            # Remove markdown formatting for HTML
                            clean_insight = insight.replace("**", "<strong>").replace("**", "</strong>")
                            html_content += f'<div class="insight">{clean_insight}</div>'
                        html_content += "</div>"

                    if include_correlations and correlation_insights:
                        html_content += """
                        <div class="section">
                            <h2>🔗 Correlation Analysis</h2>
                        """
                        for insight in correlation_insights:
                            clean_insight = insight.replace("**", "<strong>").replace("**", "</strong>")
                            html_content += f'<div class="insight">{clean_insight}</div>'
                        html_content += "</div>"

                    # Descriptive statistics
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 0:
                        html_content += f"""
                        <div class="section">
                            <h2>📈 Descriptive Statistics</h2>
                            {numeric_df.describe().round(3).to_html(classes='table')}
                        </div>
                        """

                    # Data sample
                    html_content += f"""
                        <div class="section">
                            <h2>📋 Data Sample (First 10 Rows)</h2>
                            {df.head(10).to_html(classes='table')}
                        </div>
                    </body>
                    </html>
                    """

                    if report_format == "HTML":
                        # Provide download link for HTML
                        b64 = base64.b64encode(html_content.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="{report_title.replace(" ", "_")}.html">📥 Download HTML Report</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Preview
                        st.subheader("📖 Report Preview")
                        with st.expander("View Report Preview", expanded=False):
                            st.components.v1.html(html_content, height=600, scrolling=True)

                    elif report_format == "CSV":
                        # Generate comprehensive CSV report
                        csv_buffer = io.StringIO()

                        # Write report header
                        csv_buffer.write(f"{report_title}\n")
                        csv_buffer.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                        # Data summary
                        csv_buffer.write("DATA SUMMARY\n")
                        csv_buffer.write(f"Rows,{df.shape[0]}\n")
                        csv_buffer.write(f"Columns,{df.shape[1]}\n")
                        csv_buffer.write(f"Missing Values,{df.isnull().sum().sum()}\n")
                        csv_buffer.write(f"Duplicates,{df.duplicated().sum()}\n\n")

                        # Column information
                        csv_buffer.write("COLUMN INFORMATION\n")
                        csv_buffer.write("Column,Data Type,Non-Null Count,Null Count\n")
                        for col in df.columns:
                            non_null = df[col].count()
                            null_count = df[col].isnull().sum()
                            col_type = str(df[col].dtype)
                            csv_buffer.write(f'"{col}",{col_type},{non_null},{null_count}\n')

                        csv_buffer.write("\n")

                        # Descriptive statistics for numeric columns
                        numeric_df = df.select_dtypes(include=[np.number])
                        if len(numeric_df.columns) > 0:
                            csv_buffer.write("DESCRIPTIVE STATISTICS\n")
                            numeric_df.describe().to_csv(csv_buffer)
                            csv_buffer.write("\n")

                        # AI Insights
                        if include_insights and insights:
                            csv_buffer.write("AI INSIGHTS\n")
                            for i, insight in enumerate(insights, 1):
                                # Remove markdown and HTML formatting
                                clean_insight = insight.replace("**", "").replace("*", "")
                                csv_buffer.write(f'"{i}","{clean_insight}"\n')
                            csv_buffer.write("\n")

                        # Correlation insights
                        if include_correlations and correlation_insights:
                            csv_buffer.write("CORRELATION INSIGHTS\n")
                            for i, insight in enumerate(correlation_insights, 1):
                                clean_insight = insight.replace("**", "").replace("*", "")
                                csv_buffer.write(f'"{i}","{clean_insight}"\n')
                            csv_buffer.write("\n")

                        # Sample data
                        csv_buffer.write("DATA SAMPLE (First 50 rows)\n")
                        df.head(50).to_csv(csv_buffer, index=False)

                        csv_content = csv_buffer.getvalue()
                        b64 = base64.b64encode(csv_content.encode()).decode()
                        href = f'<a href="data:text/csv;base64,{b64}" download="{report_title.replace(" ", "_")}.csv">📥 Download CSV Report</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Show preview of CSV content
                        st.subheader("📄 CSV Report Preview")
                        preview_lines = csv_content.split('\n')[:20]
                        st.text('\n'.join(preview_lines) + '\n...(truncated)')

                    st.success("✅ Report generated successfully!")

                    # Additional export options
                    st.subheader("📊 Additional Export Options")

                    col1, col2 = st.columns(2)
                    with col1:
                        # Export current dataset
                        if st.button("📁 Export Current Dataset (CSV)"):
                            csv_data = df.to_csv(index=False)
                            b64 = base64.b64encode(csv_data.encode()).decode()
                            href = f'<a href="data:text/csv;base64,{b64}" download="dataset.csv">📥 Download Dataset</a>'
                            st.markdown(href, unsafe_allow_html=True)

                    with col2:
                        # Export data summary
                        if st.button("📋 Export Data Summary (JSON)"):
                            summary_data = {
                                "dataset_info": {
                                    "shape": df.shape,
                                    "columns": df.columns.tolist(),
                                    "missing_values": df.isnull().sum().to_dict(),
                                    "data_types": df.dtypes.astype(str).to_dict()
                                },
                                "numeric_summary": df.describe().to_dict() if len(
                                    df.select_dtypes(include=[np.number]).columns) > 0 else {},
                                "insights": insights,
                                "correlation_insights": correlation_insights,
                                "generated_at": datetime.now().isoformat()
                            }

                            json_str = json.dumps(summary_data, indent=2)
                            b64 = base64.b64encode(json_str.encode()).decode()
                            href = f'<a href="data:application/json;base64,{b64}" download="data_summary.json">📥 Download JSON Summary</a>'
                            st.markdown(href, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.error("Please check your data and try again.")

    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for Using IntelliViz:

    1. **Start with Data Upload**: Always begin by uploading your CSV or Excel file in the 'Data Upload & Analysis' section
    2. **Clean Your Data**: Use the data cleaning options to handle missing values, duplicates, and outliers
    3. **Explore Visualizations**: Create interactive charts to understand your data patterns
    4. **Get AI Insights**: Let the AI analyze your data and provide intelligent insights
    5. **Monitor with Alerts**: Set up threshold alerts for important metrics
    6. **Real-time Data**: Try the demo stock and weather data features
    7. **Export Reports**: Generate comprehensive reports for sharing and documentation

    ### 🚀 Pro Tips:
    - **Filter your data** in the Visualizations section to focus on specific subsets
    - **Use correlation analysis** to find relationships between variables
    - **Try clustering analysis** to discover hidden patterns in your data
    - **Set up alerts** for business-critical thresholds
    - **Export reports** in different formats for different audiences
    """)


if __name__ == "__main__":
    main()
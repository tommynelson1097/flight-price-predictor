# Streamlit & Visualization
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import io

# Scikit-learn: Modeling & Evaluation
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,    mean_squared_error, r2_score

st.set_page_config(
    page_title="Airline Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)

st.markdown("""
# ✈️ Airline Flight Price Predictor
Welcome to the interactive dashboard for exploring and predicting airline flight prices!
""")

@st.cache_data #avoiding reloading data every time
def load_data():
    path = 'airlines_flights_data.csv'  # Use local file in current folder
    df = pd.read_csv(path)
    return df
df = load_data().copy()

# Create tabs
tab1, tab2, tab3 = st.tabs([ "✈️ Flight Data Exploration", "📊 Flight Price Distribution","🧮 Interactive Flight Predictor Form"])

# Tab 1: Histogram Page
with tab1:
    # --- Model Performance Summary (Static) ---
    st.markdown("---")
    st.header("Model Performance Summary (Linear Regression)")
    st.markdown("""
    <ul>
        <li><b>Mean Squared Error (MSE):</b> 45,154,392.45</li>
        <li><b>Root Mean Squared Error (RMSE):</b> 6,719.70</li>
        <li><b>Mean Absolute Error (MAE):</b> 4,572.61</li>
        <li><b>R-squared (R²) Score:</b> 0.9123</li>
    </ul>
    """, unsafe_allow_html=True)

    st.subheader("Head of Data - 100 Sample Records")
    st.write(df.head(100), use_container_width=True)

    # Display DataFrame info in a table format
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Dtype": df.dtypes.astype(str)
    })
    st.subheader("Info")
    st.dataframe(info_df)

    # Basic Data Exploration
    buffer = io.StringIO()
    s = buffer.getvalue()
    st.text(s)
    st.subheader("Describe")
    # Display DataFrame summary statistics
    st.write(df.describe(), use_container_width=True)



with tab2:
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bins, patches = ax.hist(df['price'], bins=20, color='turquoise', edgecolor='black')
    ax.set_title('Binned Distribution of Flight Prices')
    ax.set_xlabel('Price')
    ax.set_ylabel('Number of Flights')
    ax.grid(axis='y', alpha=0.75)

    # Add labels on top of each bin
    for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
        ax.text((bin_left + bin_right) / 2, count, int(count), ha='center', va='bottom', fontsize=8)

    # Display in Streamlit
    st.pyplot(fig)

    #Box plot for flight prices
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.boxplot(df['price'], vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    ax2.set_title('Box Plot of Flight Prices')
    ax2.set_xlabel('Price')
    st.pyplot(fig2)



####### Data Processing ########

# 1. Create 'route' feature for one-hot encoding
df_proc = df.copy()
if 'source_city' in df_proc.columns and 'destination_city' in df_proc.columns:
    df_proc['route'] = df_proc['source_city'] + '_' + df_proc['destination_city']

# 2. Drop columns not needed for analysis
cols_to_drop = ['index', 'flight', 'source_city', 'destination_city']
df_proc.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# 3. One-hot encode categorical variables
categorical_cols = ['airline', 'route', 'stops', 'class', 'departure_time', 'arrival_time']
df_encoded = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=True)

# 4. Split features and target
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



with tab3:
    st.subheader("Predict Flight Price")

    # Example input fields (customize as needed)
    airline = st.selectbox("Airline", df_proc['airline'].unique())
    route = st.selectbox("Route", df_proc['route'].unique())
    departure_time = st.selectbox("Departure Time", df_proc['departure_time'].unique())
    arrival_time = st.selectbox("Arrival Time", df_proc['arrival_time'].unique())
    stops = st.selectbox("Number of Stops", df_proc['stops'].unique())
    travel_class = st.selectbox("Class", df_proc['class'].unique())
    duration = st.number_input("Duration (minutes)", min_value=0, value=int(df_proc['duration'].mean()))
    days_left = st.number_input("Days Left", min_value=0, value=int(df_proc['days_left'].mean()))


    error_messages = []

        # Input validation
    if duration <= 0:
        error_messages.append("Duration must be greater than 0.")
    if days_left < 0:
        error_messages.append("Days Left cannot be negative.")


    if st.button("Predict Price"):
        if error_messages:
            for msg in error_messages:
                st.error(msg)   
        else:

            # Prepare input as DataFrame
            input_dict = {
                'airline': [airline],
                'route': [route],
                'departure_time': [departure_time],
                'arrival_time': [arrival_time],
                'stops': [stops],
                'class': [travel_class],
                'duration': [duration],
                'days_left': [days_left]
            }
            input_df = pd.DataFrame(input_dict)

            # One-hot encode input to match training features
            input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
            # Align input with training data columns (fill missing columns with 0)
            input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

            # Scale input
            input_scaled = scaler.transform(input_encoded)

            # Predict using the trained model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)  # In production, load a pre-trained model instead of fitting here!
            predicted_price = model.predict(input_scaled)[0]



            # Calculate 95% confidence interval for the prediction
            y_pred_train = model.predict(X_train_scaled)
            residuals = y_train - y_pred_train
            dof = X_train_scaled.shape[0] - X_train_scaled.shape[1] - 1
            residual_std_error = np.sqrt(np.sum(residuals ** 2) / dof)

            # Add intercept to input for variance calculation
            X_train_scaled_with_intercept = np.column_stack((np.ones(X_train_scaled.shape[0]), X_train_scaled))
            input_scaled_with_intercept = np.insert(input_scaled, 0, 1, axis=1)
            # Calculate (X'X)^-1
            xtx_inv = np.linalg.inv(np.dot(X_train_scaled_with_intercept.T, X_train_scaled_with_intercept))
            # Standard error of prediction
            se_pred = np.sqrt(
                residual_std_error ** 2 * (
                    1 + np.dot(np.dot(input_scaled_with_intercept, xtx_inv), input_scaled_with_intercept.T)
                )
            )
            t_value = stats.t.ppf(0.975, dof)
            ci_lower = predicted_price - t_value * se_pred[0][0]
            ci_upper = predicted_price + t_value * se_pred[0][0]

            st.write("Entered Data:")
            st.dataframe(input_df)
            st.success(f"Predicted Flight Price: ${predicted_price:,.2f}")
            st.info(f"95% Confidence Interval: $ {ci_lower:,.2f}  to  $ {ci_upper:,.2f}")
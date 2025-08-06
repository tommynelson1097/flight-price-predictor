# Airline Flight Price Predictor

Welcome to the Airline Flight Price Predictor project!  
This interactive dashboard allows users to explore airline flight data, visualize price distributions, and predict flight prices based on custom input features.

---

## Folder Contents

- `streamlit_app.py`  
  Main Streamlit application for data exploration and flight price prediction.

- `airlines_flights_data.csv`  
  The dataset containing flight information and prices.

- `Airline-Flight-Price-Analysis.ipynb`  
  Jupyter notebook for exploratory data analysis and model development.

---

## How to Run Locally 

1. **Start the Streamlit app**  
   In your terminal, run:
   ```
   streamlit run streamlit_app.py
   ```

2. **Open in your browser**  
   The app will launch automatically. If not, visit [http://localhost:8501](http://localhost:8501).

---

## Data Processing & Modeling

- Data is cleaned, encoded, and scaled before modeling.
- Linear Regression is used for price prediction (can be replaced with other models).
- Input validation and error messages are included for user-friendly experience.

---

## Features

- **Price Distribution Tab**  
  View histograms and box plots of flight prices.

- **Exploratory Analysis Tab**  
  Explore sample data, column info, and summary statistics.

- **Interactive Flight Predictor Form**  
  Enter flight details and get a predicted price using a regression model.

---


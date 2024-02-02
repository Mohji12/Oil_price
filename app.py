import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import joblib

st.header(":bar_chart: Oil Price Prediction Project ",divider='rainbow')

# Load your dataset
# Replace 'your_dataset.csv' with the actual filename of your dataset
df = pd.read_csv('oil.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

df1 = df.iloc[:,0:5]
print(df1)


# Set the 'Date' column as the index

# Display basic information about the dataset
st.write("Basic Information:")
st.write(df.info())

# Display summary statistics of numerical columns
st.write("\nSummary Statistics:")
st.write(df1.describe())

# Visualize the time series of oil prices
st.line_chart(df1['Price'])

# Visualize distribution of oil prices

st.header('Histogram of Price Column',divider='rainbow')
fig, ax = plt.subplots()
ax.hist(df1['Price'], bins=30, edgecolor='black',color='purple')
st.pyplot(fig)


# Correlation heatmap
#correlation_matrix = df1.corr()
#heat = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#st.pyplot(heat)

# Pairplot for selected columns
#selected_columns = ['Price', 'Open', 'High', 'Low', 'Volume']
#st.pair_plot(df[selected_columns])
st.header("Pairplots",divider='rainbow')
pairplot = sns.pairplot(df1,palette='pastel')
st.pyplot(pairplot)

# Display correlation heatmap in the main content area
st.header('Correlation Heatmap of the Data',divider='rainbow')
    
# Calculate correlation matrix
corr_matrix = df1.corr()

# Display heatmap using Seaborn
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
st.pyplot(fig)

# Seasonal decomposition of oil prices
st.header("Seasonal Decompose Graph",divider='rainbow')
result = seasonal_decompose(df['Price'], model='additive', period=30)  # Adjust period as needed
result.plot()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

df2 = df.iloc[:,0:2]

df2.set_index('Date', inplace=True)


# Feature Engineering: Daily Price Change
df2['Daily_Price_Change'] = df2['Price'].diff()

# Feature Engineering: Rolling Mean
window_size = 7  # Adjust window size as needed
df2['Rolling_Mean'] = df2['Price'].rolling(window=window_size).mean()

# Drop rows with NaN values resulting from feature engineering
df2.dropna(inplace=True)

# Split the data into features (X) and target variable (y)
X = df2.drop('Price', axis=1)
y = df2['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeling: ARIMA
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

order = (5, 1, 0)  # Adjust order as needed
arima_model = ARIMA(train['Price'], order=order)
fit_model_arima = arima_model.fit()


st.header('ARIMA model for Future price Prediction')

# Sidebar
st.sidebar.header('Input for ARIMA model')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('today'))
end_date = st.sidebar.date_input('End Date')

# Prediction
if st.sidebar.button('Predict Future Prices'):
    # Make ARIMA predictions
    arima_future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    arima_future_predictions = fit_model_arima.get_forecast(steps=len(arima_future_dates)).predicted_mean
    arima_future_df = pd.DataFrame({'Date': arima_future_dates, 'ARIMA_Predicted_Price': arima_future_predictions})

    # Display ARIMA predictions
    st.subheader('ARIMA Predicted Future Oil Prices')
    st.table(arima_future_df)

    # Plot ARIMA predictions
    plt.figure(figsize=(12, 6))
    plt.plot(arima_future_df['Date'], arima_future_df['ARIMA_Predicted_Price'], label='ARIMA Predicted Prices')
    plt.title('ARIMA Future Oil Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('ARIMA Predicted Price')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the saved Random Forest model
model = joblib.load('random_forest_model_save14.pkl')

st.header('Oil Price Prediction Using Random Forest',divider='rainbow')

# Input features for prediction

st.text("Input for Random Forst Model")
year = st.number_input('Year', value=2023)
month = st.number_input('Month', min_value=1, max_value=12, value=1)
day = st.number_input('Day', min_value=1, max_value=31, value=1)
volume = st.number_input('Volume', value=100000)

# Make a prediction
prediction_data = {'Year': [year], 'Month': [month], 'Day': [day], 'Volume': [volume]}
input_df = pd.DataFrame(prediction_data)
prediction = model.predict(input_df)

st.write('Predicted Oil Price:', prediction[0])

# Information
st.info(
    "This Streamlit app presents the entire Oil Price Prediction project. "
    "It includes data preprocessing, exploratory data analysis (EDA)"
    "ARIMA modeling and Random Forest Model. Adjust the settings in the sidebar to predict future oil prices using ARIMA."
)


st.header('_Streamlit_ is :blue[cool] :sunglasses:')
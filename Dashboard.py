#!/usr/bin/env python
# coding: utf-8

# In[16]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Load the data
df = pd.read_excel('ConsumptionData.xlsx')
df.set_index('Year', inplace=True)

# Define economic periods
economic_periods = {
    "Great Depression": (1929, 1933),
    "World War II Period": (1941, 1945),
    "Post-War Period": (1945, 1973),
    "Oil Crisis":(1973, 1974),
    "Stagflation Period": (1974, 1982),
    "Great Moderation": (1982, 2007),
    "Great Recession": (2007, 2009),
    "Covid-19 Pandemic": (2019, 2021),
    "All Periods": (1929, 2021)
}

# Streamlit app
st.title('US Economic Consumption Analysis Dashboard')

# Dropdown for economic periods
selected_period = st.selectbox("Select a period", list(economic_periods.keys()))
period_range = economic_periods[selected_period]

# Year range slider
year_range = st.slider("Select the year range", 1929, 2021, period_range)

# Filter dataframe by year range
df_filtered = df[(df.index >= year_range[0]) & (df.index <= year_range[1])]

# Display the filtered data
st.dataframe(df_filtered)

# Plotting
st.subheader(f'Data for {selected_period} from {year_range[0]} to {year_range[1]}')

# Create line charts for each pair of variables
st.line_chart(df_filtered[["Consumption (Billion USD)", "Revenue  (Billion USD)"]])
st.line_chart(df_filtered[["Population"]])
st.line_chart(df_filtered[["Price Index"]])

# Correlation matrix
st.subheader('Correlation Matrix')
corr = df_filtered.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax)
st.pyplot(fig)

# ...
# Regression analysis
x = df_filtered.index.values.reshape(-1, 1)  # years
y = df_filtered['Consumption (Billion USD)'].values.reshape(-1, 1)  # consumption
model = LinearRegression()
model.fit(x, y)
st.subheader(f'Regression score: {round(model.score(x, y),3)}')

# Plot regression
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, model.predict(x), color='orange')  # use model.predict(x) instead of y_pred

# Custom x-axis formatter
if isinstance(df_filtered.index, pd.DatetimeIndex):
    formatter = FuncFormatter(lambda x, _: pd.to_datetime(x).year)
else:
    formatter = FuncFormatter(lambda x, _: '{:.0f}'.format(x))  # whole number
ax.xaxis.set_major_formatter(formatter)

# Add title and labels
ax.set_title('Regression Line')
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (Billion USD)')

st.pyplot(fig)

# Prediction functionality
st.subheader('Predict Consumption (Billion USD) for a given year')
input_year = st.number_input('Enter a year', min_value=1929, max_value=2021, step=1)
if st.button('Predict'):
    input_data = np.array([[input_year]])  # input_data should be 2D array
    predicted_consumption = model.predict(input_data)
    st.write(f'Predicted Consumption (Billion USD) for year {input_year} is {round(predicted_consumption[0][0], 1)}')



# In[ ]:





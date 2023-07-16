#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
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
    "Covid-19 Pandemic": (2019, 2021)
}

# Streamlit code
st.title('US Consumption Habits Dashboard')

# Dropdown for economic periods
selected_period = st.selectbox("Select an economic period", list(economic_periods.keys()))
period_range = economic_periods[selected_period]

# Year range slider
year_range = st.slider("Select the year range", 1929, 2021, period_range)

# Filter dataframe by year range
df_filtered = df[(df.index >= year_range[0]) & (df.index <= year_range[1])]

# Plotting
st.subheader(f'{selected_period} from {year_range[0]} to {year_range[1]}')

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
def plot_consumption_inflation(df):
    st.header('Consumption vs Price Index over the years')
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Consumption', color=color)
    ax1.plot(df.index, df['Consumption (Billion USD)'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Price Index', color=color) 
    ax2.plot(df.index, df['Price Index'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    st.pyplot(fig)

plot_consumption_inflation(df_filtered)

# Custom x-axis formatter
if isinstance(df_filtered.index, pd.DatetimeIndex):
    formatter = FuncFormatter(lambda x, _: pd.to_datetime(x).year)
else:
    formatter = FuncFormatter(lambda x, _: '{:.0f}'.format(x))  # whole number
ax.xaxis.set_major_formatter(formatter)


# Prediction functionality
st.subheader('Predict Consumption (Billion USD) for a given year')
input_year = st.number_input('Enter a year', min_value=1929, max_value=2021, step=1)
if st.button('Predict'):
    input_data = np.array([[input_year]])  # input_data should be 2D array
    predicted_consumption = model.predict(input_data)
    st.write(f'Predicted Consumption (Billion USD) for year {input_year}: {predicted_consumption[0][0]}')


# In[ ]:





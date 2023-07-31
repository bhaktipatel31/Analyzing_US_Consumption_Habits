#Load the libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Load data
df = pd.read_excel('ConsumptionData.xlsx')

# Set the "Year" column as the index
df.set_index('Year', inplace=True)

# Define the historical time periods
periods = {
    "Great Depression": (1929, 1933),
    "World War II Period": (1941, 1945),
    "Post-War Period": (1945, 1973),
    "Oil Crisis": (1973, 1974),
    "Stagflation Period": (1974, 1982),
    "Great Moderation": (1982, 2007),
    "Great Recession": (2007, 2009),
    "Covid-19 Pandemic": (2019, 2021),
    "All Periods": (1929, 2021)
}

# Define the color scheme
color_scheme = px.colors.qualitative.Plotly

# Title
st.title('US Economic Consumption Habits Dashboard')

# Dropdown for historical periods
selected_period = st.selectbox("Select a period", list(periods.keys()))
period_range = periods[selected_period]

# Add a slider to select the year range
year_range = st.slider('Select the year range', 1929, 2021, period_range)

# Filter the data based on the selected year range
df_filtered = df[(df.index >= year_range[0]) & (df.index <= year_range[1])]

# Show the dataframe
st.write(df_filtered)


# Plotting
st.subheader(f'Data for {selected_period} from {year_range[0]} to {year_range[1]}')

# Create a plot for Consumption
fig = px.line(df_filtered, y='Consumption (Billion USD)', title='Consumption Over Time', color_discrete_sequence=[color_scheme[6]])
st.plotly_chart(fig)

# Create a plot for Revenue
fig2 = px.line(df_filtered, y='Revenue  (Billion USD)', title='Revenue Over Time', color_discrete_sequence=[color_scheme[4]])
st.plotly_chart(fig2)

# Create a plot for Population
fig3 = px.line(df_filtered, y='Population', title='Population Over Time', color_discrete_sequence=[color_scheme[7]])
st.plotly_chart(fig3)

# Create a plot for Price Index
fig4 = px.line(df_filtered, y='Price Index', title='Price Index Over Time', color_discrete_sequence=[color_scheme[5]])
st.plotly_chart(fig4)

# # Correlation matrix
# st.subheader('Correlation Matrix')
# corr = df_filtered.corr()
# fig, ax = plt.subplots()
# sns.heatmap(corr, annot=True, ax=ax)
# st.pyplot(fig)

####REGRESSION
# Regression analysis
X = df_filtered[['Revenue  (Billion USD)', 'Population', 'Price Index']]
y = df_filtered['Consumption (Billion USD)']
reg = LinearRegression().fit(X, y)
st.write(f'Regression score: {reg.score(X, y)}')

#### PREDICT
# Add a number input to select the year for prediction
st.subheader('Predict Consumption (Billion USD) for a given year')
year_to_predict = st.number_input('Enter a year', min_value=1929, max_value=2021, step=1)

# Prepare the data for prediction
predict_data = df.loc[year_to_predict][['Revenue  (Billion USD)', 'Population', 'Price Index']].values.reshape(1, -1)

# Check if the data for the selected year is available
if not pd.Series(predict_data.flatten()).isnull().any():
    # Predict the Consumption for the selected year
    predicted_consumption = reg.predict(predict_data)
    st.write(f'Predicted Consumption (Billion USD) for {year_to_predict}: {predicted_consumption[0]}')
else:
    st.write(f'Data for {year_to_predict} is not available. The model will predict based on the last available data.')


# If the data for the selected year is not available, use the data from the last available year
if pd.Series(predict_data.flatten()).isnull().any():
    last_year = df.index.max()
    predict_data = df.loc[last_year][['Revenue  (Billion USD)', 'Population', 'Price Index']].values.reshape(1, -1)

    # Predict the Consumption for the selected year
    predicted_consumption = reg.predict(predict_data)

    # Display the predicted Consumption
    st.write(f'Predicted Consumption (Billion USD) for {year_to_predict} based on the data from {last_year}: {predicted_consumption[0]}')

# Predict the Consumption values
df_filtered['Predicted Consumption (Billion USD)'] = reg.predict(X)

# Calculate the standard error of the predictions
pred_error = df_filtered['Consumption (Billion USD)'] - df_filtered['Predicted Consumption (Billion USD)']
std_error = pred_error.std()

# Calculate confidence intervals (95% confidence interval)
confidence_interval = 1.96 * std_error


#### Create a plot to compare actual and predicted Consumption
fig5 = px.line(df_filtered, y=['Consumption (Billion USD)', 'Predicted Consumption (Billion USD)'],
               title='Actual vs Predicted Consumption Over Time',
               line_shape='linear',  # Change line_shape to 'linear' for clear lines
               labels={'value': 'Consumption (Billion USD)'})  # Update axis label for better clarity

# Add confidence intervals with reduced transparency
fig5.add_traces(go.Scatter(x=df_filtered.index, y=df_filtered['Predicted Consumption (Billion USD)'] - confidence_interval,
                           line=dict(color='rgba(128,177,211,1.0)'), showlegend=False, name='95% CI',
                           hoverinfo='none'))
fig5.add_traces(go.Scatter(x=df_filtered.index, y=df_filtered['Predicted Consumption (Billion USD)'] + confidence_interval,
                           line=dict(color='rgba(128,177,211,1.0)'), showlegend=False, fill='tonexty',
                           hoverinfo='none'))

# Add actual data points with larger markers and different color
fig5.add_traces(go.Scatter(x=df_filtered.index, y=df_filtered['Consumption (Billion USD)'],
                           mode='markers', name='Actual Consumption', marker=dict(size=7, color=color_scheme[7]), line=dict(color=color_scheme[7]),
                           hovertemplate='<b>Year: %{x}</b><br>Actual Consumption: %{y} Billion USD'))


# Add legend and update axis labels
fig5.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                   xaxis_title='Year', yaxis_title='Consumption (Billion USD)')

# Add grid lines
fig5.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig5.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

st.plotly_chart(fig5)

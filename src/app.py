import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from knapsack import heuristic_budget_cut
import plotly.express as px

# # Read local data
# data = pd.read_csv('../data/Branch Data.csv')
# required_columns = ['Branch', 'geohash', 'population', 'Value', 'Weight']
# if not all(col in data.columns for col in required_columns):
#     st.error("The CSV file is missing one or more required columns.")
#     st.stop()

# Read from Google Drive
url = "https://docs.google.com/spreadsheets/d/1PQSfzsXm1oAMe9vOcD8L1jiIxE2lpuu-A6Dp7ZSAsSE/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)
data = conn.read(spreadsheet=url, names=['Branch', 'geohash', 'population', 'Value', 'Weight'], header=None, skiprows=1)

# Data cleaning
data = data.dropna(subset=['Branch', 'population', 'Value', 'Weight'])
data['population'] = pd.to_numeric(data['population'], errors='coerce')
data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
data['Weight'] = pd.to_numeric(data['Weight'], errors='coerce')

# Filters
capacity_max = st.number_input("Maximum Capacity", min_value=50000, max_value=200000, value=100000, step=1000)
capacity_min = st.number_input("Minimum Capacity", min_value=10000, max_value=100000, value=50000, step=1000)
cut_step = st.number_input("Cut Step", min_value=1000, max_value=50000, value=10000, step=1000)

# Knapsack
itm = 'Branch'
val = 'Value'
pop = 'population'
wgh = 'Weight'

solution = heuristic_budget_cut(data, itm, val, pop, wgh, capacity_max, capacity_min, cut_step)

# Result
st.write("Binary Knapsack Result:")
chart_data = solution[['Budget', 'Total_Value']]
fig = px.line(chart_data, x='Budget', y='Total_Value', title='Budget vs Total Value')
st.plotly_chart(fig)
st.dataframe(solution)
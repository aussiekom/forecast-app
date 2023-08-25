# pip install streamlit fbprophet yfinance plotly
# pip install streamlit pandas plotly yfinance fbprophet
# conda install -c conda-forge prophet ---to escape the errors during forecasting

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(page_title="Dashboard", layout="wide")
st.subheader("Forecasting Application")
st.markdown("##")

df = pd.read_csv('superstore.csv')

# Sidebar filters
st.sidebar.header("Please select")
category = st.sidebar.selectbox("Category", options=df["Category"].unique())

# Dynamically populate sub-category options based on selected category
sub_category_options = df[df["Category"] == category]["Sub-Category"].unique()
sub_category = st.sidebar.selectbox("Sub-Category", options=sub_category_options)

df_selection = df.query("Category == @category & `Sub-Category` == @sub_category")

def Overview():
    # Display data in an expander
    with st.expander("Dataset Overview"):
        show_data = st.multiselect('Filter: ', df_selection.columns, default=["Category", "Sub-Category", "Sales", "Profit"])
        st.dataframe(df_selection[show_data], use_container_width=True)

    # Compute top analytics
    total_sales = float(df_selection['Sales'].sum())
    total_profit = float(df_selection['Profit'].sum())

    total1, total2 = st.columns(2, gap='large')
    with total1:
        st.info('Total Sales')
        st.metric(label="Total Sales", value=f"${total_sales:,.2f}")

    with total2:
        st.info('Total Profit')
        st.metric(label="Total Profit", value=f"${total_profit:,.2f}")

    st.markdown("""---""")

    # Line chart for sales and profit over time
    time_series_data = df_selection.groupby("Order Date").sum()[["Sales", "Profit"]]
    time_series_data.reset_index(inplace=True)

    fig_time_series = px.line(
        time_series_data,
        x="Order Date",
        y=["Sales", "Profit"],
        title="<b>Sales and Profit Over Time</b>",
        labels={"value": "Amount", "Order Date": "Date"},
        template="plotly_white"
    )

    fig_time_series.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False)
    )

    # Render the time series line chart
    st.plotly_chart(fig_time_series, use_container_width=True)

# Forecast
TODAY = date.today().strftime("%Y-%m-%d")

def forecast():
    st.title('Sales Forecast')

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    # Prepare data for forecasting
    date_sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()
    date_sales_data.columns = ['ds', 'y']

    m = Prophet()
    m.fit(date_sales_data)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.subheader(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.subheader("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

# Main function
def main():
    st.sidebar.header("Main Menu")
    selected = st.sidebar.selectbox(
        "Options",
        options=["Overview", "Forecast"],
        format_func=lambda x: "Overview" if x == "Overview" else "Forecast"
    )

    if selected == "Overview":
        Overview()
    elif selected == "Forecast":
        forecast()

if __name__ == "__main__":
    main()

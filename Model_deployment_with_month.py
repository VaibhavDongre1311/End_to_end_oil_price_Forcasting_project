from prophet import Prophet
import pandas as pd
import streamlit as st
import pickle
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10


def main():
    # Set the app title
    st.title("Prophet Model Deployment G-3")

    # Apply custom CSS styling
    st.markdown(
        """
        <style>
            /* Change the font family and color of the heading */
            .title-wrapper {
                font-family: 'Arial', sans-serif;
                color: blue;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    df = pd.read_csv("crude-oil-price.csv")
    Newdf = df.copy()
    Newdf['date'] = pd.to_datetime(Newdf['date'])
    dfm = pd.DataFrame(data=Newdf[['date', 'price']])

    # Add month and year input widgets
    year_input = st.number_input("Enter the year:", value=dfm['date'].dt.year.min())
    month_input = st.number_input("Enter the month:", value=1, min_value=1, max_value=12)
    predict_button = st.button("Predict", key="predict_button")

    predictions = pd.DataFrame()  # Initialize an empty DataFrame for predictions

    if predict_button:
        # Prepare the input data for prediction
        input_date = pd.to_datetime(f"{int(year_input)}-{int(month_input)}-1")

        # Ignore the day component and check if the month and year exist in the dataset
        month_year_exists = any((dfm['date'].dt.month == input_date.month) & (dfm['date'].dt.year == input_date.year))

        if month_year_exists:
            # Date is within the training and testing range, return actual price from the dataset
            actual_price = dfm.loc[(dfm['date'].dt.month == input_date.month) & (dfm['date'].dt.year == input_date.year), 'price'].values
            if len(actual_price) > 0:
                actual_price = actual_price[0]
                st.write("Actual Price (USD/BBL):", actual_price)
            else:
                st.write("Actual Price not found for the specified month and year.")
        else:
            # Generate predictions for the input month and year
            with open('prophet.pkl', 'rb') as file:
                model = pickle.load(file)

            # Generate month-wise dates for prediction
            future_dates = pd.date_range(start=dfm['date'].min().replace(day=1), end=input_date.replace(day=1),
                                         freq='MS')

            predictions = model.predict(pd.DataFrame({'ds': future_dates}))

            # Get the forecasted price for the input month and year
            forecasted_price = predictions.loc[(predictions['ds'].dt.month == input_date.month) & (predictions['ds'].dt.year == input_date.year), 'yhat'].values
            if len(forecasted_price) > 0:
                forecasted_price = forecasted_price[0]
                st.write("Forecasted Oil Price (USD/BBL):", forecasted_price)
            else:
                st.write("No forecast available for the specified month and year.")

    # Create DataFrame for hovertool
    hover_df1 = pd.DataFrame({'date': dfm['date'], 'price': dfm['price']})
    source1 = ColumnDataSource(hover_df1)

    # Display the graph
    st.subheader("Oil Price Prediction")

    p = figure(x_axis_type='datetime', title='Oil Price Prediction', width=800, height=400)
    p.line(dfm['date'], dfm['price'], line_color='blue', legend_label='Actual Price')
    p.circle('date', 'price', size=4, fill_color=Category10[3][2], source=ColumnDataSource(hover_df1))

    hover_tool1 = HoverTool(tooltips=[
        ('Date', '@date{%B %Y}'),
        ('Price', '@price{0.00}')
    ], formatters={'@date': 'datetime'}, mode='vline')
    p.add_tools(hover_tool1)

    if not predictions.empty:
        hover_df2 = pd.DataFrame({'date': predictions['ds'], 'price': predictions['yhat']})
        source2 = ColumnDataSource(hover_df2)

        hover_tool2 = HoverTool(tooltips=[
            ('Date', '@date{%B %Y}'),
            ('Forecasted Price', '@price{0.00}')
        ], formatters={'@date': 'datetime'}, mode='vline')

        p.line(predictions['ds'], predictions['yhat'], line_color='green', legend_label='Forecasted Price')
        #p.x_range.start=dfm['date'].max()
        #p.x_range.end=predictions['ds'].max()
        p.circle('date', 'price', size=4, fill_color=Category10[3][1], source=source2)
        p.add_tools(hover_tool2)

    # Style the plot
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price (USD/BBL)'
    p.legend.location = 'top_left'

    # Display the graph
    st.bokeh_chart(p)


if __name__ == '__main__':
    main()

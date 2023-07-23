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

    # Add a date input widget
    date_input = st.date_input("Enter a date:")
    predict_button = st.button("Predict", key="predict_button")

    if predict_button:
        # Prepare the input data for prediction
        input_date = pd.to_datetime(date_input)

        if input_date in dfm['date'].values:
            # Date is within the training and testing range, return exact price from the dataset
            exact_price = dfm.loc[dfm['date'] == input_date, 'price'].values
            if len(exact_price) > 0:
                exact_price = exact_price[0]
                st.write("Exact Price (USD/BBL):", exact_price)
            else:
                st.write("Exact Price not found for the specified date.")
        else:
        # Generate predictions for the input date and beyond
            with open('prophet.pkl', 'rb') as file:
                model = pickle.load(file)
            future_dates = pd.date_range(start=dfm['date'].min(), end=input_date, freq='D')
            predictions = model.predict(pd.DataFrame({'ds': future_dates}))

        # Get the forecasted price for the input date
            forecasted_price = predictions.loc[predictions['ds'] == input_date, 'yhat'].values
            if len(forecasted_price) > 0:
                forecasted_price = forecasted_price[0]
                st.write("Forecasted Oil Price (USD/BBL):", forecasted_price)
            else:
                st.write("No forecast available for the specified date.")





        # Create DataFrame for hovertool
            hover_df = pd.DataFrame({'date': predictions['ds'], 'price': predictions['yhat']})

        # Display the forecasted price
            #st.subheader('Predicted Price')
            #st.write("Forecasted Oil Price (USD/BBL):", predictions.loc[predictions['ds'] == input_date, 'yhat'].values[0])

        # Visualize the graph
    st.subheader("Oil Price Prediction")

    p = figure(x_axis_type='datetime', title='Oil Price Prediction', width=800, height=400)
    p.line(dfm['date'], dfm['price'], line_color='blue', legend_label='Actual Price')



    if predict_button:
        # Generate predictions for the input date and beyond
        with open('prophet.pkl', 'rb') as file:
            loadmodel = pickle.load(file)
        future_dates = pd.date_range(start=dfm['date'].min(), end=input_date, freq='D')
        predictions = loadmodel.predict(pd.DataFrame({'ds': future_dates}))


        # Plot the actual graph
        p.line(dfm['date'],dfm['price'], line_color='blue', legend_label='Actual Price')

        # Update hover_df with forecasted values
        hover_df1 = pd.DataFrame({'date': dfm['date'], 'price': dfm['price']})

        source1 = ColumnDataSource(hover_df1)

        # Add hover tool
        hover_tool1 = HoverTool(tooltips=[
            ('Date', '@date{%F}'),
            ('Price', '@price{0.00}')
        ], formatters={'@date': 'datetime'}, mode='vline')

        p.add_tools(hover_tool1)
        p.circle('date', 'price', size=4, fill_color=Category10[3][0], source=source1)


        # Add forecasted prices
        p.line(predictions['ds'], predictions['yhat'], line_color='green', legend_label='Forecasted Price')

        # Add vertical line for the testing date
        #p.line([dfm['date'].max(), dfm['date'].max()], [dfm['price'].min(), dfm['price'].max()], line_color='red',
         #      line_dash='dashed', legend_label='Testing Date')

        # Create a DataFrame for the hover tool
        hover_df = pd.DataFrame({'date': predictions['ds'], 'price': predictions['yhat']})

        # Create a ColumnDataSource for the hover tool
        source = ColumnDataSource(hover_df)

        # Add hover tool
        hover_tool = HoverTool(tooltips=[
            ('Date', '@date{%F}'),
            ('Price', '@price{0.00}')
        ], formatters={'@date': 'datetime'},mode='vline')

        p.add_tools(hover_tool)
        p.circle('date', 'price', size=4, fill_color=Category10[3][0], source=source)

        # Style the plot
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = 'Price (USD/BBL)'
        p.legend.location = 'top_left'
        p.legend.title = 'Legend'
        #p.legend.location = 'top_left'
        #p.legend.click_policy = 'hide'


        # Display the graph
        st.bokeh_chart(p)


if __name__ == '__main__':
    main()
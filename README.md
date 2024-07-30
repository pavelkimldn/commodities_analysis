# Analyzing Commodity Price Trends and Predictive Modeling

*Project Description:*

This project aims to analyze historical commodity price trends and develop predictive models to forecast future prices, focusing specifically on Gold for the sake of keeping this report concise. However, the code is designed to output graphs and perform analysis for all commodities in the dataset. The analysis includes data cleaning, exploratory data analysis, feature engineering, and predictive modeling using ARIMA.

*Methodology:*

For this project, I began by collecting and cleaning a dataset that spans from 2000 to March 2022, encompassing daily prices for various commodities. The dataset includes columns such as Commodity, Date, Open, High, Low, Close, and Volume. Missing values were handled using forward filling to maintain data integrity.

The next step was exploratory data analysis (EDA), where I visualized price trends over time to identify patterns and anomalies. I calculated 50-day and 200-day moving averages to highlight short-term and long-term trends. These moving averages were crucial for understanding the overall movement of commodity prices and identifying potential buy/sell signals. Additional visualizations included a correlation heatmap, bar charts for average prices, and histograms for price distributions.

Feature engineering involved computing these moving averages, which provided a clearer view of the trends and smoothed out short-term fluctuations. This was followed by predictive modeling using the ARIMA model. I split the data into training and testing sets, applied the ARIMA model, and evaluated its performance using the Mean Squared Error (MSE). I extended the predictions to forecast prices until 2030 to showcase the model's long-term forecasting capabilities.

*Analysis and Implications:*

1. Gold Prices with Moving Averages:

The graph shows a consistent upward trend in Gold prices from 2000 to 2022, with noticeable spikes around 2011 and 2020. The 2011 spike coincides with the global financial crisis, where investors flocked to Gold as a safe-haven asset, pushing its price up significantly. The 2020 spike is likely due to the economic uncertainties brought by the COVID-19 pandemic.
The 50-day and 200-day moving averages provide insights into short-term and long-term price movements. The moving averages smooth out daily price fluctuations, making it easier to identify overall trends. The crossovers between the 50-day and 200-day moving averages often indicate potential buy and sell signals, with the short-term average crossing above the long-term average suggesting a buying opportunity, and vice versa.
![github](https://github.com/pavelkimldn/commodities_analysis/blob/main/image2.png)

2. Gold Price Prediction using ARIMA:

The ARIMA model's predictions for Gold prices demonstrate reasonable accuracy for short-term forecasting. The model was trained on data until 2018 and tested on data from 2018 to 2022. The model's predictions closely follow the actual test data, validating its reliability.
Future price predictions extended until 2030 indicate potential upward and downward trends. However, the model appears to predict a more stable trend post-2022, which may not fully capture the volatility seen in the historical data. This could be due to the ARIMA model's limitations in accounting for external factors influencing commodity prices.
![github](https://github.com/pavelkimldn/commodities_analysis/blob/main/image1.png)

3. Average Closing Prices by Commodity:

The bar chart illustrates the average closing prices for each commodity, with Nickel showing the highest average price. This significant difference could be due to Nickel's high demand in industrial applications, particularly in stainless steel production and battery manufacturing.
Other commodities such as Gold, Palladium, and Brent Oil have relatively lower average prices but still play crucial roles in their respective markets. Natural Gas and US Wheat have the lowest average prices, reflecting their abundant supply and varying demand.
![github](https://github.com/pavelkimldn/commodities_analysis/blob/main/image4.png)

4. Price Distribution of Commodities:

The histograms show the distribution of closing prices for each commodity, providing insights into their price volatility and distribution patterns. The price distribution for Gold indicates a higher frequency of prices between $1,000 and $2,000, reflecting its stability and safe-haven status.
Palladium and Nickel show wider distributions with higher peaks, indicating greater price volatility. Brent Oil and Natural Gas exhibit relatively normal distributions, with prices clustered around their means, while US Wheat displays multiple peaks, suggesting varying demand and supply conditions over time.
![github](https://github.com/pavelkimldn/commodities_analysis/blob/main/image3.png)

*Critical Evaluation:*

However, there are limitations. The ARIMA model, while useful for short-term predictions, may not capture long-term market dynamics and external factors influencing commodity prices. The analysis is based solely on historical price data and does not consider macroeconomic indicators, geopolitical events, or market sentiment, which could impact accuracy.
*How to Run the Project:*
To replicate this project, ensure that Python and the necessary libraries are installed. Download the commodity.csv dataset and place it in the same directory as the script. Running the script will perform the analysis and generate the visualizations.


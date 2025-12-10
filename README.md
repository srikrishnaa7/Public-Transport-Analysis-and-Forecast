# Public Transport Passenger Forecasting Project

## 1. Project Overview
This project forecasts daily public transport passenger journeys for the next **365 days**. It uses a machine learning approach (**Random Forest Regressor**) with a **Recursive Forecasting** strategy to predict future daily traffic, and subsequently aggregates these predictions into Weekly, Monthly, and Hourly views.

The project evolved from testing standard statistical models (Prophet, Holt-Winters) to this Random Forest approach, which achieved the highest accuracy on the test data.

## 2. Methodology

### Data Preprocessing
* **Input File:** `Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv`
* **Target Variable:** `Total` (Sum of Local Route, Light Rail, Peak Service, Rapid Route, School, and Other).
* **Data Cleaning:** The raw dataset contained incomplete entries at the end of September 2024 (e.g., days with only 4-11 passengers). To prevent model skew, all data after **September 20, 2024**, was removed.
* **Imputation:** Missing dates were filled via interpolation.

### Feature Engineering
The model utilizes "Lag Features" to provide historical context (memory) to the Random Forest:
* **Lag_1:** Passenger count from Yesterday.
* **Lag_7:** Passenger count from Last Week (Same Day).
* **Lag_30:** Passenger count from Last Month.
* **Rolling_Mean_7:** The average trend over the previous 7 days.
* **Time Features:** Day of Week, Month, Day of Year.

### Forecasting Strategy
* **Recursive Forecasting:** Since we cannot know the "lag" values for 365 days in the future, the model predicts one day at a time. It predicts *Tomorrow*, adds that prediction to the history, and uses it to calculate the features for the *Day After*.

## 3. Model Performance
The model was evaluated on the last **60 valid days** of the dataset.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **MAPE** | **4.39%** | **Mean Absolute Percentage Error.** The forecast is ~95.6% accurate on average. |
| **MAE** | **1,921** | **Mean Absolute Error.** On a typical day, the prediction is off by ~1,921 passengers. |
| **RMSE** | **3,547** | **Root Mean Squared Error.** Slightly higher than MAE, indicating occasional larger errors (likely on holidays/events). |

## 4. Output Files
The script generates the following CSV files:
1.  `Forecast_Daily.csv`: Day-by-day predictions for the next year.
2.  `Forecast_Weekly.csv`: Aggregated total passengers per week.
3.  `Forecast_Monthly.csv`: Aggregated total passengers per month.
4.  `Forecast_Hourly.csv`: Simulated hourly breakdown based on standard commuter profiles (peaks at 8 AM & 5 PM).

## 5. Installation Requirements
Ensure you have Python installed with the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib

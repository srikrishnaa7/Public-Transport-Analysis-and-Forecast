# Public Transport Passenger Forecasting

## 1. Project Overview
This project forecasts daily public transport passenger journeys for the next 365 days. It utilizes a **Random Forest Regressor** with a **Recursive Forecasting strategy** to handle time-series predictions.

The project evolved from testing standard statistical models (Prophet, Holt-Winters) to a Machine Learning approach (Random Forest with Lag Features), which achieved the highest accuracy.

## 2. Data Source
* **File:** `Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv`
* **Columns:** `Date`, `Local Route`, `Light Rail`, `Peak Service`, `Rapid Route`, `School`, `Other`.
* **Target:** `Total` (Sum of all service types).

## 3. Methodology & Cleaning

### Data Cleaning
* **Issue:** The raw dataset contained incomplete data entries at the end of September 2024 (e.g., extremely low passenger counts like 4 or 11, likely due to partial data upload).
* **Solution:** All data after **September 20, 2024** was removed. This reduced the Mean Absolute Percentage Error (MAPE) from ~4000% to ~4.4%.

### Feature Engineering
The model uses "Lag Features" to give the Random Forest memory of past events:
* **Lag_1:** Passenger count yesterday.
* **Lag_7:** Passenger count on the same day last week.
* **Lag_30:** Passenger count on the same day last month.
* **Rolling_Mean_7:** The average trend over the last week.
* **Time Features:** Day of week, Month, Day of Year.

## 4. Model Performance (Test Set: Last 60 Days)

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **MAE** | **1,921** | On average, the prediction is off by ~1,921 passengers per day. |
| **MAPE** | **4.39%** | The forecast is ~95.6% accurate on tested data. |

## 5. Installation Requirements

Requires Python 3.x and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib

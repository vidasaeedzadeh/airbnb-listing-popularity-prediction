# airbnb-listing-popularity-prediction
Predicting Airbnb listing popularity using NYC Airbnb Open Data (2019)


# NYC Airbnb Listing Popularity Prediction

## Project Overview

This project builds a regression model to predict the popularity of Airbnb listings in New York City using the 2019 NYC Airbnb Open Data dataset. Popularity is approximated using the variable `reviews_per_month`.

Predicting listing popularity could help Airbnb estimate demand for new listings before they are posted and help hosts optimize listing characteristics.

## Dataset

Source: NYC Airbnb Open Data (2019)  
https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

The dataset contains information about Airbnb listings including:

- location
- room type
- price
- number of reviews
- availability
- host information

## Objective

The goal of this project is to:

- Explore factors that influence Airbnb listing popularity
- Build regression models to predict `reviews_per_month`
- Evaluate model performance and interpret important features


---

## Pipeline

The project follows a modular workflow:

1. Data validation & overview  
2. Data cleaning  
3. Exploratory Data Analysis (EDA)  
4. Feature engineering  
5. Preprocessing  
6. Model training & evaluation  
7. Model interpretation  

---

## Models Used

- Linear Regression  
- Ridge Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

---

## Results (Test Performance)

| Model            | R²   | RMSE | NRMSE |
|------------------|------|------|------|
| Linear Regression | 0.23 | 1.38 | 0.103 |
| Ridge Regression  | 0.23 | 1.38 | 0.103 |
| Decision Tree     | 0.56 | 1.04 | 0.078 |
| Random Forest     | **0.61** | **0.99** | **0.074** |
| XGBoost           | 0.60 | 0.99 | 0.074 |

**Random Forest selected as final model**

---

## Key Insights

- Listing popularity is driven more by **behavioral signals** than static attributes  
- **Recent activity (`days_since_last_review`)** is the strongest predictor  
- **Minimum nights and availability** strongly impact engagement  
- Nonlinear models significantly outperform linear models  

---

## How to Run

```bash
# Step 1: install dependencies
pip install -r requirements.txt

# Step 2: run pipeline
python -m scripts.run_pipeline





from src.data_wrangling.data_loader import load_raw_data
from src.data_wrangling.data_cleaning import clean_airbnb_data
from src.data_wrangling.feature_engineering import run_feature_engineering

def main():

    df = load_raw_data()

    df_clean, _ = clean_airbnb_data(df, columns_to_drop=[...])

    df_fe = run_feature_engineering(df_clean)

    df_fe.to_csv("data/processed/airbnb_features.csv", index=False)

if __name__ == "__main__":
    main()

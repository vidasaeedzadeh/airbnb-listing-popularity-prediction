from scripts.run_data_overview import main as run_data_overview
from scripts.run_data_cleaning import main as run_data_cleaning
from scripts.run_eda import main as run_eda
from scripts.run_feature_engineering import main as run_feature_engineering
from scripts.run_preprocessing import main as run_preprocessing
from scripts.run_training import main as run_training
from scripts.run_training import main as run_feature_importance


def main() -> None:
    
    print("Step 1: Data Overview")
    run_data_overview()

    print("Step 2: Data Cleaning")
    run_data_cleaning()

    print("Step 3: Exploratory Data Analysis")
    run_eda()

    print("Step 4: Feature Engineering")
    run_feature_engineering()

    print("Step 5: Preprocessing")
    run_preprocessing()

    print("Step 6: Model Training")
    run_training()

    print("Step 7: feature importance")
    run_feature_importance()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
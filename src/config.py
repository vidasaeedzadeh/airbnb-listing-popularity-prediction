from pathlib import Path

################Projest Root and Data Path###################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "AB_NYC_2019.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "airbnb_cleaned.csv"


################Results and Figures###################

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

################Data and Modeling Configuration###################

TARGET_COLUMN = "reviews_per_month"

X_TRAIN_FILE = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_FILE = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_FILE = PROCESSED_DATA_DIR / "y_test.csv"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"

RANDOM_SEED = 123
TEST_SIZE = 0.2

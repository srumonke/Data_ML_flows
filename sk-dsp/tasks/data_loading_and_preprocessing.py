import pandas as pd
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
df = pd.read_csv(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\data\AmesHousing.csv")
logger.info("Dataset loaded for basic statistics.")
logger.info(f"DataFrame head:\n{df.head()}")

# Summary statistics
logger.info("Summary Statistics:")
logger.info(f"\n{df.describe(include='all')}")

# Data type information
logger.info("Data Types:")
logger.info(f"\n{df.dtypes}")

# Check and impute missing values
logger.info("Checking for missing values...")
missing_summary = df.isnull().sum()
logger.info(f"\nMissing values in each column:\n{missing_summary[missing_summary > 0]}")

# Impute missing values for numeric columns with mean and categorical columns with mode
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Applying imputers
df[df.select_dtypes(include=['float64', 'int64']).columns] = numeric_imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))
df[df.select_dtypes(include=['object']).columns] = categorical_imputer.fit_transform(df.select_dtypes(include=['object']))

# Verifying that missing values have been imputed
logger.info("Missing values imputed. Checking again:")
missing_summary_after = df.isnull().sum()
logger.info(f"\nRemaining missing values in each column:\n{missing_summary_after[missing_summary_after > 0]}")

# Normalization of numeric features like 'Lot Area' and 'SalePrice'
scaler = MinMaxScaler()
for column in ['Lot Area', 'SalePrice']:
    if column in df.columns:
        df[[column]] = scaler.fit_transform(df[[column]])
        logger.info(f"{column} has been normalized using Min-Max scaling.")

logger.info("Data preprocessing completed.")
logger.info(f"\nProcessed DataFrame head:\n{df.head()}")

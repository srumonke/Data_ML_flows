import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\data\AmesHousing.csv")
logger.info("Dataset loaded for feature importance analysis.")

# Select numeric features and target
X = df.select_dtypes(include=[np.number]).drop(columns=['SalePrice'], errors='ignore')
y = df['SalePrice']

# Impute missing values in X with the median
X = X.fillna(X.median())
logger.info("Missing values imputed with median.")

# Calculate mutual information for numeric features
mutual_info = mutual_info_regression(X, y)
feature_importance = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.ylabel("Mutual Information Score")
plt.show()

logger.info("Feature importance analysis completed.")

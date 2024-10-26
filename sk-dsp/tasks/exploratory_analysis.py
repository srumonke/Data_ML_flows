import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
df = pd.read_csv(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\data\AmesHousing.csv")
logger.info("Dataset loaded for EDA.")

# Only select numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()


# Identify Highly Correlated Features
correlation_threshold = 0.7
high_corr_features = correlation_matrix.index[abs(correlation_matrix["SalePrice"]) > correlation_threshold]
logger.info(f"Highly correlated features with 'SalePrice': {high_corr_features.tolist()}")

# Feature Importance using Mutual Information
logger.info("Calculating feature importance using mutual information.")
X = df.select_dtypes(include=[np.number]).drop(columns=['SalePrice'])
y = df['SalePrice']
mutual_info = mutual_info_regression(X, y)
feature_importance = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)
logger.info(f"Top 5 important features:\n{feature_importance.head()}")

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.ylabel("Mutual Information Score")
plt.show()

# Binning (Discretization) - Example on 'Lot Area'
logger.info("Binning 'Lot Area' into categories.")
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['LotArea_Binned'] = binner.fit_transform(df[['Lot Area']])
logger.info("Binned 'Lot Area' added as 'LotArea_Binned'.")

# Encoding Categorical Variables
logger.info("Encoding categorical variables.")
encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = encoder.fit_transform(df[column].astype(str))
logger.info("Categorical variables encoded.")

# Univariate Visualization
logger.info("Plotting univariate visualizations.")
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True)
plt.title("SalePrice Distribution")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Lot Area'], kde=True)
plt.title("Lot Area Distribution")
plt.show()

# Bivariate Visualization
logger.info("Plotting bivariate visualizations.")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Lot Area', y='SalePrice', data=df)
plt.title("SalePrice vs Lot Area")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='LotArea_Binned', y='SalePrice', data=df)
plt.title("SalePrice Distribution by Binned Lot Area")
plt.show()

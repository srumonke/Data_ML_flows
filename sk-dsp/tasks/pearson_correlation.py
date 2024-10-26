import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\data\AmesHousing.csv")
logger.info("Dataset loaded for Pearson correlation analysis.")

# Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])
logger.info("Filtered numeric columns for correlation analysis.")

# Calculate Pearson correlation matrix
correlation_matrix = numeric_df.corr(method='pearson')
logger.info("Pearson correlation matrix calculated.")

# Limit features to only those with high correlation with SalePrice (threshold can be adjusted)
threshold = 0.3
high_corr_features = correlation_matrix[correlation_matrix['SalePrice'].abs() > threshold]['SalePrice'].index.tolist()

# Plot heatmap of the correlation matrix for selected features
plt.figure(figsize=(12, 8))  # Increased size for more details
sns.heatmap(correlation_matrix.loc[high_corr_features, high_corr_features],
             annot=True, cmap='coolwarm', fmt=".2f", cbar=True,
             square=True, linewidths=0.5, linecolor='grey',
             annot_kws={"size": 8})  # Adjust the size of the annotations
plt.title("Pearson Correlation Matrix (SalePrice > 0.3)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

logger.info("Pearson correlation heatmap displayed.")

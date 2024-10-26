import pandas as pd
import logging
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from sklearn.impute import SimpleImputer

# Set default theme
sns.set_theme()

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
df = pd.read_csv(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\data\AmesHousing.csv")
logger.info("Dataset loaded for Visualization.")

# Impute missing values for numeric columns
for c in df.select_dtypes(include=['number']).columns:
    df[c] = df[c].fillna(df[c].mean())

# --------------------------------------- UNIVARIATE ANALYSIS ------------------------------

# 1.1 Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Lot Area'])  
plt.title('Box Plot of Lot Area')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Box Plot Lot Area.png")
logger.info("Box Plot of Lot Area saved as 'Box Plot Lot Area.png'")

# Clear the figure and close the plot
plt.clf()

# 1.2 Strip Plot
plt.figure(figsize=(8, 6))
sns.stripplot(y=df['Lot Area'])
plt.title('Strip Plot of Lot Area')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Strip Plot Lot Area.png")
logger.info("Strip Plot of Lot Area saved as 'Strip Plot Lot Area.png'")

# Clear the figure and close the plot
plt.clf()

# 1.3 Swarm Plot
plt.figure(figsize=(8, 6))
sns.swarmplot(x=df['Garage Area'], size=1)
plt.title('Swarm Plot of Garage Area')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Swarm Plot Garage Area.png")
logger.info("Swarm Plot of Garage Area saved as 'Swarm Plot Garage Area.png'")

# Clear the figure and close the plot
plt.clf()

# 1.4 Histogram
plt.figure(figsize=(8, 6))
plt.hist(df['Lot Area'], bins=30, color='blue', edgecolor='black')  
plt.title('Histogram of Lot Area')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Histogram Plot Lot Area.png")
logger.info("Histogram of Lot Area saved as 'Histogram Plot Lot Area.png'")

# Clear the figure and close the plot
plt.clf()

# 1.5 Seaborn displot
plt.figure(figsize=(8, 6))
sns.histplot(df['Lot Area'], kde=False, color='blue', bins=30)
plt.title('Dis Plot of Lot Area with 30 bins')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Dis Plot Lot Area.png")
logger.info("Dis Plot of Lot Area with 30 bins saved as 'Dis Plot Lot Area.png'")

# Clear the figure and close the plot
plt.clf()

# 1.6 Count Plot
plt.figure(figsize=(8, 6))
sns.countplot(x=df['Central Air'])
plt.title('Count Plot of Central Air (Categorical)')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Count Plot Central Air.png")
logger.info("Count Plot of Central Air (Categorical) saved as 'Count Plot Central Air.png'")

# Clear the figure and close the plot
plt.clf()

# --------------------------------------- BIVARIATE ANALYSIS -----------------------------

# 2.1 Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Land Slope', y='Lot Area', data=df)
plt.title('Box Plot of Land Slope vs Lot Area')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Box Plot Land Slope vs Lot Area.png")
logger.info("Box Plot of Land Slope vs Lot Area saved as 'Box Plot Land Slope vs Lot Area.png'")

# Clear the figure and close the plot
plt.clf()

# 2.2 Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Lot Frontage'], y=df['Lot Area'])
plt.title('Scatter Plot of Lot Frontage vs Lot Area')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Scatter Plot Lot Frontage vs Lot Area.png")
logger.info("Scatter Plot of Lot Frontage vs Lot Area saved as 'Scatter Plot Lot Frontage vs Lot Area.png'")

# Clear the figure and close the plot
plt.clf()

# 2.3 Scatter Plot with Hue
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Lot Frontage'], y=df['Lot Area'], hue=df['Garage Type'])
plt.title('Scatter Plot of Lot Frontage vs Lot Area vs Garage Type (hue value)')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Scatter Plot Lot Frontage vs Lot Area vs Garage Type.png")
logger.info("Scatter Plot of Lot Frontage vs Lot Area vs Garage Type (hue value) saved as 'Scatter Plot Lot Frontage vs Lot Area vs Garage Type.png'")

# Clear the figure and close the plot
plt.clf()

# 2.4 FacetGrid for Central Air vs SalePrice
g = sns.FacetGrid(df, col="Central Air", height=6.5, aspect=.85)
g.map(sns.histplot, "SalePrice")
plt.title('Facet Grid of Central Air vs SalePrice')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\Facet Grid Central Air vs SalePrice.png")
logger.info("Facet Grid of Central Air vs SalePrice saved as 'Facet Grid Central Air vs SalePrice.png'")

# Clear the figure and close the plot
plt.clf()

# 2.5 lmplot - SalePrice vs Lot Area vs Central Air
sns.lmplot(data=df, x="SalePrice", y="Lot Area", hue="Central Air", height=6)
plt.title('lmplot of SalePrice vs Lot Area vs Central Air (hue)')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\lmplot SalePrice vs Lot Area vs Central Air.png")
logger.info("lmplot of SalePrice vs Lot Area vs Central Air (hue) saved as 'lmplot SalePrice vs Lot Area vs Central Air.png'")

# Clear the figure and close the plot
plt.clf()

# 2.6 Lot Area vs Lot Frontage vs Central Air
sns.lmplot(data=df, x="Lot Area", y="Lot Frontage", hue="Central Air", height=6)
plt.title('lmplot of Lot Area vs Lot Frontage vs Central Air (hue)')
plt.tight_layout()

# Save the plot as a file
plt.savefig(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\output\lmplot Lot Area vs Lot Frontage vs Central Air.png")
logger.info("lmplot of Lot Area vs Lot Frontage vs Central Air (hue) saved as 'lmplot Lot Area vs Lot Frontage vs Central Air.png'")

# Clear the figure and close the plot
plt.clf()

logger.info("All plots have been successfully generated and saved.")

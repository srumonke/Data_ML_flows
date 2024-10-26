import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\data\AmesHousing.csv")
logger.info("Dataset loaded for binning.")

# Binning 'Lot Area' into quantile-based categories
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['LotArea_Binned'] = binner.fit_transform(df[['Lot Area']])
logger.info("Binning completed. 'LotArea_Binned' column added.")

# Display binned data for verification
logger.info(f"Binned 'Lot Area':\n{df[['Lot Area', 'LotArea_Binned']].head()}")

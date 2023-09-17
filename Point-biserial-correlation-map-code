import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

df = pd.read_csv('smoking_driking_dataset_Ver01.csv')

# Below part is to calculate point-biserial correlation for each continuous variable with 'Drinking' (DRK_YN)
correlations = {}
for column in df.columns:
    if column != 'DRK_YN':  
        correlation, p_value = pointbiserialr(df['DRK_YN'], df[column])
        correlations[column] = (correlation, p_value)

# To sort variables by absolute correlation in descending order
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)

# to get variable names and correlation coefficients for visualization
variable_names = [item[0] for item in sorted_correlations]
correlation_values = [item[1][0] for item in sorted_correlations]

# Creating a bar chart to visualize correlations
plt.figure(figsize=(10, 6))
plt.barh(variable_names, correlation_values)
plt.xlabel('Point-Biserial Correlation Coefficient')
plt.ylabel('Variable')
plt.title('Point-Biserial Correlation with Drinking')
plt.show()

import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv('smoking_driking_dataset_Ver01.csv')

# Below, I've initialized the Isolation Forest model
clf = IsolationForest(contamination=0.02, random_state=42)

# This is to fit the Isolation Forest model to the data and predict outliers
outliers = clf.fit_predict(df)

# to create a mask to identify outliers (-1) in the DataFrame
mask = outliers == -1

# To remove the outliers from the DataFrame
df_cleaned = df[~mask]

df_cleaned['BMI'] = df_cleaned['weight'] / (df_cleaned['height'] * df_cleaned['height'])
print(df_cleaned[['weight', 'height', 'BMI']].head())

df_cleaned.to_csv('smoking_driking_dataset_Ver01.csv', index=False)

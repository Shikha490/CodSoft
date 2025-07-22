from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from scipy.stats import zscore


# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.isnull().sum())

# Optionally, handle missing values if any
df = df.dropna()  # or use imputation if needed
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows.")
print(df.dtypes)

# Convert object columns if needed
# e.g., df['sepal length (cm)'] = df['sepal length (cm)'].astype(float)
# Basic sanity checks for value ranges
assert df['sepal length (cm)'].between(0, 10).all(), "Out-of-range sepal length"
assert df['sepal width (cm)'].between(0, 10).all(), "Out-of-range sepal width"
assert df['petal length (cm)'].between(0, 10).all(), "Out-of-range petal length"
assert df['petal width (cm)'].between(0, 10).all(), "Out-of-range petal width"
print(df['target'].unique())  # Should be [0, 1, 2]

assert set(df['target'].unique()) == {0, 1, 2}, "Unexpected class labels"
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
def validate_iris_data(df):
    assert df.isnull().sum().sum() == 0, "Missing values found"
    assert df.duplicated().sum() == 0, "Duplicates found"
    assert df['sepal length (cm)'].between(0, 10).all()
    assert df['sepal width (cm)'].between(0, 10).all()
    assert df['petal length (cm)'].between(0, 10).all()
    assert df['petal width (cm)'].between(0, 10).all()
    assert set(df['target'].unique()) == {0, 1, 2}
    print("✅ Iris dataset passed integrity and consistency checks.")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_no_outliers[iris.feature_names])

#Min-Max Normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df_no_outliers[iris.feature_names])

# Data Transformation
#Log or Square Root Transformation
df_log = df_no_outliers.copy()
for col in iris.feature_names:
    df_log[col] = np.log1p(df_log[col])  # log(1 + x) to handle 0s

# Handeling missing values
np.random.seed(0)
df.loc[df.sample(frac=0.1).index, 'sepal length (cm)'] = np.nan

# Check for missing values
print(df.isnull().sum())

# Handling Missing Values (Impute with mean)
imputer = SimpleImputer(strategy='mean')
df[df.columns[:-1]] = imputer.fit_transform(df[df.columns[:-1]])

print(df.isnull().sum())

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
#identifying patterns and trends 
print(df.groupby('species').mean())
import seaborn as sns
sns.pairplot(df, hue='species', diag_kind='kde')
import matplotlib.pyplot as plt
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
for col in iris.feature_names:
    sns.boxplot(x='species', y=col, data=df)
    plt.title(col)
    plt.show()

#--------Identifying Anomolies (Outliers)--------
z_scores = df[iris.feature_names].apply(zscore)
outliers = (abs(z_scores) > 3).any(axis=1)
df[outliers]

#Visualize Anomolies
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='anomaly', palette={1:'blue', -1:'red'})
plt.title('Anomalies in Iris Dataset')
plt.show()




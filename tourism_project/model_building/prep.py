
# Import libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset from Hugging Face
DATASET_PATH = "hf://datasets/vidyaa2026/tourism_project/tourism.csv"
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Dataset Shape:", df.shape)

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------

# Remove unnecessary column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Check and remove duplicates
duplicates = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)

df.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing Values Summary")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("\nNo missing values found in the dataset.")

# Rectify categorical values
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

print("\nDataset Shape after Cleaning:", df.shape)

# --------------------------------------------------
# Basic Anomaly Checks
# --------------------------------------------------

print(
    "\nUnrealistic Age Records:",
    df[(df['Age'] < 18) | (df['Age'] > 100)].shape[0]
)

print(
    "Invalid PreferredPropertyStar:",
    df[~df['PreferredPropertyStar'].isin([1, 2, 3, 4, 5])].shape[0]
)

# --------------------------------------------------
# Separate Features and Target
# --------------------------------------------------

X = df.drop(columns=['ProdTaken', 'CustomerID'])
y = df['ProdTaken']

# Separate numerical and categorical variables
num_cols = X.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

cat_cols = X.select_dtypes(
    include=['object', 'category']
).columns.tolist()

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)

# Check target distribution
print("\nTarget Distribution")
print(y.value_counts())

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nXtrain Shape:", Xtrain.shape)
print("Xtest Shape :", Xtest.shape)

# --------------------------------------------------
# Save datasets locally
# --------------------------------------------------

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("\nTrain and test datasets saved successfully.")

# --------------------------------------------------
# Upload datasets to Hugging Face
# --------------------------------------------------

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="vidyaa2026/tourism_project",
        repo_type="dataset"
    )

print("\nUploaded Files:")
print(files)

print("\nTrain and test datasets uploaded successfully.")

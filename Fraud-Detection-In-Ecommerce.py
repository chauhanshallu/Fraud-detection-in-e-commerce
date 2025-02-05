import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
file_path = "E-commerce-(Responses).xlsx"
df = pd.read_excel(file_path)

# Preview the dataset
print(df.head())

"""##Clean and Preprocess Data"""

# Renaming columns for clarity
df.columns = [
    "timestamp", "gender", "age_group", "purchase_frequency", "fraud_experience",
    "fraud_concern", "ai_familiarity", "fraud_detection_effectiveness", "platform_trust",
    "shop_likelihood", "security_importance", "effectiveness_rating",
    "fraud_encounters", "robust_systems", "ai_vs_traditional",
    "fraud_priority", "satisfaction_mechanisms"
]

# Convert categorical columns to numerical where necessary
categorical_cols = ["gender", "age_group", "purchase_frequency", "fraud_experience",
                    "fraud_concern", "ai_familiarity", "fraud_detection_effectiveness",
                    "platform_trust", "shop_likelihood", "security_importance",
                    "ai_vs_traditional", "fraud_priority", "satisfaction_mechanisms"]

df[categorical_cols] = df[categorical_cols].apply(lambda col: pd.factorize(col)[0])

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Preview the processed dataset
print(df.info())



"""##Exploratory Data Analysis (EDA)"""

# Gender Distribution
sns.countplot(x="gender", data=df)
plt.title("Gender Distribution")
plt.show()

# Fraud Concern by Age Group
sns.boxplot(x="age_group", y="fraud_concern", data=df)
plt.title("Fraud Concern by Age Group")
plt.show()

# Frequency of Fraudulent Experiences
sns.countplot(x="fraud_experience", data=df)
plt.title("Fraudulent Experiences")
plt.show()

import numpy as np

# Convert any categorical data to numeric if necessary
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Numeric Variables')
plt.tight_layout()
plt.show()

#Proportion of Gender Categories
gender_counts = df['gender'].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Proportion of Gender Categories')
plt.tight_layout()
plt.show()

# Fraud Concern vs. Effectiveness Rating
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='fraud_concern', y='effectiveness_rating', palette='coolwarm')
plt.title('Fraud Concern vs. Effectiveness Rating')
plt.xlabel('Fraud Concern Level')
plt.ylabel('Effectiveness Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Platform Trust vs. Security Importance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='platform_trust', y='security_importance', hue='fraud_priority', palette='coolwarm')
plt.title('Platform Trust vs. Security Importance')
plt.xlabel('Platform Trust')
plt.ylabel('Security Importance')
plt.tight_layout()
plt.show()

# Fraud Encounters
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='fraud_encounters', palette='viridis')
plt.title('Distribution of Fraud Encounters')
plt.xlabel('Fraud Encounters')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# AI Familiarity vs. Shop Likelihood
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='ai_familiarity', y='shop_likelihood', ci=None, palette='mako')
plt.title('AI Familiarity vs. Shop Likelihood')
plt.xlabel('AI Familiarity Level')
plt.ylabel('Shop Likelihood')
plt.tight_layout()
plt.show()



"""##Model Evaluation"""

# Define features and target
X = df.drop(columns=["fraud_experience", "timestamp"])  # Features
y = df["fraud_experience"]  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.info())

print(X_train['fraud_encounters'].unique())
print(X_train['robust_systems'].unique())

def convert_fraud_encounters(value):
    if isinstance(value, str) and '–' in value:
        parts = value.split('–')
        return (int(parts[0]) + int(parts[1])) / 2  # Midpoint of the range
    return float(value)  # If it's already numeric, return as is

def convert_robust_systems(value):
    if isinstance(value, str) and '–' in value:
        parts = value.split('–')
        return (int(parts[0]) + int(parts[1].replace('%', ''))) / 2  # Midpoint of the range
    return float(value)  # If it's already numeric, return as is

# Apply the conversion to 'fraud_encounters' and 'robust_systems'
X_train['fraud_encounters'] = X_train['fraud_encounters'].apply(convert_fraud_encounters)
X_test['fraud_encounters'] = X_test['fraud_encounters'].apply(convert_fraud_encounters)

X_train['robust_systems'] = X_train['robust_systems'].apply(convert_robust_systems)
X_test['robust_systems'] = X_test['robust_systems'].apply(convert_robust_systems)

# Check if the columns are now numeric
print(X_train['fraud_encounters'].unique())
print(X_train['robust_systems'].unique())

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
print("Random Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
print("Logistic Regression Results:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))



"""##Statistical tests"""

#T-Test: Effectiveness Rating by Fraud Experience
# Split data into groups
group1 = df[df["fraud_experience"] == 0]["effectiveness_rating"]
group2 = df[df["fraud_experience"] == 1]["effectiveness_rating"]

# Perform T-Test
t_stat, p_value = ttest_ind(group1, group2)
print(f"T-Test:\nT-statistic = {t_stat}, p-value = {p_value}")

#Chi-Square Test: Fraud Experience vs Gender
# Create a contingency table
contingency_table = pd.crosstab(df["fraud_experience"], df["gender"])

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Test:\nChi2 = {chi2}, p-value = {p}")


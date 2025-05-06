import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Constants
target = 'carbon_footprint'
id_column = 'ID'

# Clean specific columns
def clean_house_area(df):
    df['house_area_sqft'] = df['house_area_sqft'].astype(str).str.extract(r'(\d+)')
    df['house_area_sqft'] = pd.to_numeric(df['house_area_sqft'], errors='coerce')
    df['house_area_sqft'].fillna(df['house_area_sqft'].mean(), inplace=True)
    return df
def clean_household(df):
    df['household_size'] = df['household_size']
def clean_heating_type(df):
    valid_types = ['gas', 'electric', 'none']
    df['heating_type'] = df['heating_type'].apply(lambda x: x if x in valid_types else 'none')
    return df

# Preprocessing function
def preprocess_data(df, is_train=True):
    df = df.copy()
    df = clean_house_area(df)
    df = clean_heating_type(df)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if target in numeric_cols: numeric_cols.remove(target)
    if id_column in numeric_cols: numeric_cols.remove(id_column)
    if id_column in categorical_cols: categorical_cols.remove(id_column)

    for col in numeric_cols:
        if col not in ['public_transport_usage_per_week']:
            df.loc[df[col] < 0, col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df, numeric_cols, categorical_cols

data,numeric_cols,categorical_cols  = preprocess_data(train_df,is_train=True)

# Feature engineering
def engineer_features(df, numeric_cols):
    df = df.copy()
    df['household_size'] = df['household_size'].str.extract('(\d+)')
    df['household_size'] =df['household_size'].astype(float)
    df['household_size'].replace(0, np.nan, inplace=True)
    df['household_size'].fillna(df['household_size'].mean(), inplace=True)

    # Total energy
    df['natural_gas_kWh_per_month'] = df['natural_gas_therms_per_month'] * 29.3
    numeric_cols.append('natural_gas_kWh_per_month')

    df['total_energy'] = df['electricity_kwh_per_month'] + df['natural_gas_kWh_per_month']
    numeric_cols.append('total_energy')

    # Energy per person and per square foot
    df['energy_per_person'] = df['total_energy'] / (df['household_size'] + 1e-5)
    numeric_cols.append('energy_per_person')

    df['energy_per_sqft'] = df['total_energy'] / (df['house_area_sqft'] + 1)
    numeric_cols.append('energy_per_sqft')

    # Water usage and vehicle miles per person
    df['water_usage_per_person'] = df['water_usage_liters_per_day'] / (df['household_size'] + 1e-5)
    numeric_cols.append('water_usage_per_person')

    df['vehicle_miles_per_person'] = df['vehicle_miles_per_month'] / (df['household_size'] + 1e-5)
    numeric_cols.append('vehicle_miles_per_person')

    # Meat consumption per person
    df['meat_consumption_kg_per_person'] = df['meat_consumption_kg_per_week'] / (df['household_size'] + 1e-5)
    numeric_cols.append('meat_consumption_kg_per_person')

    # Sustainability score
    sustainability_factors = [
        'recycles_regularly', 'composts_organic_waste', 'uses_solar_panels',
        'energy_efficient_appliances', 'smart_thermostat_installed']
    df['sustainability_score'] = df[sustainability_factors].fillna(0).astype(int).sum(axis=1)
    numeric_cols.append('sustainability_score')

    # Diet type mapping
    diet_impact = {'vegan': 1, 'vegetarian': 2, 'omnivore': 3}
    df['diet_impact'] = df['diet_type'].map(diet_impact).fillna(2)
    numeric_cols.append('diet_impact')

    # Carbon intensity
    df['carbon_intensity'] = df['total_energy'] / (df['sustainability_score'] + 1)
    numeric_cols.append('carbon_intensity')

    # Income per square foot (if income data available)
    if 'monthly_income' in df.columns:
        df['income_per_sqft'] = df['monthly_income'] / (df['house_area_sqft'] + 1)
        numeric_cols.append('income_per_sqft')

    return df, numeric_cols



train_clean, numeric_features, categorical_features = preprocess_data(train_df)
test_clean, _, _ = preprocess_data(test_df, is_train=False)
train_featured, numeric_features = engineer_features(train_clean, numeric_features)
test_featured, _ = engineer_features(test_clean, numeric_features)
print(train_featured)
X = train_featured.drop([target, id_column], axis=1)
y = train_featured[target]
test_X = test_featured.drop([id_column], axis=1)

# Pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Model pipeline
estimators = [
    ('ridge', Ridge(alpha=1.0)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1))
]

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', StackingRegressor(
        estimators=estimators,
        final_estimator=xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE),
        cv=3))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print("Training model...")
model.fit(X_train, y_train)
val_predictions = model.predict(X_val)

print("\nValidation Performance:")
print(f"R²: {r2_score(y_val, val_predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, val_predictions)):.4f}")
print(f"MAE: {mean_absolute_error(y_val, val_predictions):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_val, val_predictions):.4f}")

# CV
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV R²:", np.mean(cv_scores))

# Retrain and predict
print("\nRetraining on full data and predicting test...")
model.fit(X, y)
test_predictions = model.predict(test_X)

submission = pd.DataFrame({
    id_column: test_df[id_column],
    target: test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission saved.")

# Save model
joblib.dump(model, 'final_model.pkl')

# Plot distribution
plt.figure(figsize=(10, 5))
sns.histplot(test_predictions, kde=True)
plt.title('Prediction Distribution')
plt.savefig('prediction_distribution.png')
print("Prediction plot saved.")

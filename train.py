import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv(r"C:\Users\vishn\Downloads\loan_approval_dataset.csv")

# Clean column names (removes hidden spaces)
df.columns = df.columns.str.strip()

# User's original cleaning steps
df.drop_duplicates(inplace=True)   
df.dropna(inplace=True) 
df = df.fillna(df.mean(numeric_only=True))

df = df.drop(['loan_id'], axis=1)

# Map categorical variables (using your exact original keys)
df['education'] = df['education'].str.strip().map({'Graduate': 1, 'NotGraduate': 0})
df['self_employed'] = df['self_employed'].str.strip().map({'Yes': 1, 'No': 0})
df['loan_status'] = df['loan_status'].str.strip().map({'Approved': 1, 'Rejected': 0})

# Safety check: drop any rows that might have become NaN during mapping
df.dropna(inplace=True)

# Split features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status'] 

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train) 

# Export for Flask app
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(list(X.columns), 'features.joblib')

print("Training complete! Model and scaler exported successfully.")
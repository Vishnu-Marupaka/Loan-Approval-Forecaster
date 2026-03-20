import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split    
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,f1_score

df=pd.read_csv(r"C:\Users\vishn\Downloads\loan_approval_dataset.csv")

df.drop_duplicates(inplace=True)   
df.dropna(inplace=True) 
df = df.fillna(df.mean(numeric_only=True))
print("Missing values in each column:", df.isnull().sum())
df = df.drop(['loan_id'], axis=1)
df['education'] = df['education'].map({'Graduate': 1, 'NotGraduate': 0})
df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
print(df.head())

x=df.drop('loan_status', axis=1)
y=df['loan_status'] 

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)  
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=42, stratify=y)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train) 
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]  
acc = accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred,average='weighted',zero_division=0)
f1 = f1_score(y_test, y_pred,average='weighted',zero_division=0)
print(f"Accuracy: {acc:.4f}")   
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_prob, alpha=0.5)  
plt.xlabel('True Labels')
plt.ylabel('Predicted Probabilities')   
plt.title('True Labels vs Predicted Probabilities')
plt.grid()
plt.show()

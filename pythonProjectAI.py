import pandas as pd

#Choose db
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

print(data)

#Loading and exploring data
X = data.drop('Outcome', axis=1)

y = data['Outcome']

#imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fit
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

#ModelsList
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN Classifier': KNeighborsClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier()
}

#PredictionsList
predictions = {}
for nameM, model in models.items():
    model.fit(X_train, y_train)
    predictions[nameM] = model.predict(X_test)

#Metrics

from sklearn.metrics import accuracy_score, classification_report

def print_metrics(y_test, y_pred, modelN):
    print(modelN, ':')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print("Confusen Metrics:")
    print(confusion_matrix(y_test, y_pred))
    print("classification Report")
    print(classification_report(y_test, y_pred))


for name, preds in predictions.items():
    print_metrics(y_test, preds, name)

#Plots
import matplotlib.pyplot as plt
from sklearn import tree

#tree

# plt.figure(figsize=(20, 10))
# tree.plot_tree(models['Decision Tree Classifier'],
#                feature_names=X.columns,
#                class_names=["You dont have Diagnosed with diabetes", "you Diagnosed with diabetes"],
#                filled=True)
# plt.title("Decision Tree Structure")
# plt.show()

model = models['Decision Tree Classifier']
new_person = pd.DataFrame([{
    'Pregnancies': 4,
    'Glucose': 190,
    'BloodPressure': 92,
    'SkinThickness': 0,
    'Insulin': 0,
    'BMI': 35.5,
    'DiabetesPedigreeFunction': 0.278,
    'Age': 42
}])
new_person = scalar.transform(new_person)
print(model.predict(new_person))

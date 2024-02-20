import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("/content/HepatitisCdata.csv")


data.dropna(subset=["Age", "ALB", "ALP", "ALT", "AST", "BIL", "Category"], inplace=True)

X = data[["Age", "ALB", "ALP", "ALT", "AST", "BIL"]]
y = data["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_data = [[32, 38.5, 52.5, 7.7, 22.1, 7.5]]
prediction = model.predict(new_data)

if prediction[0] == 0:
    print("Blood Donor")
elif prediction[0] == 00:
    print("Suspect Blood Donor")
elif prediction[0] == 1:
    print("Hepatitis")
elif prediction[0] == 2:
    print("Fibrosis")
else:
    print("Cirrhosis")

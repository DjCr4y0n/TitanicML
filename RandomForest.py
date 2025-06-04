import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')

df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = df_train[['Pclass', 'Sex', 'Age','SibSp','Parch', 'Fare', 'Embarked']]  # features
y = df_train['Survived']

X.loc[:, 'Embarked'] = X['Embarked'].fillna(0)
X.loc[:, 'Embarked'] = X['Embarked'].replace([np.inf, -np.inf], 0)
X = pd.get_dummies(X, columns=['Embarked'], dtype=int)
X.loc[:, 'Age'] = X['Age'].fillna(X['Age'].median())
X.loc[:, 'Fare'] = X['Fare'].fillna(X['Fare'].median())

print(X.head())

# dzielnie danych na testowe i treningowe
X_train, X_test, y_train, y_test = train_test_split(
    X.values.tolist(), y.tolist(), test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=35)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

Y_predicted = model.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, Y_predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
#plt.show()

def PredictTestData(model):
    #pobranie testowych danych
    df_test = pd.read_csv("test.csv")
    df_test = df_test[['PassengerId','Pclass','Sex', 'Age','SibSp','Parch', 'Fare', 'Embarked']]
    df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
    df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    df_test.loc[:, 'Embarked'] = df_test['Embarked'].fillna(0)
    df_test.loc[:, 'Embarked'] = df_test['Embarked'].replace([np.inf, -np.inf], 0)
    df_test = pd.get_dummies(df_test, columns=['Embarked'], dtype=int)
    df_test.loc[:, 'Age'] = df_test['Age'].fillna(X['Age'].median())
    df_test.loc[:, 'Fare'] = df_test['Fare'].fillna(X['Fare'].median())




    #uzupełnienie NaN
    df_test = df_test.fillna(df_test.median(numeric_only=True))

    test_data = df_test.values.tolist()

    #predykcja
    ids = [row[0] for row in test_data]
    features = [row[1:] for row in test_data]
    predicted = model.predict(features)

    # Przygotowywanie danych pod kaggle
    output_data = {
        'PassengerId': [int(id) for id in ids],
        'Survived': [int(p) for p in predicted]
    }

    return output_data


#print(PredictTestData(model))
pd.DataFrame(PredictTestData(model)).to_csv('output.csv', index=False)




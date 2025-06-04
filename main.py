import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics



#Deklaracja metody knn
def euclidianDistance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.point = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = euclidianDistance(point, new_point)
                distances.append([distance, category])
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result

#pobranie danych z csv
df_train = pd.read_csv('train.csv')

df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = df_train[['Pclass', 'Sex', 'Age','SibSp','Parch', 'Fare', 'Embarked']]  # features
y = df_train['Survived']                        # labels


# uzupełnianie medianą brakującego danych
X.loc[:, 'Age'] = X['Age'].fillna(X['Age'].median())
X.loc[:, 'Fare'] = X['Fare'].fillna(X['Fare'].median())
X.loc[:, 'Parch'] = X['Parch'].fillna(X['Parch'].median())
X.loc[:, 'SibSp'] = X['SibSp'].fillna(X['SibSp'].median())

# dzielnie danych na testowe i treningowe
X_train, X_test, y_train, y_test = train_test_split(
    X.values.tolist(), y.tolist(), test_size=0.2, random_state=42
)

#wrzucanie wszystko do jednego dict
titanic_data = {}

for features, label in zip(X_train, y_train):
    if label not in titanic_data:
        titanic_data[label] = []
    titanic_data[label].append(features)

#użycie metody knn
clf = KNearestNeighbors(k=3)
clf.fit(titanic_data)

# sprawdzenie ile predykcji udało się zrobić
predicted_list = []
correct = 0
for features, actual in zip(X_test, y_test):
    predicted = clf.predict(features)
    predicted_list.append(predicted)
    if predicted == actual:
        correct += 1

#sprawdzenie dokładności
accuracy = correct / len(y_test)
print(f"Accuracy: {accuracy:.2f}")

#confusion matrix
confusion_matrix = metrics.confusion_matrix(predicted_list, y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.show()


def PredictTestData():
    #pobranie testowych danych
    df_test = pd.read_csv("test.csv")
    df_test = df_test[['PassengerId','Pclass','Sex', 'Age','SibSp','Parch', 'Fare', 'Embarked']]
    df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
    df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    df_test.loc[:, 'Age'] = df_test['Age'].fillna(X['Age'].median())
    df_test.loc[:, 'Fare'] = df_test['Fare'].fillna(X['Fare'].median())
    df_test.loc[:, 'Parch'] = df_test['Parch'].fillna(X['Parch'].median())
    df_test.loc[:, 'SibSp'] = df_test['SibSp'].fillna(X['SibSp'].median())


    #uzupełnienie NaN
    df_test = df_test.fillna(df_test.median(numeric_only=True))

    test_data = df_test.values.tolist()

    #predykcja danych
    test_predictions = []
    for row in test_data:
        passenger_id = row[0]         # First element
        features = row[1:]            # Remaining values for prediction

        predicted = clf.predict(features)
        test_predictions.append([passenger_id, predicted])


    #sformatowanie danych pod wysyłke do kaggle
    output_data = {'PassengerId':[], 'Survived':[]}
    for data in test_predictions:
        output_data['PassengerId'].append(int(data[0]))
        output_data['Survived'].append(int(data[1]))

    return pd.DataFrame(output_data)


#print(PredictTestData)
#PredictTestData.to_csv('output.csv', index=False)









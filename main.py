import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


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

df_train = pd.read_csv('train.csv')
df_train = df_train[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]  # wybierz kolumny

df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

titanic_data = {}

for _, row in df_train.iterrows():
    label = row['Survived']
    features = [row['Pclass'], row['Sex'], row['Age'], row['Fare']]

    if label not in titanic_data:
        titanic_data[label] = []
    titanic_data[label].append(features)

#odpalanie KNN
clf  = KNearestNeighbors()
clf.fit(titanic_data)

df_test = pd.read_csv("test.csv")
df_test = df_test[['PassengerId','Pclass', 'Sex', 'Age', 'Fare']]

df_test = df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})

test_data = {}

for _, row in df_test.iterrows():
    label = row['PassengerId']
    features = [row['Pclass'], row['Sex'], row['Age'], row['Fare']]

    if label not in test_data:
        test_data[label] = []
    test_data[label].append(features)

print(test_data)





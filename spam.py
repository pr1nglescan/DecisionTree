import numpy as np
from sklearn.model_selection import train_test_split
import decisiontree



data = []
with open('spambase.data', 'r') as emails: 
    for line in emails:
        entry = [float(n) for n in line.split(',')]
        data.append(entry)

data = np.array(data)
print(data.shape)
length = data.shape[0]
datax = data[:, :-1]
datay = data[:, -1:]
X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.2)

rando = decisiontree.Random_Forest(trees=decisiontree.fit_random_forest(np.column_stack((X_train, y_train)), theta=0.8, B=5))
print("Random Forest: " + str(decisiontree.accuracy(rando, X_test, y_test)))


spamtree = decisiontree.Tree(decisiontree.fit(np.column_stack((X_train, y_train))))
print("One tree: " + str(decisiontree.accuracy(spamtree, X_test, y_test)))
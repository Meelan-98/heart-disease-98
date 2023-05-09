
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def classify(data,classify_type):

    accuracies = []

    num_folds = 5

    # Split the dataset into folds using KFold
    kf = KFold(n_splits=num_folds, shuffle=True)
    fold_indices = kf.split(data)

    for fold, (train_indices, test_indices) in enumerate(fold_indices):

        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

        train_X = train_data.drop('num', axis=1)
        train_y = train_data['num']

        test_X = test_data.drop('num', axis=1)
        test_y = test_data['num']

        if(classify_type=="XGBoost"):

            xgb_model = xgb.XGBClassifier()
            xgb_model.fit(train_X, train_y)

            y_pred = xgb_model.predict(test_X)
        
        elif(classify_type=="Naive Bayes"):

            clf = MultinomialNB()
            clf.fit(train_X, train_y)

            y_pred = clf.predict(test_X)

        elif(classify_type=="K-Nearest Neigbour"):

            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(train_X, train_y)

            y_pred = clf.predict(test_X)

        accuracy = accuracy_score(test_y, y_pred)

        accuracies.append(accuracy)

    average = sum(accuracies) / len(accuracies)

    print('Average Accuracy with 5 folds for ',classify_type, ' is :', average)



from process.pcsv import preprocess
from classify.cpddf import classify
import sys
import inquirer

print("Welcome to the heart disease classification pipeline")

preprocessed_data = preprocess('data/cleaveland.csv')

choice = [
    inquirer.List('option',
                message="Select a Classification Algorithm :",
                choices=['XGBoost', 'Naive Bayes', 'K-Nearest Neigbour','Multinomial Logistic Regression']
                )
]

answer = inquirer.prompt(choice)

classify(preprocessed_data,str(answer['option']))
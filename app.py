from process.pcsv import preprocess
from classify.cpddf import classify

print("Welcome to the heart disease classification pipeline")

preprocessed_data = preprocess('data/cleaveland.csv')

classify(preprocessed_data)
"""
Reza Marzban
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from dataset_dowloader import dowload_dataset
from preprocess import PreProcess
from models import Linear_Regression, Logistic_Regression, NeuralNetwork1, NeuralNetwork2


print("\nCS 5783 - Machine Learning\nReza Marzban - A20098444\nFinal Project\n")


def main():
    dowload_dataset()
    preprocess = PreProcess()
    df = preprocess.load_tsv()
    features = preprocess.clean_data(df)
    features = preprocess.balance_data(features, 15000)
    x_train, y_train, x_test, y_test = preprocess.split_data(features)

    y_train_round, y_test_round = preprocess.round_labels(y_train, y_test)
    y_train_one, y_test_one = preprocess.labels_to_one_hot(y_train_round, y_test_round)
    lr = Logistic_Regression()
    lr.fit_and_evaluate(x_train, y_train_round, x_test, y_test_round)
    nn1 = NeuralNetwork1()
    nn1.fit_and_evaluate(x_train, y_train_one, x_test, y_test_one)
    linr = Linear_Regression()
    linr.fit_and_evaluate(x_train, y_train, x_test, y_test)
    nn2 = NeuralNetwork2()
    nn2.fit_and_evaluate(x_train, y_train, x_test, y_test)

    print("\n___________________End of the output___________________")


if __name__ == "__main__":
    main()

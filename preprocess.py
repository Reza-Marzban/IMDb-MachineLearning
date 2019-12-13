"""
Reza Marzban
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher


class PreProcess:
    """
    preprocess raw data into a numpy array n*m with n data points and m features
    the raw features are:
        Directors, titleType, isAdult, startYear, runtime, genres
    the label is the average rating of the movie.
    the categorical data is transformed into one-hot-encoder
    other numerical data has been normalized.

    It will create a numpy array containing features and label
    """
    def __init__(self):
        self.min_year = None
        self.max_year = None
        self.rating_index = None

    @staticmethod
    def _min_to_hours(min):
        hour = min/60
        if hour > 11:
            hour = 11
        return hour/11

    def _year_offset(self, year):
        return (year-self.min_year) / (self.max_year-self.min_year)

    @staticmethod
    def _remove_comma(text):
        text = text.replace("\\n", "")
        text = text.replace(" ", "")
        text = text.replace("-", "")
        text = text.replace(",", " ")
        return text

    @staticmethod
    def _choose_first(text):
        text = text.replace("\\n", "")
        text = text.replace(" ", "")
        text = text.replace(",", " ")
        text = text.split(" ", 1)[0]
        return text

    @staticmethod
    def _visualize_raw_data(df):
        df = df.sort_values(by=['rating'])
        df["rating"].hist(bins=9)
        plt.title("Rating Distribution")
        plt.xlabel("ratings")
        plt.ylabel("counts")
        plt.xticks(np.arange(0, 10, 1.0))
        plt.show()

        plt.scatter(df['rating'], df['runtime'], alpha=0.02, c="green")
        plt.xticks(np.arange(0, 10, 1.0))
        plt.xlabel("rating")
        plt.ylabel("runtime - Hour")
        plt.title("Runtime vs Rating")
        plt.show()

        plt.scatter(df['rating'], df['startYear'] + 1874, alpha=0.003, c="red", marker="x")
        plt.xticks(np.arange(0, 10, 1.0))
        plt.xlabel("rating")
        plt.ylabel("year")
        plt.title("Year vs Rating")
        plt.show()

        plt.scatter(df['rating'], df['genres'].apply(lambda x: x.split(' ')[0]), alpha=0.002, c="purple", marker="x")
        plt.xticks(np.arange(0, 10, 1.0))
        plt.xlabel("rating")
        plt.ylabel("genres")
        plt.title("Genres vs Rating")
        plt.show()

    @staticmethod
    def load_tsv():
        print("Loading and preprocessing data.")
        basics = pd.read_csv("data/title.basics.tsv", sep='\t', low_memory=False)
        crew = pd.read_csv("data/title.crew.tsv", sep='\t', low_memory=False)
        ratings = pd.read_csv("data/title.ratings.tsv", sep='\t', low_memory=False)
        temp = pd.merge(crew, basics, on="tconst")
        df = pd.merge(temp, ratings, on="tconst")
        df = df.drop(['tconst', 'primaryTitle', 'originalTitle', 'endYear', 'numVotes', "writers"], axis=1)
        return df

    def clean_data(self, df):
        df.replace("\\N", np.nan)
        df['startYear'] = df['startYear'].map(lambda x: "2000" if x == '\\N' else x)
        df = df[df.startYear.apply(lambda x: x.isnumeric())]
        df['runtimeMinutes'] = df['runtimeMinutes'].map(lambda x: "40" if x == '\\N' else x)
        df = df[df.runtimeMinutes.apply(lambda x: x.isnumeric())]
        df = df.apply(lambda x: x.astype(str).str.lower())
        convert_dict = {'startYear': float, 'runtimeMinutes': float, 'isAdult': float, 'averageRating': float}
        df = df.astype(convert_dict)
        df["runtimeMinutes"] = df["runtimeMinutes"].apply(self._min_to_hours)
        df = df.rename(columns={'runtimeMinutes': 'runtime', 'averageRating': 'rating'})
        self.min_year = df["startYear"].min()
        self.max_year = df["startYear"].max()
        df["startYear"] = df["startYear"].apply(self._year_offset)
        df["genres"] = df["genres"].apply(self._remove_comma)
        # self._visualize_raw_data(df)
        cv = CountVectorizer()
        df = df.join(pd.SparseDataFrame(cv.fit_transform(df["genres"]),
                                        df.index, cv.get_feature_names(), default_fill_value=0))
        df = df.drop(columns="genres")
        # df["titleType"] = df["titleType"].apply(self._choose_first)
        df["titleType"] = df["titleType"].astype('category')
        dfdummy = pd.get_dummies(df["titleType"], prefix='titleType')
        df = pd.concat([df, dfdummy], axis=1)
        df = df.drop(["titleType"], axis=1)
        df["directors"] = df["directors"].apply(self._choose_first)
        h = FeatureHasher(n_features=59, input_type="string")
        temp = h.transform(df.directors)
        temp = pd.DataFrame(temp.toarray())
        temp.columns = ["director-" + str(col) for col in temp.columns]
        df = df.drop(["directors"], axis=1)
        self.rating_index = df.columns.get_loc("rating")
        temp1 = df.to_numpy()
        temp2 = temp.to_numpy()
        features = np.hstack((temp1, temp2))
        return features

    def _seprate_label(self, features):
        label = features[:, self.rating_index]
        features = np.delete(features, self.rating_index, 1)
        return features, label

    def split_data(self, features):
        np.random.shuffle(features)
        n = len(features)
        train_size = int(n*0.8)
        test_size = n - train_size
        print(f"Total # of rows:\t{n}.\n # of rows for train:\t{train_size}.\n # of rows for test:\t{test_size}.")
        training, test = features[:train_size, :], features[train_size:, :]
        x_train, y_train = self._seprate_label(training)
        x_test, y_test = self._seprate_label(test)
        print(f"Number of features:\t{x_test.shape[1]}")
        print("____________________________________________________________________")
        x_train = x_train.astype(float)
        y_train = y_train.astype(float)
        x_test = x_test.astype(float)
        y_test = y_test.astype(float)
        return x_train, y_train, x_test, y_test

    def balance_data(self, features, n):
        np.random.shuffle(features)
        features = np.array(features, dtype=np.float)
        features[:, self.rating_index] = features[:, self.rating_index] - 1
        zero = features[np.logical_and(features[:, self.rating_index] >= 0.0, features[:, self.rating_index] < 0.5)][:n]
        one = features[np.logical_and(features[:, self.rating_index] >= 0.5, features[:, self.rating_index] < 1.5)][:n]
        two = features[np.logical_and(features[:, self.rating_index] >= 1.5, features[:, self.rating_index] < 2.5)][:n]
        three = features[np.logical_and(features[:, self.rating_index] >= 2.5, features[:, self.rating_index] < 3.5)][:n]
        four = features[np.logical_and(features[:, self.rating_index] >= 3.5, features[:, self.rating_index] < 4.5)][:n]
        five = features[np.logical_and(features[:, self.rating_index] >= 4.5, features[:, self.rating_index] < 5.5)][:n]
        six = features[np.logical_and(features[:, self.rating_index] >= 5.5, features[:, self.rating_index] < 6.5)][:n]
        seven = features[np.logical_and(features[:, self.rating_index] >= 6.5, features[:, self.rating_index] < 7.5)][:n]
        eight = features[np.logical_and(features[:, self.rating_index] >= 7.5, features[:, self.rating_index] < 8.5)][:n]
        nine = features[np.logical_and(features[:, self.rating_index] >= 8.5, features[:, self.rating_index] <= 9.0)][:n]
        return np.vstack((zero, one, two, three, four, five, six, seven, eight, nine))

    @staticmethod
    def round_labels(y_train, y_test):
        y_train = np.round(np.array(y_train, dtype=np.int))
        y_test = np.round(np.array(y_test, dtype=np.int))
        return y_train, y_test

    @staticmethod
    def labels_to_one_hot(y_train, y_test):
        dummy_train = np.zeros((len(y_train), 10))
        dummy_test = np.zeros((len(y_test), 10))
        dummy_train[np.arange(len(y_train)), y_train] = 1
        dummy_test[np.arange(len(y_test)), y_test] = 1
        return dummy_train, dummy_test


if __name__ == "__main__":
    print("\nCS 5783 - Machine Learning\nReza Marzban - A20098444\nFinal Project\n")
    print("Usage:preprocess.py is a helper function. Please run main.py\n")
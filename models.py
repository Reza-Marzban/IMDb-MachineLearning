"""
Reza Marzban
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def r2(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))


def evaluator(y_train, pred_train, y_test, pred_test, title, is_regression=False):
    """
    print out reports and evaluation on model prediction
    """
    if is_regression:
        y_test = np.round(y_test)
        pred_test = np.round(pred_test)
    else:
        train_acc = round(accuracy_score(y_train, pred_train), 4) * 100
        test_acc = round(accuracy_score(y_test, pred_test), 4) * 100
        print(f"\nTrain accuracy: {train_acc}, Test accuracy: {test_acc} .")
        precision, recall, fScore, _ = precision_recall_fscore_support(y_test, pred_test, average='weighted')
        print(f"performance on Test set:\nPrecision=\t{round(precision, 4)}\n"
              f"Recall=\t{round(recall, 4)}\nF-Score=\t{round(fScore, 4)}\n")

    labels = list(range(10))
    if is_regression or title == "Logistic Regression (classification)":
        cf = confusion_matrix(y_test, pred_test, labels)
    else:
        cf = confusion_matrix(y_test.argmax(axis=-1), pred_test.argmax(axis=-1), labels)
    cf = np.flip(cf, 0)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cf, vmin=0, cmap="gist_heat_r" )
    plt.title('Confusion matrix '+ title)
    fig.colorbar(cax)
    ax.xaxis.set_ticks(labels)
    ax.xaxis.set_ticks_position('bottom')
    labels1 = list(range(9, -1, -1))
    ax.yaxis.set_ticks(labels)
    ax.set_yticklabels(labels1)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Predicted Rating')
    plt.ylabel('True Rating')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cf[i, j], ha="center", va="center", color="g")
    plt.show()


class Linear_Regression:

    def __init__(self):
        self.reg = LinearRegression()

    def fit_and_evaluate(self, x_train, y_train, x_test, y_test):
        print("-----Training Linear Regression (Regression)-----")
        t1 = time.time()
        self.reg.fit(x_train, y_train)
        print(f"Training finished. Training time:{(time.time()-t1)/60} minutes.")
        train_pred = self.reg.predict(x_train)
        test_pred = self.reg.predict(x_test)
        print(f"Train R2: {round(r2_score(y_train, train_pred),4)}\n Test R2: {round(r2_score(y_test, test_pred),4)}")
        evaluator(y_train, train_pred, y_test, test_pred,  "Linear Regression", True)
        print("_____________________________________________________________________")
        print()
        return self.reg


class Logistic_Regression:

    def __init__(self):
        self.reg = LogisticRegression(solver='saga', verbose=10, max_iter=20, n_jobs=-1)

    def fit_and_evaluate(self, x_train, y_train, x_test, y_test):
        print("-----Training Logistic Regression (Classification)-----")
        t1 = time.time()
        self.reg.fit(x_train, y_train)
        print(f"Training finished. Training time:{(time.time()-t1)/60} minutes.")
        train_pred = self.reg.predict(x_train)
        test_pred = self.reg.predict(x_test)
        evaluator(y_train, train_pred, y_test, test_pred,  "Logistic Regression (classification)")
        print("_____________________________________________________________________")
        print()
        return self.reg


class NeuralNetwork1:
    """
    ANN for Classification
    """

    def __init__(self):
        print("-----Training Neural Network 1 (Classification)-----")
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(1024, input_shape=(100,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        adam = tf.keras.optimizers.Adam(0.001)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

    def fit_and_evaluate(self, x_train, y_train, x_test, y_test):
        t1 = time.time()
        epochs = 20
        history = self.model.fit(x_train, y_train, verbose=2, epochs=epochs,
                                 batch_size=1024, validation_data=(x_test, y_test))
        print(f"Training finished. Training time:{(time.time()-t1)/60} minutes.")
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        ticks = list(range(1, epochs+1))
        plt.plot(ticks, history.history['acc'], color='darkblue', linewidth=3)
        plt.plot(ticks, history.history['val_acc'], color='green', linewidth=3)
        plt.title('model accuracy NN1 (classification)')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.xticks(ticks)
        plt.subplot(1, 2, 2)
        # summarize history for loss
        plt.plot(ticks, history.history['loss'], color='darkblue', linewidth=3)
        plt.plot(ticks, history.history['val_loss'], color='green', linewidth=3)
        plt.title('model loss NN1 (classification)')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.xticks(ticks)
        plt.show()
        train_pred_score = self.model.predict(x_train)
        test_pred_score = self.model.predict(x_test)
        train_pred = np.zeros_like(train_pred_score)
        train_pred[np.arange(len(train_pred_score)), train_pred_score.argmax(1)] = 1
        test_pred = np.zeros_like(test_pred_score)
        test_pred[np.arange(len(test_pred_score)), test_pred_score.argmax(1)] = 1
        evaluator(y_train, train_pred, y_test, test_pred,  "NN1 (classification)")
        print("_____________________________________________________________________")
        print()
        return self.model


class NeuralNetwork2:
    """
    ANN for Regression
    """

    def __init__(self):
        print("-----Training Neural Network 2 (Regression)-----")
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(1024, input_shape=(100,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='relu'))
        adam = tf.keras.optimizers.Adam(0.001)
        self.model.compile(optimizer=adam, loss='mse', metrics=[r2])
        print(self.model.summary())

    def fit_and_evaluate(self, x_train, y_train, x_test, y_test):
        t1 = time.time()
        epochs = 20
        history = self.model.fit(x_train, y_train, verbose=2, epochs=epochs,
                                 batch_size=1024, validation_data=(x_test, y_test))
        print(f"Training finished. Training time:{(time.time()-t1)/60} minutes.")
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        ticks = list(range(1, epochs+1))
        plt.plot(ticks, history.history['r2'], color='darkblue', linewidth=3)
        plt.plot(ticks, history.history['val_r2'], color='green', linewidth=3)
        plt.ylim(0.2, 0.6)
        plt.title('model R Squared NN2 (Regression)')
        plt.ylabel('R2')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.xticks(ticks)
        plt.subplot(1, 2, 2)
        # summarize history for loss
        plt.plot(ticks, history.history['loss'], color='darkblue', linewidth=3)
        plt.plot(ticks, history.history['val_loss'], color='green', linewidth=3)
        plt.ylim(2.5, 4.5)
        plt.title('model loss (MSE) NN2 (Regression)')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.xticks(ticks)
        plt.show()
        train_pred = self.model.predict(x_train)
        test_pred = self.model.predict(x_test)
        print(f"Train R2: {round(r2_score(y_train, train_pred),4)}\n Test R2: {round(r2_score(y_test, test_pred),4)}")
        evaluator(y_train, train_pred, y_test, test_pred, "NN2 (Regression)", True)
        print("_____________________________________________________________________")
        print()
        return self.model


if __name__ == "__main__":
    print("\nCS 5783 - Machine Learning\nReza Marzban - A20098444\nFinal Project\n")
    print("Usage:models.py is a helper function. Please run main.py\n")

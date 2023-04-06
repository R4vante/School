import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use(['seaborn-v0_8-colorblind'])

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.math import confusion_matrix

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class Neural:

    def __init__(self, dataset, activation, model_name):
        """
        __init__ initializing function

        In this function, the inputvariables are initialized.

        Args:
            dataset (List): Used Dataset
            activation (String): Name of the activationfunction used
        """        
        self.model_name = model_name
        self.dataset = dataset
        self.activation = activation
        if self.dataset == "mnist":
            (self.X, self.y), (self.X_test, self.y_test) = keras.datasets.mnist.load_data()

            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, stratify=self.y, test_size=0.2)
        else:
            print("no dataset imported")
            exit()

    def model(self):
        """
        model Descides wich model is used. The model is then trained and a prediction will be made over the test set.

        - small: 1 input layer, 4 hidden layers, 1 output layer
        - big: 1 input layer, 10 hidden layers, 1 ouput layer
        - drop: same as small layer, but with a dropout of 30% at every layer.
        Args:
            model_name (string): name of the model used (small, big or drop)

        Returns:
            object:
                - model (object): the model function after training
                - results_df (dataframe): Pandas dataframe with the results of the model (losses and accuracy)
                - y_pred (array): array with the predictions of the testset 
        """
        if self.model_name == 'small':

            self.model = keras.Sequential([
                layers.Flatten(input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(10, activation = 'sigmoid'),
        ])


        elif self.model_name == 'big':

            self.model = keras.Sequential([
                layers.Flatten(input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(512, activation = self.activation),
                layers.Dense(10, activation = 'sigmoid'),
        ])


        elif self.model_name == 'drop':
            self.model = keras.Sequential([
                layers.Flatten(input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
                layers.Dropout(0.3),
                layers.Dense(512, activation = self.activation),
                layers.Dropout(0.3),
                layers.Dense(512, activation = self.activation),
                layers.Dropout(0.3),
                layers.Dense(512, activation = self.activation),
                layers.Dropout(0.3),
                layers.Dense(512, activation = self.activation),
                layers.Dropout(0.3),
                layers.Dense(10, activation = 'sigmoid'),
            ])

        else:
            print("Model name incorrect")
            exit()


        early_stopping = keras.callbacks.EarlyStopping(
            patience = 5,
            min_delta = 0.001,
            restore_best_weights = True
        )

        self.model.compile(
            loss = 'sparse_categorical_crossentropy', 
            optimizer = 'adam',
            metrics = ['accuracy', 'RootMeanSquaredError']
        )

        results = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_valid, self.y_valid),
            epochs = 200,
            batch_size=2000,
            callbacks=[early_stopping]
        )

        self.results_df = pd.DataFrame(results.history)

        self.y_pred = self.model.predict(self.X_test)
        self.y_pred = [np.argmax(i) for i in self.y_pred]
            
        self.f1_macro = f1_score(self.y_test, self.y_pred, average='macro')
        self.f1_micro = f1_score(self.y_test, self.y_pred, average='micro')
        self.f1_none = f1_score(self.y_test, self.y_pred, average=None)

        print("----------------------------------------------")
        print("\n")
        print("\n")
        print("-------------------F1 SCORES------------------")
        print("\n")
        print("\n")
        print(f"F1 score per class: {self.f1_none}")
        print(f"F1 score avarage: {self.f1_macro}")
        print(f"F1 score weighted avarage: {self.f1_micro}")
        print("\n")
        print("\n")
        print("----------------------------------------------")

        return self.model, self.results_df, self.y_pred

    def plot(self):
        """
        plot Function for making the plots

        Function returns a plots of:
        - First training samples as indication
        - The training and validation losses during training
        - The training and validation accuracies during training
        - The confusion matrix with dedicated F1-scores
        - The first predictions of the test set
        """
        x = tf.linspace(-5, 5, 100)
        activation_layer = layers.Activation(self.activation)
        y = activation_layer(x)


        fig, ax = plt.subplots(figsize=(15, 10))
        ax.grid(True)
        ax.plot(x, y)
        
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'f(x)')

        fig, ax = plt.subplots(1, 2, figsize=(15,8))

        self.results_df.loc[:, ['loss', 'val_loss']].plot(ax=ax[0], grid=True)
        self.results_df.loc[:, ['accuracy', 'val_accuracy']].plot(ax=ax[1], grid=True)

        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('Loss')

        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('accuracy  ')

        # Zoom in on losses

        fig, ax = plt.subplots()

        if self.model_name == "small" or self.model_name=='drop':
            self.results_df.loc[5:, ['loss', 'val_loss']].plot(ax=ax, grid=True)

            ax.set_xlabel('epoch')
            ax.set_ylabel('Loss')

        elif self.model_name == "big":
            self.results_df.loc[2:, ['loss', 'val_loss']].plot(ax=ax, grid=True)

            ax.set_xlabel('epoch')
            ax.set_ylabel('Loss')



        # confusion matrix

        if self.activation == 'linear':
            y_train_p = self.model.predict(self.X_train)
            y_train_p = [np.argmax(i) for i in y_train_p] 
            y_valid_p = self.model.predict(self.X_valid)
            y_valid_p = [np.argmax(i) for i in y_valid_p]
            cm_train = confusion_matrix(self.y_train, y_train_p)
            cm_valid = confusion_matrix(self.y_valid, y_valid_p)

            fig, ax = plt.subplots()

            sns.heatmap(cm_train, annot=True, fmt='d', ax = ax)

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')

            fig, ax = plt.subplots()
            sns.heatmap(cm_valid, annot=True, fmt='d', ax = ax)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')

        cm = confusion_matrix(self.y_test, self.y_pred)

        fig, ax = plt.subplots()
        
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title(f'f1-macro: %.4f    f1-micro: %.4f' %(self.f1_macro, self.f1_micro))
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')


        # plot of first 10 prediction against pictures
        fig, ax = plt.subplots(2, 5, figsize=(10,5))

        for i in range(10):
            plt.subplot(2,5, i+1)
            plt.imshow(self.X_test[i])
            plt.title(f"Prediction: {self.y_pred[i]}")

        plt.tight_layout()
        plt.show()


            


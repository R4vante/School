import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.math import confusion_matrix

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

plt.style.use(['seaborn-v0_8-colorblind'])

def main():

    (X, y), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.80)

    fig, (ax1, ax2) = plt.subplots(2,2)

    ax1[0].imshow(X_train[0])
    ax1[1].imshow(X_train[1])

    ax2[0].imshow(X_train[2])
    ax2[1].imshow(X_train[3])

    fig2, ax = plt.subplots()
    activation_layer1 = layers.Activation('linear')
    activation_layer2 = layers.Activation('relu')
    activation_layer3 = layers.Activation('elu')
    activation_layer4 = layers.Activation('gelu')

    x = tf.linspace(-5,5,100)
    y1 = activation_layer1(x)
    y2 = activation_layer2(x)
    y3 = activation_layer3(x)
    y4 = activation_layer4(x)

    ax.plot(x,y1, label = 'linear')
    ax.plot(x,y2, label = 'ReLu')
    ax.plot(x,y3, label = 'ELU')
    ax.plot(x,y4, label = 'GELU')

    plt.legend()

    model = keras.Sequential([
        layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
        layers.Dense(512, activation = 'relu'),
        layers.Dense(512, activation = 'relu'),
        layers.Dense(512, activation = 'relu'),
        layers.Dense(10, activation = 'sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='sparse_categorical_crossentropy'
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=5,
        min_delta = 0.001,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=200,
        batch_size=2000,
        callbacks=[early_stopping]
    )

    history_df = pd.DataFrame(history.history)

    fig, ax = plt.subplots(1,2, figsize=(15,8))

    history_df.loc[:,['loss', 'val_loss']].plot(ax=ax[0])
    history_df.loc[:,['accuracy', 'val_accuracy']].plot(ax=ax[1])

    y_pred = model.predict(X_test)
    y_pred = [np.argmax(i) for i in y_pred]

    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')

    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_title(f'f1-macro: %.4f    f1-micro: %.4f' %(f1_macro, f1_micro))
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')    

    fig, ax = plt.subplots(2,5, figsize=(10,15))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(X_test[i])
        plt.title(f'predicted {y_pred[i]}')
    fig.tight_layout()

    plt.show()




if __name__=='__main__':
    main()
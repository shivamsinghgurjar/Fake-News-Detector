import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM


MAXLEN = 100
NUM_WORDS = 5000


def build_tokenizer(texts):
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def encode(tokenizer, texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=MAXLEN)


def build_ann():
    model = Sequential([
        Flatten(input_shape=(MAXLEN,)),
        Dense(64, activation='relu'),
        Dense(32, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_rnn():
    model = Sequential([
        Embedding(input_dim=NUM_WORDS, output_dim=64, input_length=MAXLEN),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_deep_models(data: pd.DataFrame):
    X = data['statements']
    y = data['BinaryNumTarget']

    tokenizer = build_tokenizer(X)
    X_pad = encode(tokenizer, X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=0.2, stratify=y, random_state=42
    )

    # ANN
    print("\n Training ANN...")
    ann = build_ann()
    ann.fit(X_train, y_train, epochs=5, batch_size=32,
            validation_data=(X_test, y_test))
    loss, acc = ann.evaluate(X_test, y_test)
    print(f"ANN Accuracy: {acc:.4f}")

    # RNN (LSTM)
    print("\n Training RNN (LSTM)...")
    rnn = build_rnn()
    rnn.fit(X_train, y_train, epochs=5, batch_size=32,
            validation_data=(X_test, y_test))
    loss, acc = rnn.evaluate(X_test, y_test)
    print(f"RNN Accuracy: {acc:.4f}")

    # Save
    os.makedirs("models", exist_ok=True)
    ann.save("models/ann_model.h5")
    rnn.save("models/rnn_model.h5")
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("Saved ANN, RNN and tokenizer to models/")
    return ann, rnn, tokenizer


if __name__ == "__main__":
    data = pd.read_csv("data/processed_data.csv")
    train_deep_models(data)

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load data (top 10,000 most frequent words)
vocab_size = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# 2. Pad sequences to equal length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# 3. Build the RNN
model = keras.Sequential([
    layers.Embedding(vocab_size, 32),
    layers.LSTM(32),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 4. Train and evaluate
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.2%}")

# 5. Predict on new text
word_index = keras.datasets.imdb.get_word_index()

def encode(text):
       ids = []
       for w in text.lower().split():
           idx = word_index.get(w)
           if idx is not None and idx + 3 < vocab_size:
               ids.append(idx + 3)
           else:
               ids.append(2)  # unknown word
       return keras.preprocessing.sequence.pad_sequences([ids], maxlen=max_len)

for review in ["this movie was fantastic and moving",
               "boring plot and terrible acting"]:
    score = model.predict(encode(review), verbose=0)[0][0]
    print(f"{score:.2f} -> {'positive' if score > 0.5 else 'negative'}: {review}")
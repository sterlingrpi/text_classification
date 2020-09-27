import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import re
import string
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

titles = pd.read_csv('netflix_titles.csv')
descriptions = titles['description']
genres = titles['listed_in']

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
genres_encoded = vectorizer.fit_transform(genres).toarray()
genres_list = vectorizer.get_feature_names()

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

vocab_size = 10000
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=100)
vectorize_layer.adapt(np.array(descriptions))

embedding_dim = 16
model = Sequential([vectorize_layer,
                    Embedding(vocab_size, embedding_dim, name="embedding"),
                    Bidirectional(LSTM(128, return_sequences=True)),
                    Bidirectional(LSTM(128)),
                    Dense(len(genres_list))#, activation='sigmoid')
                    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_path = 'model_vectorize'

callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
             ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=0, save_freq='epoch'), ]

model.fit(x=np.array(descriptions),
          y=genres_encoded,
          epochs=10,
          batch_size=8,
          #steps_per_epoch=1000,
          callbacks=callbacks
)
model.save(model_path)

synopsis = np.array(descriptions)[0]
print('the synopsis is:')
print(synopsis)
predictions = model.predict(np.expand_dims(synopsis, axis=0))[0]
print('the top five predicted genres are:')
top_indexes = predictions.argsort()[-5:][::-1]
for index in top_indexes:
    print(genres_list[index])

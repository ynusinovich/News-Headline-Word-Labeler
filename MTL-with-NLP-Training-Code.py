# Imports and Load Data

import pandas as pd
import os
import numpy as np
import math
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers, regularizers
# from tensorflow.keras.utils import plot_model

# import pydot
# import graphviz

df = pd.read_csv("./data/ner_dataset.csv", encoding = 'unicode_escape')

# Create Training and Test Datasets

# for i in np.arange(0, len(df) - 1):
#     if pd.isnull(df.loc[i + 1, "Sentence #"]):
#         df.loc[i + 1, "Sentence #"] = df.loc[i, "Sentence #"]

# sentence_number_values = []
# for i in np.arange(0, len(df)):
#     sentence_number_values.append(int(df.loc[i, "Sentence #"].split()[1]))
# df["Sentence # Val"] = sentence_number_values

# number_of_train_sentences = math.floor(47959 * .8)
# train_sentences_numbers = set(np.random.choice(47959, number_of_train_sentences, replace = False))

# test_sentences_numbers = set(df["Sentence # Val"]) - train_sentences_numbers

# df.drop(columns = ["Sentence #"], inplace = True)

# train_df = df.loc[df["Sentence # Val"].isin(train_sentences_numbers)]
# test_df = df.loc[df["Sentence # Val"].isin(test_sentences_numbers)]

# train_df.to_csv("./data/ner_train_dataset.csv", index = False)
# test_df.to_csv("./data/ner_test_dataset.csv", index = False)


# Load Training and Test Datasets After Creation

train_df = pd.read_csv("./data/ner_train_dataset.csv")
test_df = pd.read_csv("./data/ner_test_dataset.csv")

for i in np.arange(0, len(train_df) - 1):
    if train_df.loc[i + 1, "Word"] == "'s":
        train_df.loc[i, "Word"] = train_df.loc[i, "Word"] + train_df.loc[i + 1, "Word"]
        train_df.loc[i + 1, "Word"] = "*"
    if i%1000 == 0:
        print(f"Row {i} of {len(train_df)} processed.")
train_df.drop(index = train_df[train_df["Word"] == "*"].index, inplace = True)

train_df.drop(index = train_df[train_df["Word"] == ","].index, inplace = True)
train_df.drop(index = train_df[train_df["Word"] == "."].index, inplace = True)
train_df.drop(index = train_df[train_df["Word"] == ":"].index, inplace = True)
train_df.drop(index = train_df[train_df["Word"] == ";"].index, inplace = True)
train_df.drop(index = train_df[train_df["Word"] == "\""].index, inplace = True)

train_df.reset_index(inplace = True, drop = True)

for i in np.arange(0, len(test_df) - 1):
    if test_df.loc[i + 1, "Word"] == "'s":
        test_df.loc[i, "Word"] = test_df.loc[i, "Word"] + test_df.loc[i + 1, "Word"]
        test_df.loc[i + 1, "Word"] = "*"
    if i%1000 == 0:
        print(f"Row {i} of {len(test_df)} processed.")
test_df.drop(index = test_df[test_df["Word"] == "*"].index, inplace = True)

test_df.drop(index = test_df[test_df["Word"] == ","].index, inplace = True)
test_df.drop(index = test_df[test_df["Word"] == "."].index, inplace = True)
test_df.drop(index = test_df[test_df["Word"] == ":"].index, inplace = True)
test_df.drop(index = test_df[test_df["Word"] == ";"].index, inplace = True)
test_df.drop(index = test_df[test_df["Word"] == "\""].index, inplace = True)

test_df.reset_index(inplace = True, drop = True)

train_df["CHU"] = ["O"] * len(train_df)
test_df["CHU"] = ["O"] * len(test_df)

train_df.rename(columns = {"Tag": "NER"}, inplace = True)
for i in np.arange(0, len(train_df)):
    if "-" in train_df.loc[i, "NER"]:
        train_df.loc[i, "CHU"] = train_df.loc[i, "NER"].split("-")[0]
        train_df.loc[i, "NER"] = train_df.loc[i, "NER"].split("-")[1]
    if i%1000 == 0:
        print(f"Row {i} of {len(train_df)} processed.")

test_df.rename(columns = {"Tag": "NER"}, inplace = True)
for i in np.arange(0, len(test_df)):
    if "-" in test_df.loc[i, "NER"]:
        test_df.loc[i, "CHU"] = test_df.loc[i, "NER"].split("-")[0]
        test_df.loc[i, "NER"] = test_df.loc[i, "NER"].split("-")[1]
    if i%1000 == 0:
        print(f"Row {i} of {len(test_df)} processed.")


# Encode Parts of Speech, Named Entity Recognition Tags, and Chunking Tags

train_combined = pd.DataFrame()
train_combined["Word"] = train_df.groupby("Sentence # Val")["Word"].apply(list)
train_combined["Word"] = [' '.join(i) for i in train_combined["Word"]]
train_combined["POS"] = train_df.groupby("Sentence # Val")["POS"].apply(list)
train_combined["NER"] = train_df.groupby("Sentence # Val")["NER"].apply(list)
train_combined["CHU"] = train_df.groupby("Sentence # Val")["CHU"].apply(list)
train_combined.reset_index(inplace = True)
train_df = train_combined

test_combined = pd.DataFrame()
test_combined["Word"] = test_df.groupby("Sentence # Val")["Word"].apply(list)
test_combined["Word"] = [' '.join(i) for i in test_combined["Word"]]
test_combined["POS"] = test_df.groupby("Sentence # Val")["POS"].apply(list)
test_combined["NER"] = test_df.groupby("Sentence # Val")["NER"].apply(list)
test_combined["CHU"] = test_df.groupby("Sentence # Val")["CHU"].apply(list)
test_combined.reset_index(inplace = True)
test_df = test_combined

for i in np.arange(0, len(train_df)):
    sentence_length = len(train_df.loc[i, "POS"])
    for j in np.arange(sentence_length, 105):
        train_df.loc[i, "POS"].append('*')
for i in np.arange(0, len(train_df)):
    sentence_length = len(train_df.loc[i, "NER"])
    for j in np.arange(sentence_length, 105):
        train_df.loc[i, "NER"].append('*')  
for i in np.arange(0, len(train_df)):
    sentence_length = len(train_df.loc[i, "CHU"])
    for j in np.arange(sentence_length, 105):
        train_df.loc[i, "CHU"].append('*')

for i in np.arange(0, len(test_df)):
    sentence_length = len(test_df.loc[i, "POS"])
    for j in np.arange(sentence_length, 105):
        test_df.loc[i, "POS"].append('*')
for i in np.arange(0, len(test_df)):
    sentence_length = len(test_df.loc[i, "NER"])
    for j in np.arange(sentence_length, 105):
        test_df.loc[i, "NER"].append('*')
for i in np.arange(0, len(test_df)):
    sentence_length = len(test_df.loc[i, "CHU"])
    for j in np.arange(sentence_length, 105):
        test_df.loc[i, "CHU"].append('*')

training_pos = []
for i in np.arange(0, len(train_df)):
    for i in train_df.loc[i, "POS"]:
        training_pos.append(i)
        
le_pos = preprocessing.LabelEncoder()
le_pos.fit(training_pos)

encoded_pos_train = train_df["POS"].apply(lambda x: le_pos.transform(x))
train_df["POS"] = encoded_pos_train
encoded_pos_test = test_df["POS"].apply(lambda x: le_pos.transform(x))
test_df["POS"] = encoded_pos_test

training_tag = []
for i in np.arange(0, len(train_df)):
    for i in train_df.loc[i, "NER"]:
        training_tag.append(i)
        
le_ner = preprocessing.LabelEncoder()
le_ner.fit(training_tag)

encoded_tag_train = train_df["NER"].apply(lambda x: le_ner.transform(x))
train_df["NER"] = encoded_tag_train
encoded_tag_test = test_df["NER"].apply(lambda x: le_ner.transform(x))
test_df["NER"] = encoded_tag_test

training_chu = []
for i in np.arange(0, len(train_df)):
    for i in train_df.loc[i, "CHU"]:
        training_chu.append(i)
        
le_chu = preprocessing.LabelEncoder()
le_chu.fit(training_chu)

encoded_chu_train = train_df["CHU"].apply(lambda x: le_chu.transform(x))
train_df["CHU"] = encoded_chu_train
encoded_chu_test = test_df["CHU"].apply(lambda x: le_chu.transform(x))
test_df["CHU"] = encoded_chu_test


# Create Vocabulary Index

tot_words = len(set(df["Word"]))
train_samples = train_df["Word"]
test_samples = test_df["Word"]

vectorizer = TextVectorization(max_tokens = None, output_sequence_length = 105)
text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))


# Load Pre-Trained Word Embeddings

# Load GloVe vectors
glove_dir = './data/'
path_to_glove_file = os.path.join(glove_dir, 'glove.6B.200d.txt')

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit = 1)
        coefs = np.fromstring(coefs, "f", sep = " ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 200
hits = 0
misses = 0

# Prepare Embedding Matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be zeroes.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer = keras.initializers.Constant(embedding_matrix),
    trainable = False)


# Build Model

sentence_input = keras.Input(shape = (None, ), dtype = "int64", name = 'sentence_input')
embedded_sentence = embedding_layer(sentence_input)

x = layers.Bidirectional(layers.LSTM(400, return_sequences = True, dropout = 0.50, recurrent_dropout = 0.25, name = "LSTM_1"), name = "bi_1")(embedded_sentence)

pos_branch = layers.TimeDistributed(layers.Dense(len(le_pos.classes_), activation = 'softmax', name = "pos_dense"), name = 'pos_output')(x)
ner_branch = layers.TimeDistributed(layers.Dense(len(le_ner.classes_), activation = 'softmax', name = "ner_dense"), name = 'ner_output')(x)
chu_branch = layers.TimeDistributed(layers.Dense(len(le_chu.classes_), activation = 'softmax', name = "chu_dense"), name = 'chu_output')(x)

model = keras.Model(inputs = sentence_input, outputs = [pos_branch, ner_branch, chu_branch], name = "model_1")
model.summary()

# tf.keras.utils.plot_model(model, to_file = 'model_plot.png', show_shapes = True, show_layer_names = True)


# Train Model

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

y_train_pos_array = to_categorical(train_df["POS"], len(le_pos.classes_))
y_test_pos_array = to_categorical(test_df["POS"], len(le_pos.classes_))

y_train_ner_array = to_categorical(train_df["NER"], len(le_ner.classes_))
y_test_ner_array = to_categorical(test_df["NER"], len(le_ner.classes_))

y_train_chu_array = to_categorical(train_df["CHU"], len(le_chu.classes_))
y_test_chu_array = to_categorical(test_df["CHU"], len(le_chu.classes_))

x_train = vectorizer(np.array([[s] for s in train_samples]))
x_test = vectorizer(np.array([[s] for s in test_samples]))

y_train_pos = tf.convert_to_tensor(y_train_pos_array)
y_test_pos = tf.convert_to_tensor(y_test_pos_array)

y_train_ner = tf.convert_to_tensor(y_train_ner_array)
y_test_ner = tf.convert_to_tensor(y_test_ner_array)

y_train_chu = tf.convert_to_tensor(y_train_chu_array)
y_test_chu = tf.convert_to_tensor(y_test_chu_array)

print(x_train.shape)
print(x_test.shape)

print(y_train_pos.shape)
print(y_test_pos.shape)

print(y_train_ner.shape)
print(y_test_ner.shape)

print(y_train_chu.shape)
print(y_test_chu.shape)

opt = keras.optimizers.RMSprop(lr = 1e-3)
model.compile(optimizer = opt,
              loss = {'pos_output': 'categorical_crossentropy', 'ner_output': 'categorical_crossentropy', 'chu_output': 'categorical_crossentropy'},
              loss_weights = {'pos_output': 0.2, 'ner_output': 1.0, 'chu_output': 1.0},
              metrics = ["acc"])

model.fit(x = x_train,
          y = {'pos_output': y_train_pos, 'ner_output': y_train_ner, 'chu_output': y_train_chu},
          epochs = 15,
          batch_size = 128,
          verbose = 1,
          validation_data = (x_test, {'pos_output': y_test_pos, 'ner_output': y_test_ner, 'chu_output': y_test_chu})
         )


# Save Model

np.save('./data/le_pos_classes.npy', le_pos.classes_)
np.save('./data/le_ner_classes.npy', le_ner.classes_)
np.save('./data/le_chu_classes.npy', le_chu.classes_)

string_input = keras.Input(shape=(1,), dtype = "string")
x = vectorizer(string_input)
preds = model(x)
end_to_end_model = keras.Model(string_input, preds)

end_to_end_model.save("./data/fitted_model")

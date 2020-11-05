# Imports

import pandas as pd
import os
import numpy as np
import math
import time
from sklearn import preprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import string
import re
import argparse

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Take Input from User in Terminal

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sentence", nargs = '?', default = "The North Sentinelese petitioned India's Department of Environmental Health to improve water quality in the Indian Ocean.",
                help = "Input sentence, preferably a news headline, use a backslash before each quotation mark")
ap.add_argument("-m", "--model_path", nargs = '?', default = "./data/fitted_model",
                help = "Path to model file on your computer", type = str)
ap.add_argument("-p", "--part_of_speech_label_encoder_path", nargs = '?', default = "./data/le_pos_classes.npy",
                help = "Path to part-of-speech label encoder on your computer", type = str)
ap.add_argument("-n", "--named_entity_recognition_label_encoder_path", nargs = '?', default = "./data/le_ner_classes.npy",
                help = "Path to named entity recognition label encoder on your computer", type = str)
ap.add_argument("-c", "--chunking_label_encoder_path", nargs = '?', default = "./data/le_chu_classes.npy",
                help = "Path to chunking label encoder on your computer", type = str)
args = vars(ap.parse_args())

# Load Label Encodings and Model

le_pos = preprocessing.LabelEncoder()
le_pos.classes_ = np.load(args["part_of_speech_label_encoder_path"])

le_ner = preprocessing.LabelEncoder()
le_ner.classes_ = np.load(args["named_entity_recognition_label_encoder_path"])

le_chu = preprocessing.LabelEncoder()
le_chu.classes_ = np.load(args["chunking_label_encoder_path"])

model = load_model(args["model_path"], compile = False)


# Make Predictions

input_sentence = args["sentence"]

sentence_array = input_sentence.split(" ")
for i in string.punctuation:
    while(i in sentence_array): 
        sentence_array.remove(i)
sentence_length = len(sentence_array)

prob_pos, prob_ner, prob_chu = model.predict(
    [[input_sentence]]
)

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
    return token_sequences

pos_list = logits_to_tokens(prob_pos, le_pos.classes_)
ner_list = logits_to_tokens(prob_ner, le_ner.classes_)
chu_list = logits_to_tokens(prob_chu, le_chu.classes_)

pos_list = pos_list[0][0:sentence_length]
ner_list = ner_list[0][0:sentence_length]
chu_list = chu_list[0][0:sentence_length]

pos_dict = {"CC": "Coordinating Conjunction",
            "CD": "Cardinal Number",
            "DT": "Determiner",
            "EX": "Existencial There",
            "FW": "Foreign Word",
            "IN": "Prepositon or Subordinating Conjunction",
            "JJ": "Adjective",
            "JJR": "Adjective, Comparative",
            "JJS": "Adjective, Superlative",
            "LS": "List Item Marker",
            "MD": "Modal",
            "NN": "Noun, Singular or Mass",
            "NNS": "Noun, Plural",
            "NNP": "Proper Noun, Singular",
            "NNPS": "Proper Noun, Plural",
            "PDT": "Predeterminer",
            "POS": "Possessive Ending",
            "PRP": "Personal Pronoun",
            "PRP$": "Possessive Pronoun",
            "RB": "Adverb",
            "RBR": "Adverb, Comparative",
            "RBS": "Adverb, Superlative",
            "RP": "Particle",
            "SYM": "Symbol",
            "TO": "To",
            "UH": "Interjection",
            "VB": "Verb, Base Form",
            "VBD": "Verb, Past Tense",
            "VBG": "Verb, Gerund or Present Pariciple",
            "VBN": "Verb, Past Participle",
            "VBP": "Verb, Non-3rd Person Singular Present",
            "VBZ": "Verb, 3rd Person Singular Present",
            "WDT": "Whdeterminer",
            "WP": "Whpronoun",
            "WP$": "Possessive Whpronoun",
            "WRB": "Whadverb",
            "$": "$",
            "*": "*",
            ',': ',', 
            '.': '.', 
            ':': ':', 
            ';': ';',
            '``': '``',
            'LRB': "Left Parentheses",
            'RRB': "Right Parentheses"}

ner_dict = {"geo": "Geographical Entity",
            "org": "Organization",
            "per": "Person",
            "gpe": "Geopolitical Entity",
            "tim": "Time Indicator",
            "art": "Artifact",
            "eve": "Event",
            "nat": "Natural Phenomenon",
            "O": "Not a Named Entity",
            "*": "*"}

chu_dict = {"B": "Begin Chunk",
            "I": "Inside Chunk",
            "O": "Not a Named Entity",
            "*": "*"}

indices = np.arange(0, len(sentence_array))
results = pd.DataFrame(columns = ["-Word-", "-Part-of-Speech-", "-Named Entity-", "-Beginning or Inside Chunk-"],
                       index = indices)
for i in results.index:
    results.loc[i, "-Word-"] = sentence_array[i]
    results.loc[i, "-Part-of-Speech-"] = pos_dict[pos_list[i]]
    results.loc[i, "-Named Entity-"] = ner_dict[ner_list[i]]
    results.loc[i, "-Beginning or Inside Chunk-"] = chu_dict[chu_list[i]]
print(results)





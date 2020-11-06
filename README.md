# News Headline Word Labeler

The goal of this project was create a multi-task learning neural network that can predict part-of-speech, named entity, and chunk for each word in a news headline. This project is an addendum to my presentation for A.I. Socratic Circles (https://ai.science/) about the seminal natural language processing paper: "A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning" by Ronan Collobert and Jason Weston. This neural network accomplishes the first three tasks in the paper, as a demonstration for the presentation.

## Explanation of Linguistic Terms

Parts-of-speech:<br>
- "CC": "Coordinating Conjunction"<br>
- "CD": "Cardinal Number"<br>
- "DT": "Determiner"<br>
- "EX": "Existential There"<br>
- "FW": "Foreign Word"<br>
- "IN": "Prepositon or Subordinating Conjunction"<br>
- "JJ": "Adjective"<br>
- "JJR": "Adjective, Comparative"<br>
- "JJS": "Adjective, Superlative"<br>
- "LS": "List Item Marker"<br>
- "MD": "Modal"<br>
- "NN": "Noun, Singular or Mass"<br>
- "NNS": "Noun, Plural"<br>
- "NNP": "Proper Noun, Singular"<br>
- "NNPS": "Proper Noun, Plural"<br>
- "PDT": "Predeterminer"<br>
- "POS": "Possessive Ending"<br>
- "PRP": "Personal Pronoun"<br>
- "PRP$": "Possessive Pronoun"<br>
- "RB": "Adverb"<br>
- "RBR": "Adverb, Comparative"<br>
- "RBS": "Adverb, Superlative"<br>
- "RP": "Particle"<br>
- "SYM": "Symbol"<br>
- "TO": "To"<br>
- "UH": "Interjection"<br>
- "VB": "Verb, Base Form"<br>
- "VBD": "Verb, Past Tense"<br>
- "VBG": "Verb, Gerund or Present Pariciple"<br>
- "VBN": "Verb, Past Participle"<br>
- "VBP": "Verb, Non-3rd Person Singular Present"<br>
- "VBZ": "Verb, 3rd Person Singular Present"<br>
- "WDT": "Whdeterminer"<br>
- "WP": "Whpronoun"<br>
- "WP$": "Possessive Whpronoun"<br>
- "WRB": "Whadverb"<br>
- 'LRB': "Left Parentheses"<br>
- 'RRB': "Right Parentheses"<br><br>

Named entities:<br>
- "geo": "Geographical Entity"<br>
- "org": "Organization"<br>
- "per": "Person"<br>
- "gpe": "Geopolitical Entity"<br>
- "tim": "Time Indicator"<br>
- "art": "Artifact"<br>
- "eve": "Event"<br>
- "nat": "Natural Phenomenon"<br>
- "O": "Not a Named Entity"<br><br>

Chunks:<br>
- "B": "Begin Chunk"<br>
- "I": "Inside Chunk"<br>
- "O": "Not a Named Entity"

## Instructions

1. Clone this project to a local directory.
    - Sample terminal command to clone the repository:
        - git clone https://github.com/ynusinovich/Multitask-Learning-with-Natural-Language-Processing.git
2. Create a subfolder of the main project directory called "data".
3. Download the GloVe 6-billion-word 200-dimensional word vector file to the "data" subfolder from https://www.kaggle.com/incorpes/glove6b200d/download.
4. Download the "ner_dataset.csv" (the smaller of the two datasets) from https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus?select=ner_dataset.csv into the "data" subfolder.
5. In the Terminal, navigate to the main project directory.
6. Set up the virtual environment.
    - Sample terminal commands to set up the environment (I use "python3" and "pip3" in the commands because my system python is python2):
        - python3 -m venv ./test_env/
        - source ./test_env/bin/activate
        - pip3 install -r requirements.txt
7. Run the training code.
    - Sample terminal command:
        - python3 Training.py
8. Run the testing code, typing -s followed by an input news headline sentence in quotes (if the news headline contains quotation marks, add a \ before each one).
    - Example terminal command with a made-up news headline:
        - python3 News_Headline_Labels.py -s "The North Sentinelese petitioned India's Department of Environmental Health to improve water quality in the Indian Ocean."
    - Another example terminal command with quotation marks:
        - python3 News_Headline_Labels.py -s "This \\"isn't real\\" news."
9. Additional optional parameters to run the program are -m, -p, -n, and -c: the file paths to the (1) model, (2) part-of-speech label encoder, (3) named entity recognition label encoder, and (4) chunking label encoder. The paths must be included if these files are not located in the default relative path from my project.

## Code References

https://keras.io/examples/nlp/pretrained_word_embeddings/<br>
https://github.com/rahul-pande/faces-mtl/blob/master/faces_mtl_age_gender.ipynb<br>
https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/<br>
https://medium.com/illuin/named-entity-recognition-with-bilstm-cnns-632ba83d3d41<br>
https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs/blob/master/nn.py<br>
https://nlpforhackers.io/lstm-pos-tagger-keras/

## Data Source and Part-of-Speech References

https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus<br>
https://www.eecis.udel.edu/~vijay/cis889/ie/pos-set.pdf

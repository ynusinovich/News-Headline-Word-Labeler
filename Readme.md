# Multitask Learning with Natural Language Processing

The goal of this project was create a multi-task learning neural network that can predict part-of-speech, named entity, and chunk for each word in a news headline. This project is an addendum to my presentation for A.I. Socratic Circles (https://ai.science/) about the seminal natural language processing paper: "A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning" by Ronan Collobert and Jason Weston. This neural network accomplishes the first three tasks in the paper, as a demonstration for the presentation.

## Explanation of linguistic terms:

Parts-of-speech: "CC": "Coordinating Conjunction",
                 "CD": "Cardinal Number",
                 "DT": "Determiner",
                 "EX": "Existential There",
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
                 "PR": "Particle",
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
                 "WRB": "Whadverb"
                 "$": "$",
                 "*": "*"
Named entities: "geo": "Geographical Entity",
                "org": "Organization",
                "per": "Person",
                "gpe": "Geopolitical Entity",
                "tim": "Time Indicator",
                "art": "Artifact",
                "eve": "Event",
                "nat": "Natural Phenomenon",
                "O": "Not a Named Entity"
Chunks: "B": "Begin Chunk",
        "I": "Inside Chunk",
        "O": "Not a Named Entity"

## Instructions:

1. Clone this project to a local directory.
2. In the Terminal, navigate to this project directory.
3. Set up the virtual environment.
  - Sample terminal commands to set up the environment:
    - python3 -m venv ./test_env/
    - source ./test_env/bin/activate
    - pip3 install -r requirements.txt
4. Run the program, providing an input news headline in quotes (if the news headline contains quotation marks, add a \ before each one.).
  - Example terminal command with a made-up news headline containing a word from outside the model's vocabulary:
    - python3 MTL-with-NLP-Example-Testing-Code.py -s "The North Sentinelese petitioned India's Department of Environmental Health to improve water quality in the Indian Ocean."
  - Another example terminal command with quotation marks:
    - python3 MTL-with-NLP-Testing-Code.py -s "This \"isn't real\" news."
5. Additional optional parameters to run the program are -m, -p, -n, and -c: the file paths to the (1) model, (2) part-of-speech label encoder, (3) named entity recognition label encoder, and (4) chunking label encoder. The paths must be included if these files are not located in the default relative path from my project.

## Code References:

https://keras.io/examples/nlp/pretrained_word_embeddings/<br>
https://github.com/rahul-pande/faces-mtl/blob/master/faces_mtl_age_gender.ipynb<br>
https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/<br>
https://medium.com/illuin/named-entity-recognition-with-bilstm-cnns-632ba83d3d41<br>
https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs/blob/master/nn.py<br>
https://nlpforhackers.io/lstm-pos-tagger-keras/

## Data Source and Part-of-Speech Reference:

https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus<br>
https://m-clark.github.io/text-analysis-with-R/part-of-speech-tagging.html

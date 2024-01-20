import pandas as pd
import numpy as np
import re
import nltk
from tqdm import tqdm
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer #for the split in tokens (words)
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
#nltk.download('wordnet')
#nltk.download('omw-1.4')
np.random.seed(42)

#is used in preprocessing, for the string elimination (like "plot kept under wraps" or "See full synopsis")
def useless_synopsis(synopsis):
    #useless sentences
    patterns_to_remove = [
        re.compile(r'see.*?synopsis', re.IGNORECASE),
        re.compile(r'plot.*?wraps', re.IGNORECASE),
        re.compile(r'plot.*?disclosed.*?time', re.IGNORECASE),
        re.compile(r'plot.*?disclosed', re.IGNORECASE),
        re.compile(r'disclosed.*?plot', re.IGNORECASE),
        re.compile(r'plot.*?available.*?time', re.IGNORECASE),
        re.compile(r'available.*?time', re.IGNORECASE),
        re.compile(r'plot.*?known.*?time', re.IGNORECASE),
        re.compile(r'plot.*?known', re.IGNORECASE)
    ]

    # Apply each pattern to remove specific phrases
    for pattern in patterns_to_remove:
        synopsis = pattern.sub('', synopsis)

    return synopsis


#in df is requested the df (with the synopsis column)
#return the df and a list of lists (each nested list is a synopsys)
def preprocessing(df, remove_useless_sentences = False, tokenize = False, remove_one_characters = False, lemmatize = False, remove_stop_words = False):

    print(f"You chose: remove_useless_sentences = {remove_useless_sentences}, tokenize = {tokenize}, remove_one_characters = {remove_one_characters}, lemmatize = {lemmatize} and remove_stop_words = {remove_stop_words}")
    df = df.reset_index(drop=True) #to reset the indices 
    if remove_useless_sentences:
        df["synopsis"] = df["synopsis"].apply(useless_synopsis)

    x = [] #where we will put the list of lists
    if tokenize:
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(df["synopsis"])):
            text = df["synopsis"][idx].lower()  # convert to lowercase.
            tokens = tokenizer.tokenize(text)  # split into words.

            if remove_one_characters:
                tokens = [token for token in tokens if len(token) > 1]

            if lemmatize:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]

            if remove_stop_words:
                tokens = [token for token in tokens if token not in STOP_WORDS]

            x.append(tokens)
            df.at[idx, "synopsis"] = " ".join(tokens)  # update the "synopsis" column with processed text

        return df, x

    return df #if tokenize false


# to return the y labels array
def target_variable(df, col):
    

    df_y = df[str(col)]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_y)

    #print('y_{}:'.format(col), y.shape)
    return y


# tokenizer padding
def tokenizer_padding(x_train, max_length, x_test = False):

    flattened_x_train = [' '.join(sentence_list) for sentence_list in x_train]

    #Tokenize the words using Keras Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(flattened_x_train)

    #Convert the text to sequences of integers
    sequences = tokenizer.texts_to_sequences(flattened_x_train)

    #Pad sequences to ensure uniform length with a maximum length
    padded_sequences_train = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    


    if x_test:
        sequences = tokenizer.texts_to_sequences(x_test)

        padded_sequences_test= pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

        return padded_sequences_train, padded_sequences_test, tokenizer

    return padded_sequences_train, tokenizer


# Define a function to create a lexicon for all the genres
def create_lexicon(df_train, genre_col, clean_synopsis_col):
    genres = np.unique(df_train[genre_col].values)
    # final output:
    lexicon = {}
    
    for genre in tqdm(genres):
        genre_lexicon = []
        output = df_train[df_train[genre_col] == genre][clean_synopsis_col]
        output_list = list(output)
    
        for e in output_list:
            genre_lexicon.extend(e)
        
        lexicon[genre] = genre_lexicon
    
    return lexicon


# Function to merge genres using a dictionary (to_merge)
def merge_genres(df, to_merge, inplace=True):
    new_genres = []
    for e in df['genre']:
        if e in set(to_merge.keys()):
            new_genres.append(to_merge[e])
        else:
            new_genres.append(e)
    
    df['new_genres'] = new_genres

    if inplace == True:
        df.drop(['genre'], axis=1, inplace=True)
        df.rename(columns={'new_genres': 'genre'}, inplace=True)
    
    return df
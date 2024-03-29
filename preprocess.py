import csv
import numpy as np
import pandas as pd
import re
import string

PAD_TOKEN = "*PAD*"
UNK_TOKEN = "*UNK*"
MAX_COMMENT_LENGTH = 200
MIN_WORD_APPEARANCES = 5


def read_data(train_file, test_inputs_file, test_labels_file):
    """
    Load text data from csv files

    :param train_file:  string, name of train file
    :param test_input_file:  string, name of test input file
    :param test_labels_file:  string, name of test labels file
    :return: Tuple of train containing:
    pandas dataframe containing all the training data (columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate)
    pandas dataframe containing all the testing data (columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate)
    """

    # drops about half of the nontoxic comments from the train set
    train = pd.read_csv(train_file)
    rows_to_keep = np.ceil(0.4 * len(train.index))
    non_toxic = train[(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] == 0).all(axis = 1)]
    toxic = train[(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] != 0).any(axis = 1)]
    non_toxic = non_toxic.head(int(rows_to_keep))
    train = pd.concat([non_toxic, toxic], axis = 0)
    train = train.sample(frac=1).reset_index(drop=True)

    # reads in all test inputs labels
    test_inputs = pd.read_csv(test_inputs_file)
    test_labels = pd.read_csv(test_labels_file)
    test = pd.merge(test_inputs, test_labels, on="id")

    # drops all test comments labeled with a -1
    test = test[(test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] != -1).all(axis=1)]
    return train, test

def build_vocab(train, test):
    """
    Builds vocab a word count dictionary. Removes rare words and adds UNK and PAD tokens.

    :param train:  pandas dataframe of training data
    :param test: pandas dataframe of testing data
    :return: dictionary: word --> unique index
    """

    # create vocab dictionary with words in train and test
    word_counts = {}
    for comment in train['comment_text']:
        for token in comment:
            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    for comment in test['comment_text']:
        for token in comment:
            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    current_id = 0
    vocab_dict = {}

    # unk words that appear less times than MIN_WORD_APPEARANCES
    for token, count in word_counts.items():
        if count > MIN_WORD_APPEARANCES:
            vocab_dict[token] = current_id
            current_id += 1
    vocab_dict[UNK_TOKEN] = current_id
    vocab_dict[PAD_TOKEN] = current_id + 1
    return vocab_dict

def pad_corpus(comments):
    """
    Pads comments to all be specified length.

    :param comments: iterable of comments
    :return: list of comments with padding added
    """
    padded_comments = []
    for line in comments:
        # pad comments shorter than MAX_COMMENT_LENGTH and truncate comments that are longer
        padded_comment = line[:MAX_COMMENT_LENGTH]
        padded_comment = padded_comment + [PAD_TOKEN] * (MAX_COMMENT_LENGTH - len(padded_comment))
        padded_comments.append(padded_comment)

    return padded_comments

def convert_to_id(vocab, sentences):
    """
    Convert sentences to indices

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

def parse_strings(comments):
    """
    parse punctuation out of comments and turn all words to lowercase

    :param comments: list of comments
    :return: list of comments with all words lowercased and without their punctuation
    """
    return comments.apply(lambda x: re.findall(f"[\w']+|[{string.punctuation}]", x.lower()))

def get_data(train_file, test_inputs_file, test_labels_file):
    """
    Preprocesses data and returns it in formatting ready to be fed into model.

    :param train_file: Path to the training file.
    :param test_inputs_file: Path to the testing input file.
    :param test_labels_file: Path to the testing label file.

    :return: Tuple of train containing:
    2-d numpy array with training comments in vectorized/id form [num_comments x MAX_COMMENT_LENGTH]
    2-d numpy array with training labels form [num_comments x 6]
    2-d numpy array with testing comments in vectorized/id form [num_comments x MAX_COMMENT_LENGTH]
    2-d numpy array with testing labels form [num_comments x 6]
    vocab (Dict containg word->index mapping)
    """
    train, test = read_data(train_file, test_inputs_file, test_labels_file)

    train['comment_text'] = parse_strings(train['comment_text'])
    test['comment_text'] = parse_strings(test['comment_text'])

    vocab_dict = build_vocab(train, test)

    train_inputs = pad_corpus(train['comment_text'])
    test_inputs = pad_corpus(test['comment_text'])

    train_inputs = convert_to_id(vocab_dict, train_inputs)
    test_inputs = convert_to_id(vocab_dict, test_inputs)

    train_labels = np.array(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])
    test_labels = np.array(test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

    return train_inputs, train_labels, test_inputs, test_labels, vocab_dict

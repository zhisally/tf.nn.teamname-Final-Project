import csv
import numpy as np

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
    list of training comments,
    2-d numpy array of training labels,
    list of testing comments,
    2-d numpy array of testing labels,
    Dict containg word->word count
    """
    
    # labels stored as toxic, severe_toxic, obscene, threat, insult, identity_hate
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []
    word_counts = {}
    with open(train_file) as train_f:
        csv_reader = csv.reader(train_f, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            train_inputs.append(row[1])
            for word in row[1].split():
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
            label = np.array(list(map(int, row[2:8])))
            train_labels.append(label)
    
    tests_to_remove = set()
    with open(test_labels_file) as test_labels_f:
        csv_reader = csv.reader(test_labels_f, delimiter=',')
        next(csv_reader)
        line_number = 0
        for row in csv_reader:
            label = np.array(list(map(int, row[1:7])))
            if np.any(label == -1):
                tests_to_remove.add(line_number)
            else:
                test_labels.append(label)
            line_number += 1

    with open(test_inputs_file) as test_inputs_f:
        csv_reader = csv.reader(test_inputs_f, delimiter=',')
        next(csv_reader)
        line_number = 0
        for row in csv_reader:
            if line_number not in tests_to_remove:
                test_inputs.append(row[1])
                for word in row[1].split():
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
            line_number += 1
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_inputs, train_labels, test_inputs, test_labels, word_counts

def build_vocab(word_counts):
    """
    Builds vocab a word count dictionary. Removes rare words and adds UNK and PAD tokens.

    :param word_counts:  Dict containg word->word count
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
    current_id = 0
    vocab_dict = {}
    for word, count in word_counts.items():
        if count > MIN_WORD_APPEARANCES:
            vocab_dict[word] = current_id
            current_id += 1
    vocab_dict[UNK_TOKEN] = current_id
    vocab_dict[PAD_TOKEN] = current_id + 1
    return vocab_dict, current_id + 1

def pad_corpus(comments):
    """
    Pads comments to all be specified length.

    :param comments: list of comments
    :return: list of comments with padding added
    """
    padded_comments = []
    for line in comments:
        padded_comment = line.split()[:MAX_COMMENT_LENGTH]
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
    train_inputs, train_labels, test_inputs, test_labels, word_counts = read_data(train_file, test_inputs_file, test_labels_file)
    vocab_dict, pad_index = build_vocab(word_counts)
    
    train_inputs = pad_corpus(train_inputs)
    test_inputs = pad_corpus(test_inputs)
    
    train_inputs = convert_to_id(vocab_dict, train_inputs)
    test_inputs = convert_to_id(vocab_dict, test_inputs)
    
    return train_inputs, train_labels, test_inputs, test_labels, vocab_dict
import csv
import numpy as np

PAD_TOKEN = "*PAD*"
UNK_TOKEN = "*UNK*"
MAX_COMMENT_LENGTH = 200
MIN_WORD_APPEARANCES = 5

def read_data(train_file, test_inputs_file, test_labels_file):
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
    padded_comments = []
    for line in comments:
        padded_comment = line.split()[:MAX_COMMENT_LENGTH]
        padded_comment = padded_comment + [PAD_TOKEN] * (MAX_COMMENT_LENGTH - len(padded_comment))
        padded_comments.append(padded_comment)

    return padded_comments

def convert_to_id(vocab, sentences):
    """
    DO NOT CHANGE

  Convert sentences to indexed 

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

def get_data(train_file, test_inputs_file, test_labels_file):
    train_inputs, train_labels, test_inputs, test_labels, word_counts = read_data(train_file, test_inputs_file, test_labels_file)
    vocab_dict, pad_index = build_vocab(word_counts)
    
    train_inputs = pad_corpus(train_inputs)
    test_inputs = pad_corpus(test_inputs)
    
    train_inputs = convert_to_id(vocab_dict, train_inputs)
    test_inputs = convert_to_id(vocab_dict, test_inputs)
    
    return train_inputs, train_labels, test_inputs, test_labels, vocab_dict
import csv
import numpy as np

def get_data(train_file, test_inputs_file, test_labels_file):
    # labels stored as toxic, severe_toxic, obscene, threat, insult, identity_hate
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []
    vocab_dict = {}
    current_id = 0
    with open(train_file) as train_f:
        csv_reader = csv.reader(train_f, delimiter=',')
        for row in csv_reader:
            train_inputs.append(row[1])
            for word in row[1]:
                if word not in vocab_dict:
                    vocab_dict[word] = current_id
                    current_id += 1
            train_labels.append(row[2:8])
        del train_inputs[0]
        del train_labels[0]

    with open(test_inputs_file) as test_inputs_f:
        csv_reader = csv.reader(test_inputs_f, delimiter=',')
        for row in csv_reader:
            test_inputs.append(row[1])
            for word in row[1]:
                if word not in vocab_dict:
                    vocab_dict[word] = current_id
                    current_id += 1
        del test_inputs[0]

    with open(test_labels_file) as test_labels_f:
        csv_reader = csv.reader(test_labels_f, delimiter=',')
        for row in csv_reader:
            test_labels.append(row[1:7])
        del test_labels[0]

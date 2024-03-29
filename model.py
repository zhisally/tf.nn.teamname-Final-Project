import numpy as np
import tensorflow as tf
from preprocess import get_data
from sklearn.metrics import roc_auc_score

from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PRINT_EVERY = 100

class Model(tf.keras.Model):
    def __init__(self, vocab_size, comment_length):

        """
        The Model class predicts the tags of online comments as either non-toxic or
        one or more toxic labels.

        :param vocab_size: The number of unique words in the data
        :param comment_length: The number of words per comment
        """

        super(Model, self).__init__()

        # initialize hyperparameters
        self.vocab_size = vocab_size
        self.comment_length = comment_length
        self.embedding_size = 50
        self.batch_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        self.rnn_size = 128

        #intitalizes trainable layers
        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.comment_length)
        self.GRU = tf.keras.layers.GRU(units = self.rnn_size,return_sequences=True, return_state = True)
        self.pooling = tf.keras.layers.GlobalMaxPool1D()
        self.dense1 = tf.keras.layers.Dense(6, activation='sigmoid')

    def call(self, inputs):
        """
        Forward pass through network to make predictions (each prediction is a
        vector of dimension 6, where each number represents the binary classification
        of the comment for each of the 6 categories).

        :param inputs: word ids of shape (batch_size, comment_length)
        :return: the batch labels
        """
        # embedding layer lookup
        inputs = tf.convert_to_tensor(inputs)
        embeddings = self.E(inputs)

        # GRU and max pooling layers
        output, cell_state = self.GRU(embeddings, None)
        pooled = self.pooling(output)

        # dense and sigmoid activation layer
        dense1_output = self.dense1(pooled)
        return dense1_output

    def loss(self, logits, labels):
        """
        Calculates average cross entropy binary loss of the predictions.

        :param logits: a matrix of shape (batch_size, 6) as a tensor
        :param labels: matrix of shape (batch_size, 6) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels,logits))

    def accuracy(self, logits, labels):
        """
        Calculates the average accuracy across a batch of comments

        :param logits: a matrix of shape (batch_size, 6) as a tensor
        :param labels: a matrix of shape (batch_size, 6) containing the labels
        :return: the accuracy of the model as a tensor of size 1
        """
        preds = tf.map_fn(tf.math.round, logits)
        diff = tf.math.abs(preds - labels)
        return 1 - tf.reduce_mean(diff, axis=0)

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,comment_length)
    :param train_labels: train labels (all labels for training) of shape (num_labels,comment_length)
    :return: None
    """
    m = train_inputs.shape[0]
    for i in np.arange(0, m, model.batch_size):
        batch_inputs = train_inputs[i:i+model.batch_size]
        batch_labels = train_labels[i:i+model.batch_size]
        if(len(batch_labels) < model.batch_size):
            break

        with tf.GradientTape() as tape:
            preds = model(batch_inputs)
            loss = model.loss(preds, batch_labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if(i % (model.batch_size * PRINT_EVERY) == 0):
            print(f"Loss after {int(i / model.batch_size)} batches ({np.round(i / m * 100, 2)}%): {loss}")


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,comment_length)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,comment_length)
    :returns: touple of accuracy for each binary classification (array of dimension 6), ROC score
    """
    m = test_inputs.shape[0]
    total_acc = 0
    total_labels = np.empty((0,6))
    total_preds = np.empty((0,6))

    num_batches = int(m / model.batch_size)

    for i in np.arange(0, m, model.batch_size):
        if(i % (model.batch_size * PRINT_EVERY) == 0):
            print(f"Testing: {int(i / model.batch_size)} out of {num_batches} batches")
        batch_inputs = test_inputs[i:i+model.batch_size]
        batch_labels = test_labels[i:i+model.batch_size]
        if(len(batch_labels) < model.batch_size):
            break

        preds = model(batch_inputs)
        total_acc += model.accuracy(preds, batch_labels)

        total_labels = np.append(total_labels, batch_labels, axis=0)
        total_preds = np.append(total_preds, preds, axis=0)

    roc_auc_acc = 0
    for i in range(6):
        class_labels = total_labels[:,i]
        class_preds = total_preds[:,i]
        if i == 1 or i == 3:
            class_labels = np.append(class_labels, [1], axis=0)
            class_preds = np.append(class_preds, [1], axis=0)
        class_score = roc_auc_score(class_labels, class_preds)
        print(f"Label: {i} ROC AUC Score: {class_score}")
        roc_auc_acc += class_score

    roc_auc_acc = float(roc_auc_acc / 6)

    accuracy = total_acc / num_batches
    return accuracy, roc_auc_acc

def visualize_embeddings(model, vocab_dict):
    # reduce dimension to 2
    pca = PCA(n_components=2)
    vectors = pca.fit_transform(model.E.get_weights()[0])

    # normalize the results so that we can view them more comfortably in matplotlib
    normalizer = preprocessing.Normalizer()
    norm_vectors = normalizer.fit_transform(vectors, 'l2')

    # plot the 2D normalized vectors
    x_vec = []
    y_vec = []
    words_of_interest = ["fuck", "fucker", "bitch", "lesbian", "shit", "crap", "kill", "hate", "cute", "hell", "mother", "father", "daughter", "son", "love", "awesome", "great", "gay"]
    for x,y in norm_vectors:
      x_vec.append(x)
      y_vec.append(y)

    f, axs = plt.subplots(1,1, figsize = (18,16))
    plt.scatter(x_vec, y_vec, c='white')
    for word in words_of_interest:
        plt.annotate(word, (norm_vectors[vocab_dict[word]][0], norm_vectors[vocab_dict[word]][1]))
    plt.show()

def main():
    train_inputs, train_labels, test_inputs, test_labels, vocab_dict = get_data('data_set/train.csv','data_set/test.csv','data_set/test_labels.csv')
    model = Model(len(vocab_dict), train_inputs.shape[1])
    train(model, train_inputs, train_labels)
    accuracy, roc_score = test(model, test_inputs, test_labels)
    print("Accuracy: ", accuracy)
    print("roc_score: ", roc_score)
    visualize_embeddings(model, vocab_dict)

if __name__ == '__main__':
    main()

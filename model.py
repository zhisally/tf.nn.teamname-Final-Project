import numpy as np
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # initialize hyperparameters
        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 50
        self.batch_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
        self.rnn_size = 128

        #intitalizes trainable layers
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size],stddev = 0.1,dtype = tf.float32))
        self.LSTM = tf.keras.layers.LSTM(self.rnn_size,return_sequences = True, return_state = True)
        self.dense_1 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn

        -Note 1: If you use an LSTM, the final_state will be the last two outputs of calling the rnn.
        If you use a GRU, it will just be the second output.

        -Note 2: You only need to use the initial state during generation. During training and testing it can be None.
        """
        #embedding layer lookup
        embeddings = tf.nn.embedding_lookup(self.E,inputs)
        #if initial state is none, ignore it
        if type(initial_state) == type(None):
            lstm, last_output, cell_state = self.LSTM(embeddings)
        else:
            lstm, last_output, cell_state = self.LSTM(embeddings,initial_state = initial_state)
        dense1 = self.dense_1(lstm)
        return dense1, (last_output, cell_state)

    def loss(self, logits, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        #calculates mean loss
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,logits))

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #batches and trains data
    for i in range(0, len(train_inputs), model.batch_size*model.window_size):
        batch_inputs = np.zeros((model.batch_size*model.window_size,))
        batch_labels = np.zeros((model.batch_size*model.window_size,))

        #if not enough examples left, ignore and stop training
        if(len(train_inputs)-i<model.batch_size*model.window_size):
            break
        batch_inputs = train_inputs[i:i+model.batch_size*model.window_size]
        batch_labels = train_labels[i:i+model.batch_size*model.window_size]

        b_inputs = np.array(batch_inputs).reshape((model.batch_size,model.window_size))
        b_labels = np.array(batch_labels).reshape((model.batch_size,model.window_size))

        #calculate gradient
        with tf.GradientTape() as tape:
            probs, _ = model.call(b_inputs, None)
            loss = model.loss(probs, b_labels)
            grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)

    """
    perplexity = 0
    num_predictions = 0

    #batches data and tests model
    for i in range(0, len(test_inputs), model.batch_size*model.window_size):

        batch_inputs = np.zeros((model.batch_size*model.window_size,))
        batch_labels = np.zeros((model.batch_size*model.window_size,))
        #if there isn't enough inputs left in data set, ignore the ones left and stop testing
        if(len(test_inputs)-i<model.batch_size*model.window_size):
            break
        num_predictions+=1
        batch_inputs = test_inputs[i:i+model.batch_size*model.window_size]
        batch_labels = test_labels[i:i+model.batch_size*model.window_size]

        b_inputs = np.array(batch_inputs).reshape((model.batch_size,model.window_size))
        b_labels = np.array(batch_labels).reshape((model.batch_size,model.window_size))

        #calculate perplexity
        probs, _ = model.call(b_inputs, None)
        perplexity += model.loss(probs, b_labels)
    return tf.exp(perplexity/num_predictions)

def generate_sentence(word1, length, vocab,model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    This is only for your own exploration. What do the sequences your RNN generates look like?

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))



def main():
    #gets data and initializes model
    train_ids, test_ids, vocab_dict = get_data('data/train.txt','data/test.txt')
    model = Model(len(vocab_dict))

    # Set-up the training step
    train_inputs = train_ids[:-1]
    train_labels = train_ids[1:]
    train(model, train_inputs, train_labels)
    # Set up the testing steps
    test_inputs = test_ids[:-1]
    test_labels = test_ids[1:]
    perplexity = test(model, test_inputs, test_labels)
    # print out perplexity
    print(perplexity)
    # generates sentence
    generate_sentence("It", 10, vocab_dict, model)
if __name__ == '__main__':
    main()

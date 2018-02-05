from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.layers import Bidirectional, BatchNormalization, Conv1D, GlobalMaxPooling1D
from keras.layers import Concatenate, TimeDistributed, Activation
from keras.layers import Multiply, Add, Lambda, LSTM, Dot, RepeatVector, Permute
import keras.backend as K

class charLSTM(object):
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    # Function to do a 1D-Convolution and Global max-pooling
    def Conv1DMaxPool(self, inputs, nb_filter, filter_size, padding, activation=None):
        """
        Arguments
        ==========
            inputs: tensor
            nb_filter: number of filters
            filter_size: size of the convolution filter
            padding (str): specify the padding scheme

        Output
        ==========
            Keras layer object
        """
        x = Conv1D(nb_filter, filter_size, padding=padding, activation=activation)(inputs)
        x = GlobalMaxPooling1D()(x)

        return x

    # Implementation of the highway layer -- create a mix of transformed and original input
    def highway_layer(self, v, activation):
        """
        Arguments
        ==========
        v: tensor
        activation(str): activation function for the nonlinear transformation

        Output
        ==========
            Keras layer object
        """

        # Grab the shape of the input tensor
        dim = K.int_shape(v)[1]

        # The transform gate dictation the proportion of transformed input to be passed
        transform_gate = Dense(dim,
                               activation='sigmoid', name='transform_gate')(v)

        # Carry gate is the (1 - transform gate) to carry original input
        carry_gate = Lambda(lambda x: 1 - x, output_shape=(dim,),
                            name='carry_gate')(transform_gate)

        # Nonlinear transformation of the input tensor
        nonlinear_v = Dense(dim, activation=activation,
                            name='nonlinear_input')(v)

        # Calculate the transform value to carry over
        transformed_v = Dot(axes=1, name='transformed_input')([transform_gate, nonlinear_v])
        # Calculate the original value to carry over
        carried_v = Dot(axes=1, name='carried_input')([carry_gate, v])

        # Combine the transform and carry gate
        value = Add()([transformed_v, carried_v])

        return value

    def get_model(self, embed_dim, num_chars, max_len, nb_sent, channels, dropout_prob, highway = False):
        """
        :param embed_dim: int; dim of the character embeddings
        :param num_chars: int; number of unique characters in the data
        :param max_len: int; max sentence length
        :param nb_sent:  int; max number of sentences
        :param channels: list/tuple; each tuple specify the convolution layer kernel size and nb filters
        :param dropout_prob: float; dropout probability
        :param highway: boolean; use highway layer
        :return:
            charModel: Keras model object
        """
        # Inputs for words and each comment need to be created separately
        sent = Input((max_len,))
        comment = Input((nb_sent, max_len))

        # Embedding of the characters
        embed = Embedding(num_chars + 1, embed_dim)(sent)

        # Three channels (i.e., n-grams)
        channels = [self.Conv1DMaxPool(embed, nf, ks, 'valid') for ks, nf in channels]

        # Concatenate the three channel values
        sent_encode = Concatenate()(channels)
        sent_encode = Dropout(dropout_prob)(sent_encode)

        if highway:
            # Pass through the highway layer
            sent_encode = self.highway_layer(sent_encode, 'tanh')

        # Encoder for each word from characters
        sent_encoder = Model(inputs=sent, outputs=sent_encode)

        # A joint of all encoded word in sequence make up the encoded comment
        encoded_comment = TimeDistributed(sent_encoder)(comment)

        # A bidirectional LSTM of encoded comments
        x = Bidirectional(LSTM(256, return_sequences=True, dropout=dropout_prob,
                               recurrent_dropout=dropout_prob))(encoded_comment)
        # x = Bidirectional(LSTM(128, dropout = 0.1, recurrent_dropout=0.1))(x)

        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_prob)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(self.num_outputs, activation='sigmoid')(x)

        charModel = Model(inputs=comment, outputs=out)

        return charModel


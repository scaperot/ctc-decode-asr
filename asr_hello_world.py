import tensorflow as tf
import numpy as np
import librosa
from string import ascii_lowercase
import argparse


import pdb
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model(Ly,input_shape,filters, kernel_size, conv_stride, conv_border, n_lstm_units, n_dense_units):
    input_audio = tf.keras.layers.Input(shape=input_shape[1:],name='audio')
    labels = tf.keras.layers.Input(name="label",shape=(Ly,))

    x = tf.keras.layers.Conv1D(filters,
                                             kernel_size,
                                             strides=conv_stride,
                                             padding=conv_border,
                                             activation='relu',name="1DConv1")(input_audio)
    x = x[:,::2,:] # Decimate by factor of 2 # out = (in-1)/2 + 1

    for i in range(7):
        #x = tf.keras.layers.Conv1D((i+2)*filters,
        #                                         kernel_size,
        #                                         strides=conv_stride,
        #                                         padding=conv_border,
        #                                         activation='relu',name="1DConv"+str(i+2))(x)
        x = tf.layers.conv1d(x, filters * (i+2), kernel_size, strides=conv_strides, activation=LeakyReLU, padding=conv_border) # out = in - filter + 1
        x = x[:,::2,:] # Decimate by factor of 2 # out = (in-1)/2 + 1
    
    '''
    lstm_layer = tf.keras.layers.LSTM(n_lstm_units,
                                           return_sequences=True,
                                           activation='tanh')

    lstm_layer_back = tf.keras.layers.LSTM(n_lstm_units,
                                                return_sequences=True,
                                                go_backwards=True,
                                                activation='tanh')
    x = tf.keras.layers.Bidirectional(lstm_layer, backward_layer=lstm_layer_back,name="blstm")(x)
    '''

    # 
    x = tf.keras.layers.Conv1D(n_dense_units,
                                             1,
                                             strides=conv_stride,
                                             padding=conv_border,
                                             activation='tanh',name="1DConvLast")(x)
    # Output layer
    x = tf.keras.layers.Dense(n_dense_units, activation="softmax", name="dense")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = tf.keras.models.Model(
        inputs=[input_audio, labels], outputs=output, name="asr_model_v1"
    )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


def generate_input_from_audio_file(path_to_audio_file, resample_to=8000):
    '''
    function to create input for our neural network from an audio file.
    The function loads the audio file using librosa, resamples it, and creates spectrogram form it
    :param path_to_audio_file: path to the audio file
    :param resample_to:
    :return: spectrogram corresponding to the input file
    '''
    # load the signals and resample them
    signal, sample_rate = librosa.core.load(path_to_audio_file)
    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)
    X = librosa.core.resample(signal, sample_rate, resample_to)
    X = np.reshape(X,(X.shape[0],1))
    # normalisation
    #Xnorm = (X - np.min(X) ) / ((np.max(X)-np.min(X))+1e-10)
    #Xnorm = np.reshape(Xnorm,(Xnorm.shape[0],1))
    #return tf.convert_to_tensor(Xnorm)
    return tf.convert_to_tensor(X)


def generate_target_output_from_text(target_text):
    '''
    Target output is an array of indices for each character in your string.
    The indices comes from a mapping that will
    be used while decoding the ctc output.
    :param target_text: (str) target string
    :return: array of indices for each character in the string
    '''
    space_token = ' '
    end_token = '>'
    blank_token = '%'
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]
    char_to_index = {}
    for idx, char in enumerate(alphabet):
        char_to_index[char] = idx

    y = []
    prev_char = ''
    for char in target_text:
    #    if prev_char == char:
    #        y.append(char_to_index[blank_token]) # for the case where there are double letters.
        y.append(char_to_index[char])
    #    prev_char = char
    y.append(char_to_index[end_token])
    return y


def num_to_char(arr):
    '''
    arr = numpy array of shape > 1
    '''

    space_token = ' '
    end_token = '>'
    blank_token = '%'
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]
    index_to_char = {}
    for idx, char in enumerate(alphabet):
        index_to_char[idx] = char

    o = []
    for ndx in arr:
        o.append(index_to_char[ndx])

    return o

def get_greedy_sequence(probabilities,alphabet):
    '''
    '''
    output_text = ''
    for timestep in probabilities[0]:
        output_text += alphabet[tf.math.argmax(timestep)]
    return output_text

if __name__ == '__main__':
    # args: filename (str), transcript (str), re-train (bool)

    # loading a song with librosa to get features.
    parser = argparse.ArgumentParser(description='Perform speech to text prediction on sample.wav and transcript.  output a raw predicted sequence from CTC layer of model.')
    parser.add_argument('--retrain', required=False,action='store_const',const=True,
                   help='if used, will call fit function.  otherwise, only do prediction based on saved model.')

    args = parser.parse_args()

    sample_call = 'sample.wav'
    transcript = ' MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL '.lower()


    X = generate_input_from_audio_file(sample_call)
    X = tf.expand_dims(X, axis=0)  # converting input into a batch of size 1
    y = generate_target_output_from_text(transcript)
    y = tf.expand_dims(tf.convert_to_tensor(y), axis=0)  # converting output to a batch of size 1
    #print('Input shape: {}'.format(X.shape))
    #print('Target shape: {}'.format(y.shape))


    if args.retrain:
        # hyperparameters:
        # 1. size of the transcript in characters
        # 2. number of 'timesteps'
        # 3. number of samples per timestep.
        # 4. number of kernels
        # 5. size of each kernel
        # 6. stride 
        # 7. padding - 'valid' means no padding
        # 8. size of LSTM activation units
        # 9. size of dense unit on output (i.e. cooresponds to the size of the alphabet)

        model = build_model(y.shape[1],X.shape,24, 15, 1, 'same', 200, 29)
        model.summary()
       
        pdb.set_trace()

        epochs = 100
        early_stopping_patience = 10

        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=early_stopping_patience, restore_best_weights=True
        )

        # Train the model
        history = model.fit(
            [X,y],
            epochs=epochs,
            callbacks=[early_stopping],
        )

        ## Get the prediction model by extracting layers till the output layer
        prediction_model = tf.keras.models.Model(
            model.get_layer(name="audio").input, model.get_layer(name="dense").output
        )
        prediction_model.summary()

        # save trained model for prediction
        prediction_model.save('asr_hello_world_model.h5',include_optimizer=True)

    else:
        prediction_model = tf.keras.models.load_model('asr_hello_world_model.h5')
    
    ctc_logits = prediction_model.predict(X)

    # save probabilities for each set of labels in time.
    np.save('asr_hello_world_logits.npy',ctc_logits)

    # greedy decoding
    space_token = ' '
    end_token = ">"
    blank_token = '%'
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]
    output_text = get_greedy_sequence(ctc_logits,alphabet)
    
    print('Transcript:',transcript)
    print("Raw Sequence Output:", output_text)

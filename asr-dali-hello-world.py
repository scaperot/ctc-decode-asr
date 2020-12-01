import tensorflow as tf
import numpy as np
import librosa,csv
from string import ascii_lowercase


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


def build_model(Ly,Tx,nx,filters, kernel_size, conv_stride, conv_border, n_lstm_units, n_dense_units):
    input_audio = tf.keras.layers.Input(shape=(Tx,nx),name='audio')
    labels = tf.keras.layers.Input(name="label",shape=(Ly,))

    # Inputs to the model
    x = tf.keras.layers.Conv1D(filters,
                                             kernel_size,
                                             strides=conv_stride,
                                             padding=conv_border,
                                             activation='relu',name="1DConv1")(input_audio)


    lstm_layer = tf.keras.layers.LSTM(n_lstm_units,
                                           return_sequences=True,
                                           activation='tanh')

    lstm_layer_back = tf.keras.layers.LSTM(n_lstm_units,
                                                return_sequences=True,
                                                go_backwards=True,
                                                activation='tanh')
    x = tf.keras.layers.Bidirectional(lstm_layer, backward_layer=lstm_layer_back,name="blstm")(x)
    
    # Output layer
    x = tf.keras.layers.Dense(n_dense_units, activation="softmax", name="dense")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = tf.keras.models.Model(
        inputs=[input_audio, labels], outputs=output, name="audio_model_v1"
    )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model



def create_spectrogram(signals):
    '''
    function to create spectrogram from signals loaded from an audio file
    :param signals:
    :return:
    '''
    stfts = tf.signal.stft(signals, frame_length=200, frame_step=80, fft_length=256)
    spectrograms = tf.math.pow(tf.abs(stfts), 0.5)
    return spectrograms


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
    signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)

    # create spectrogram
    X = create_spectrogram(signal_resampled)

    # normalisation
    means = tf.math.reduce_mean(X, 1, keepdims=True)
    stddevs = tf.math.reduce_std(X, 1, keepdims=True)
    X = tf.divide(tf.subtract(X, means), stddevs)
    return X


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

def ctc_output_with_time(x,sr,frame_step=80,stride_len=2):
    '''
    x = list of characters of size l
    magic number...
    '''
    len_characters   = len(x)
    characters       = np.reshape(np.array(list(x)),(1,len_characters))
    time_steps       = np.reshape(np.arange(len_characters) * (stride_len * frame_step / sr),(1,len_characters))

    return np.concatenate((characters.T,time_steps.T),axis=1)


# A utility function to decode the output of the network
def decode_batch_predictions(pred,greedy=True):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    if greedy:
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=greedy)[0][0]
    else:
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, beam_width=3)[0][0]

    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res.numpy())).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


if __name__ == '__main__':

    #reading from csv file where dali data was preprocessed in a form compatible with deepspeech
    csvfile = open('batch.csv','r')
    rows       = csv.reader(csvfile)
    filenames = []
    transcript = []
    for row in rows:
        filenames.append(row[0])
        transcript.append(row[2])

    filename = filenames[0]
    transcript = transcript[0]

    print(filename)
    print(transcript)

    X = generate_input_from_audio_file(filename,20050)
    X = tf.expand_dims(X, axis=0)  # converting input into a batch of size 1
    y = generate_target_output_from_text(transcript)
    y = tf.expand_dims(tf.convert_to_tensor(y), axis=0)  # converting output to a batch of size 1
    print('Input shape: {}'.format(X.shape))
    print('Target shape: {}'.format(y.shape))


    model = build_model(y.shape[1],X.shape[1],X.shape[2],24, 15, 1, 'valid', 300, 29)
    model.summary()

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
    ctc_output = prediction_model.predict(X)
    


    
    # greedy decoding
    space_token = ' '
    end_token = '>'
    blank_token = '%'
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]

    #lm_string = prefix_beam_search(ctc_output,alphabet,blank_token,end_token,space_token,lm)
    output_text = ''
    for timestep in ctc_output[0]:
        output_text += alphabet[tf.math.argmax(timestep)]
    print("Timing output:", output_text)

    decoded_text = decode_batch_predictions(ctc_output,False)
    print("Decoder output:", decoded_text)

    print(ctc_output_with_time(output_text,8000,80,1))

    # alignment error calculations.
    # 1. CTC provides a way to see when it thinks the word is 'complete' (i.e. spaces)
    # 2. What seems to happen some is that there is some ambiguity and it sends blanks on the end of words
    #    when the probability of a specific token is low.  

# ctc-decode-asr
Tensorflow Automatic Speech Recognition (ASR) starter model to learn about end to end ASR and CTC decoding.<br/><br/>

Most of this work began from the following github page: https://github.com/apoorvnandan/speech-recognition-primer<br/><br/>

Additionally, as I was learning, I wanted to see how Tensorflow APIs were used for decoding, so I mostly fork lifted the following OCR code here: https://keras.io/examples/vision/captcha_ocr/<br/><br/>

The code right now requires tensorflow and keras.<br/><br/>

To run it:<br/>
>> python asr.py <br/><br/>

The input transcript (for training) is the following:<br/>
MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL

The output will be a over-fit model to a specific audio file sample.wav from LibreSpeech corpus (5s).  The output will look something like this:<br/>
Epoch 100/100<br/>
1/1 [==============================] - 1s 516ms/sample - loss: 3.4581 <br/>

['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel>']<br/><br/><br/>

You can use beam search in various ways using the decode_batch_predictions function.

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
miistteerrr%%  %%%%%%%%%%%%uuiilltt%%%%%eerr  iiiss%%%%%%%%%   %tthee  aa%%%%%%%ppoooossttllee ooff%%%%%%  tthhe  %%%%%%%%%%%%%%%%mmiii%dd%dddlllee  cccllas%%sees  aaaannddd  wwwee%%%%  %%aarrreee   %ggllaadd%%%%%  tttoo   %%%%%%%%%%%wweelcc%%%%%%ooommee  hh%%%ii%%ss  ggoo%%%%%%%ssppeel<br/><br/><br/>

I'm not really interested in working on language models, but more interested in alignment of words with samples of audio, so the ctc decoding output is all I need for this excercise.

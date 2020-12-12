
import os,time,sys
import numpy as np

import DALI as dali_code
from DALI import utilities

def dali_setup():

    ################################
    # CHECK FOR DALI dataset folder
    ################################
    if os.path.isdir('DALI/'):
        dali_path = os.path.abspath('DALI/')
    else:
        print('DALI dataset not found in', os.path.abspath('.') + '/DALI/')
        sys.exit()

    ################################
    # CHECK FOR audio folder
    ################################
    if os.path.isdir('audio/'):
        audio_path = os.path.abspath('audio/')
    else:
        print('audio directory not found, trying to create it.')
        os.makedirs(os.path.abspath('.') + '/audio/')

    dali_info = dali_code.get_info(dali_path + '/info/DALI_DATA_INFO.gz')
    return dali_path, audio_path, dali_info

def get_cropped_transcripts(dali_annot,song_ndx,sample_rate):
    '''
    Input:
    dali_annot (DALI object) - 
    song_ndx (mx2 numpy array) - values are samples relative to beginning of song (i.e. 0 is first sample) 
            row - [start of window, termination of window]
            m windows that were created with crop_song
    '''
    #find the words and times in an array for faster access...?  i'm not sure if its faster.
    mcrops = song_ndx.shape[0]
    song_transcripts = []
    for j in range(mcrops):
        start = song_ndx[j,0] / sample_rate
        term  = song_ndx[j,1] / sample_rate
        window_secs  = np.array([start,term])

        song_transcripts.append(get_transcript_for_window(dali_annot,window_secs))

        #print('window:',window_secs, ', crop num:',j,', transcript:',song_transcript[j])
    return song_transcripts

def get_transcript_for_window(dali_annot,window_secs):
    '''
    Input:
    dali_annot (DALI object) - created using entry.annotations['annot']
    window_secs (tuple) - (start of window in secs,end of window in secs)
    
    Return:
    transcript (string)
    '''
    transcript = ''
    for i in range(len(dali_annot['words'])):
        #find first full onset word
        word = dali_annot['words'][i]['text']
        word_time  = dali_annot['words'][i]['time'] 
        # word starts after  the start of window 
        # word ends   before the end   of window 
        if word_time[0] > window_secs[0] and word_time[1] < window_secs[1]:
            transcript += (word + ' ')

    return transcript

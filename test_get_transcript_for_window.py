
import matplotlib.pyplot as plt
import numpy as np
import os,time,sys
import argparse,subprocess

import pdb
import DALI as dali_code
from DALI import utilities

import dali_helpers


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='given a time window, attempt to find the characters for a DALI generated transcript')
    parser.add_argument('-s','--start', required=False,type=float,default=10.2268,
                   help='start time of window of DALI song ID = 3698c37beab64ec39196875d69720822')
    parser.add_argument('-w','--window', required=False,type=float,default=10.2268,
                   help='size of window of DALI song, to create a transcript')

    args = parser.parse_args()

    #TODO: if audio is not available in audio, download

    #required inputs
    sr = 22050.0
    song_id = '3698c37beab64ec39196875d69720822'
    start = args.start
    term  = args.start+args.window
    window_secs  = np.array([start,term])
    window_samps = window_secs / sr 

    #setup DALI 
    dali_path, audio_path, dali_info = dali_helpers.dali_setup()

    #import song metadata
    dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
    entry = dali_data[song_id]
    annot = entry.annotations['annot']

    transcript = dali_helpers.get_transcript_for_window(annot, window_secs)

    print('Song:',song_id,', window:',window_secs, '(cooresponds to second segment...labeled 01), transcript:',transcript)

    #if song is in the audio directory, play it.
    filename  = audio_path+'/'+song_id + '_01.wav'
    subprocess.call(['aplay', filename])


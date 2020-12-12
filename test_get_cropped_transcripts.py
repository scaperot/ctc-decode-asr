
import matplotlib.pyplot as plt
import numpy as np
import os,time,sys
import argparse,subprocess

import pdb
import DALI as dali_code
from DALI import utilities

import dali_helpers

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='calculate all windows based on 10.2268s window with a step thats half the size, find the words for a DALI generated transcript for song ID 3698c37beab64ec39196875d69720822 (assuming that songs are already in audio/ directory.')
    parser.add_argument('-w','--window', required=False,type=float,default=10.2268,
                   help='size of window to chope up transcript')

    args = parser.parse_args()

    #required inputs
    sample_rate = 22050.0
    song_id = '3698c37beab64ec39196875d69720822'

    #setup DALI 
    dali_path, audio_path, dali_info = dali_helpers.dali_setup()

    #import song metadata
    dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
    dali_entry = dali_data[song_id]
    dali_annot = dali_entry.annotations['annot']

    #calcuate indices...by importing from audio/
    song_ndx = np.load(audio_path+'/'+song_id+'_crop_indices.npy')
    ncrops = song_ndx.shape[0]

    tstart = time.time()
    #find the words and times in an array for faster access...?  i'm not sure if its faster.

    tlist = dali_helpers.get_cropped_transcripts(dali_annot,song_ndx,sample_rate)

    tend = time.time()
    print('Time to process one song:',tend-tstart)

    k = np.random.randint(ncrops)
    if k < 10:
        k_str = str(k).zfill(2)
    elif k < 100:
        k_str = str(k).zfill(1)
    else:
        k_str = str(k)
    for i in range(len(tlist)):
        print('selected segment:',k,', segment:',i,', transcript:',tlist[i])
    #if song is in the audio directory, play it.
    filename  = 'audio/'+song_id + '_'+str(k_str)+'.wav'
    subprocess.call(['aplay', filename])


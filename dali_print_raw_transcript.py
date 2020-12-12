
import matplotlib.pyplot as plt
import numpy as np
import os,time,sys,argparse

import pdb
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

def dali_print_raw_transcript(song_id):
    dali_path, audio_path, dali_info = dali_setup()
    dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
    dali_entry = dali_data[song_id]

    annot = dali_entry.annotations['annot']
    for i in range(len(annot['paragraphs'])):
        words = annot['paragraphs'][i]['text']
        time  = annot['paragraphs'][i]['time']
        print('PARAGRAPH',i,':',words, 'TIME:',time)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='print a song transcript')
    parser.add_argument('-s','--song-id', required=False,type=str,default='3698c37beab64ec39196875d69720822',
            help='DALI song id. default: 3698c37beab64ec39196875d69720822')

    args = parser.parse_args()
    
    song_id = args.song_id
    dali_print_raw_transcript(song_id)




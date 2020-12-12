
import matplotlib.pyplot as plt
import numpy as np
import os,time,sys

import pdb
import DALI as dali_code
from DALI import utilities

if __name__ == "__main__":

    sr = 22050
    if os.path.isdir('DALI/'):
        dali_path = os.path.abspath('DALI/')
    else:
        print('DALI dataset not found in',os.path.abspath('.')+'/DALI/')
        sys.exit()
    dali_info = dali_code.get_info(dali_path + '/info/DALI_DATA_INFO.gz')
    
    allsongfilenames = utilities.get_files_path(dali_path,'.gz')
    delta = []
    song_id = ''

    all_songs = False
    if all_songs:
        n = len(allsongfilenames)
    else:
        n = 1

    start = time.time()
    for i in range(n):
        #import song metadata
        song_id =  os.path.relpath(allsongfilenames[i],dali_path).split('.')[0]
        dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
        entry = dali_data[song_id]
        annot = entry.annotations['annot']
        for i in range(len(annot['paragraphs'])):
            words = annot['paragraphs'][i]['text']
            time  = annot['paragraphs'][i]['time']
            print('PARAGRAPH',i,':',words, 'TIME:',time)

        #if entry.info['metadata']['language']=='english':
        #    print('first song that is english...',i)
        #    sys.exit()

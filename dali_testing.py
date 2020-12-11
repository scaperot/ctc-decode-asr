
import matplotlib.pyplot as plt
import numpy as np
import os,time

import pdb
import DALI as dali_code
from DALI import utilities

if __name__ == "__main__":

    sr = 22050
    ################################
    # CHECK FOR DALI dataset folder
    ################################
    if os.path.isdir('DALI/'):
        dali_path = os.path.abspath('DALI/')
    else:
        print('DALI dataset not found in',os.path.abspath('.')+'/DALI/')
        sys.exit()

    dali_info = dali_code.get_info(dali_path + '/info/DALI_DATA_INFO.gz')
    
    allsongfilenames = utilities.get_files_path(dali_path,'.gz')
    delta = []
    song_id = ''

    all_songs = True
    if all_songs:
        n = len(allsongfilenames)
    else:
        n = 1

    start = time.time()
    h = 0
    for i in range(n):
        #import song metadata
        song_id =  os.path.relpath(allsongfilenames[i],dali_path).split('.')[0]
        dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
        entry = dali_data[song_id]

        #language analysis
        if entry.info['metadata']['language'] == 'english':
            h += 1
        

    print(h,'out of ',n,'songs are english.')

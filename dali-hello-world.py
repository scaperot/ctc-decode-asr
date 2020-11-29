import matplotlib.pyplot as plt
import numpy as np
import os,time

import pdb
import DALI as dali_code
from DALI import utilities

def line_length_hist_secs(data):
    counts,bins = np.histogram(data)
    span = (np.min(data),np.max(data))
    nbins = span[1] - span[0]
    plt.hist(data,bins=int(nbins),range=(0,20))
    plt.show(block=False)

if __name__ == '__main__':
    sr = 22050
    dali_path = '/home/das/Downloads/DALI/'
    dali_info = dali_code.get_info(dali_path + 'info/DALI_DATA_INFO.gz')

    # ###############################################
    #
    print("Measure the length of lines for 1 song...")
    #
    # ###############################################
    
    #import song metadata
    song_id = 'ff6ec422cfca4c46a0671072ace16352'
    dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
    entry = dali_data[song_id]
    
    #get all lines of song
    annot = entry.annotations['annot']['lines']
    lines = [i['time'] for i in dali_code.annot2frames(annot,1/sr)]
    nlines = len(lines)
    delta0 = np.zeros((nlines,))
    for i in range(nlines):
        delta0[i] = lines[i][1] - lines[i][0]
    print('max line length:',np.max(delta0/sr),' for song:',entry.info['title'])

    # plot histogram of values.
    plt.figure(1)
    line_length_hist_secs(delta0/sr)

    # ###############################################
    #
    print("Measure the length of lines for 10 songs...")
    #
    # ###############################################
    allsongfilenames = utilities.get_files_path(dali_path,'.gz')
    delta = []
    song_id = ''
    n = len(allsongfilenames)
    start = time.time()
    for i in range(n):
        #import song metadata
        song_id =  os.path.relpath(allsongfilenames[i],dali_path).split('.')[0]
        dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
        entry = dali_data[song_id]

        #get all lines of song
        annot = entry.annotations['annot']['lines']
        lines = []
        lines = [i['time'] for i in dali_code.annot2frames(annot,1/sr)]
        nlines = len(lines)

        for j in range(nlines):
            delta.append(lines[j][1] - lines[j][0])
    done = time.time()
    delta = np.array(delta)/sr
    print('max line length for',n,'songs:',np.max(delta))
    print('number of lines above 10 seconds:',np.sum(delta > 10))
    print('time to process',n,'songs:',done-start)


    # plot histogram of values.
    plt.figure(2)
    line_length_hist_secs(delta)

    np.save('dali_histogram.npy',delta)

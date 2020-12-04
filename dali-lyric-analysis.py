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
    dali_path = os.path.abspath('DALI/')
    dali_info = dali_code.get_info(dali_path + '/info/DALI_DATA_INFO.gz')

    # ###############################################
    #
    print("Measure the length of lines for 10 songs...")
    #  max line length for 5358 songs: 700.2 secs
    #  number of lines above 10 seconds: 605
    #  time to process 5358 songs: 215.2 secs
    #
    # ###############################################
    allsongfilenames = utilities.get_files_path(dali_path,'.gz')
    delta = []
    song_id = ''
    n = len(allsongfilenames)
    start = time.time()
    alphabet = []
    for i in range(n):
        #import song metadata
        song_id =  os.path.relpath(allsongfilenames[i],dali_path).split('.')[0]
        dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
        entry = dali_data[song_id]

        #get all lines of song
        annot = entry.annotations['annot']['lines']
        text = []
        text = [i['text'] for i in dali_code.annot2frames(annot,1/sr)]
        #pdb.set_trace()
        for i in range(len(text)):
            #transform to list of unique characters from text
            #pdb.set_trace()
            tmp = list(set(text[i]))

            #add characters to alphabet, then remove duplicates
            alphabet.extend(tmp)
            alphabet = list(set(alphabet))


    terminate = time.time()
    alphabet.sort()

    #delta = np.array(delta)/sr
    #print('max line length for',n,'songs:',np.max(delta))
    #print('number of lines above 10 seconds:',np.sum(delta > 10))
    print('alphabet for DALI is:',''.join(alphabet))
    print('time to process',n,'songs:',terminate-start)


    # plot histogram of values.
    #plt.figure(2)
    #line_length_hist_secs(delta)

    np.save('dali_alphabet.txt',''.join(alphabet))

import matplotlib.pyplot as plt
import numpy as np
import os,time

import pdb
import DALI as dali_code
from DALI import utilities

import re

#using Gupta's lyric preprocessing function...
#need to check licensing, etc.
def CleanUpLyrics(lyrics_raw):
    line = lyrics_raw.lower()
    regex = re.compile('[$*=:;/_,\.!?"\n]')
    stripped_line = regex.sub('', line)
    if stripped_line == '': return
    check_for_bracket_words = stripped_line.split(' ')
    non_bracket_words = []

    for elem in check_for_bracket_words:
        if elem == "": continue #remove extra space
        if '(' in elem or ')' in elem: continue
        if elem[-1] == '\'': elem = elem.replace('\'','g') #Check if "'" is at the end of a word, then replace it with "g", eg. makin' => making
        if elem=="'cause": elem="cause"
        elem=elem.replace('-',' ')
        elem=elem.replace('&','and')
        non_bracket_words.append(elem)
    stripped_line = ' '.join(non_bracket_words)
    return stripped_line.upper()

if __name__ == '__main__':
    sr = 22050
    dali_path = os.path.abspath('DALI/')
    dali_info = dali_code.get_info(dali_path + '/info/DALI_DATA_INFO.gz')

    # ###############################################
    #
    print("Measure the length of lines for all songs...")
    #  time to process 5358 songs: 232.2 seconds
    #
    # ###############################################
    allsongfilenames = utilities.get_files_path(dali_path,'.gz')
    delta = []
    song_id = ''
    n = len(allsongfilenames)
    start = time.time()
    alphabet = []
    songs_with_numbers = []
    for j in range(1000):
        #import song metadata
        song_id =  os.path.relpath(allsongfilenames[j],dali_path).split('.')[0]
        dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
        entry = dali_data[song_id]

        #get all lines of song
        annot = entry.annotations['annot']['lines']
        text = []
        text = [j['text'] for j in dali_code.annot2frames(annot,1/sr)]

        # go through each line and find unique characters and add them
        # to the alphabet
        for i in range(len(text)):
            tmp1 = CleanUpLyrics(text[i])
            if tmp1 == None:
                continue
            
            #transform to list of unique characters from text
            #tmp2 = list(set(text[i]))
            tmp2 = list(set(tmp1))

            searchfor = ['[',']']
            if any(substring in tmp2 for substring in searchfor):
                print('song:',j,', line:',i,text[i])
                songs_with_numbers.append(j)
                

            #add characters to alphabet, then remove duplicates
            alphabet.extend(tmp2)
            alphabet = list(set(alphabet))


    terminate = time.time()
    alphabet.sort()

    print('alphabet for DALI is:',''.join(alphabet))
    print('time to process',n,'songs:',terminate-start)

    np.save('dali_alphabet.txt',''.join(alphabet))
    np.save('dali_numeric.npy',songs_with_numbers)

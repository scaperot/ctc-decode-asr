import matplotlib.pyplot as plt
import numpy as np
import sys, os, time, subprocess
import librosa
import soundfile as sf
import pdb
import re
import DALI as dali_code
from DALI import utilities

import dali_helpers


'''
Take a single song and break up into ~10s chunks 
NOTE 1: This is a modification to Stoller's 'End to end lyrics alignment for polyphonic music 
  using an audio-to-character recogition model.', in which this code does not use context windows 
  on each side of the 10s prediction window.

Data Metadata Formata
- 225501 samples @22050Hz (10.2268s)

For Training: 
- shift 112750 samples @22050Hz (5.11s)

TODO: For Prediction
- shift by the size of the total samples (i.e. no overlap).

'''

#using Gupta's lyric preprocessing function...
#need to check licensing, etc.
# Changes: added # and @ to removed characters
#          returns lower case, not upper
# TODO: For now, copied from dali-lyric-analysis.py,
#       should put in a common file and import instead
def CleanUpLyrics(lyrics_raw):
    line = lyrics_raw.lower()
    regex = re.compile('[$*=:;/_,\.!?#@"\n]')  # Added # and @
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
    return stripped_line


def download_song(song_id, dali_info, audio_path, sample_rate):
    '''
    download_song: 
        1. download <song_id> from youtube.com
        2. save as audio/<song_id>.mp3
        3. convert to audio/<song_id>.wav (resample at sample_rate too)

    Input: 
    song_id   - (string) DALI song ID 
    dali_info - (DALI object) DALI song information needed to download song (URL, etc.)
    path      - (string) absolute path of destination for audio files
    sample_rate - (int) sampling rate used when converting to .wav

    Return:
    full path to song
    '''
    basename = audio_path + '/' + song_id

    # download...
    errors = dali_code.get_audio(dali_info, audio_path, skip=[], keep=[song_id])
    print(errors)
    i = 0

    # wait for download to finish...
    while utilities.check_file(basename + '.mp3') != True and i < 10:
        time.sleep(1)
        i += 1
        print('Waiting:',i,'seconds...')

    # convert to .wav and resample to sample_rate
    print('Creating', basename + '.wav')
    subprocess.call(['ffmpeg', '-i', basename + '.mp3', '-ar', str(sample_rate), basename + '.wav'])

    # remove mp3...
    print('Removing', basename + '.mp3')
    os.remove(basename + '.mp3')
    return (audio_path+'/'+basename+'.wav')


def save_samples_wav(song_id, audio_path, id_num, x, window_samples, sr):
    '''
    Inputs:
    song_id       (string) DALI song ID, used as basename
    id_num        (int) used to append to the name of the file.  can be something specific like line number
                         or just an arbitrary counter.
    x             (numpy array) song samples of entire song
    window_samples  (tuple) (start,end) window of samples to save to disk
    dest          (string) folder to save to

    Return:
    filename with no path      (string)
    absolute path and filename (string)
    '''
    n = x.shape[0]
    if id_num < 10:
        id_str = str(id_num).zfill(2)
    elif id_num < 100:
        id_str = str(id_num).zfill(1)
    else:
        id_str = str(line_num)

    basename = audio_path + '/' + song_id + '_' + id_str
    start = window_samples[0]
    term = window_samples[1]
    
    print('Writing:', basename + '.wav,',window_samples)
    sf.write(basename + '.wav', x[start:term], sr)
    return (basename + '.wav'), ('audio/' + song_id + '_' + id_str + '.wav')

    # n%window crops the file to a multiple of the window
    # reshape creates a 2-d matrix with rows being number of chunks and 
    # columns being the window (i.e. number of samples)
    # out = np.reshape(x[:-(n%samples)],(np.floor(n/samples).astype(int),samples))

def append_transcript_nemo(json_filename,audio_filename,duration,transcript):
    '''
    append_transcript: save in nemo manifest format
       json_filename:  filename for appending
       audio_filename: absolute path to audio file
       duration:      length of song at full_filename
       transcript:    lyrics corresponding to full_filename
    '''
    jsonfile = open(json_filename, 'a')
    line_format = "{}\"audio_filepath\": \"{}\", \"duration\": {}, \"text\": \"{}\"{}"
    jsonfile.write(line_format.format(
            "{", audio_filename, duration, transcript, "}\n"))
    jsonfile.close()
    return

def crop_song(song_id, audio_path, dali_entry, win_samples):
    '''
    crop_song - takes a DALI song and crops it m times into win_length samples.

    Inputs: 
       song_id    - DALI song id
       entry      - DALI data entry
       audio_path - absolute path where audio files are stored (read/write)
       win_samples - number of samples for each crop
    Return:
       song_ndx   - (m,start_sample,stop_sample) indices for the m crops
       filename_list - absolute path for filenames saved with save_samples_wav

    1. load song with librosa
    2. calculate indices (i.e. sample index starting at 0) for windows of chunks. 
       a. 'win_rate' is win_length/2
       b. do not keep parts of the song less than win_length
    3. crop according to indices and save to audio_path in the form
        audio/<song_id>_##.wav 
        where ## is the number of chunks in the song.
    '''
    x, sr = librosa.load(audio_path + '/' + song_id + '.wav', sr=None)
    n   = np.arange(x.shape[0])  # counter from 0 to max samples of x
    ndx = n[::win_samples][:-1] # takes every win_samples of n, then remove the last sample
    start_ndx = np.reshape(ndx,(ndx.shape[0],1))
    end_ndx   = start_ndx+win_samples
    l = start_ndx.shape[0]


    filename_list = []
    for i in range(l):
        filename_list.append( save_samples_wav(song_id, audio_path, i, x, (start_ndx[i][0],end_ndx[i][0]), sr)[0] )
    return np.concatenate((start_ndx,end_ndx),axis=1), filename_list


def download_convert_song(song_id, dali_data, dali_info, audio_path):
    print('Downloading song: ', song_id)
    # download random song to disk
    start = time.time()
    # download_song(dali_info,audio_path,song_id)
    terminate = time.time()
    print('Download/converted 1 song in', terminate - start, ' seconds.')

    start = time.time()
    # read random song into memory
    # By default, librosa resamples on load to 22K, to load with
    # native sampling rate, use sr=None
    # https://github.com/librosa/librosa/issues/509
    x, sr = librosa.load(audio_path + '/' + song_id + '.wav', sr=None)
    terminate = time.time()
    print('Reloaded 1 song in', terminate - start, ' seconds.')
    if sr != 48000:
        print("Skipping", song_id, "sampling rate is not 48K")
        return False

    # get the line information
    entry = dali_data
    annot = entry.annotations['annot']['lines']
    # magic line from the README.md to fetch all the lines and put them
    # in a nice array.
    lines_time = [i['time'] for i in dali_code.annot2frames(annot, 1 / sr)]
    nlines = len(lines_time)

    durations = [i['time'][1] - i['time'][0] for i in annot]
    line_text = [i['text'] for i in annot]
    lyrics_cleaned = [CleanUpLyrics(t) for t in line_text]
    assert (len(durations) == nlines), " ".join(
        ["Mismatching time lengths:", str(len(durations)), str(nlines)])
    assert (len(line_text) == nlines), " ".join(
        ["Mismatching time and text length:", str(len(line_text)), str(nlines)])

    pattern = re.compile("[a-z ']+")
    for lyric in lyrics_cleaned:
        if not pattern.fullmatch(lyric):
            print("Lyric has non-alphabetic characters: ", lyric)
            return False

    start = time.time()
    jsonfile = open('batch.json', 'a')
    line_format = "{}\"audio_filepath\": \"{}\", \"duration\": {}, \"text\": \"{}\"{}"
    for line_num in range(nlines):
        full_filename, rel_filename = save_samples_wav(audio_path, song_id, line_num, x, lines_time[line_num], sr)
        # filesize = os.stat(full_filename).st_size
        jsonfile.write(line_format.format(
            "{", full_filename, durations[line_num], lyrics_cleaned[line_num], "}\n"))
    jsonfile.close()
    terminate = time.time()
    print('Time to save line files:', terminate - start, 'seconds.')
    return True


def is_song_trainable(dali_data):
    '''
    filter out songs by metadata
    - only do english lyrics for now. (~75% of DALI dataset)
    - TODO: blacklist (i.e. load a list of songs to ignore) poorly transcribed songs
            (manual analysis)

    '''
    language = dali_data.info['metadata']['language']
    if language != 'english':
        print('is_song_trainable: NOT ENGLISH')
        return False
    
    #annotation type is generic to all labels (AFAIK)
    #
    #annot_type = dali_data.annotations['type']
    #if annot_type != 'horizontal':
    #    return False
    
    #number of lines should be ok as long as the songs are 
    # shuffled when performing training.  its possible 
    # they are signifcantly longer than other songs, there
    # would be some overfit to the distribution, but that
    # would have to be pretty large.
    #annot = dali_data.annotations['annot']['lines']
    #if len(annot) > 100:
    #    return False
    #
    # TODO: Possibly filter out more songs, like those that have long instrumental sections
    # print(dali_data.annotations.keys())
    # print(dali_data.annotations['annot_param'])
    # print(dali_data.annotations)
    # print(dali_data)
    # print([(i['time'], i['text']) for i in annot])
    # print([(i['time'], i['text']) for i in dali_code.annot2frames(annot, 1/22050)])
    return True

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

def preprocess_song(song_id, dali_path, audio_path, dali_info, nemo_manifest_filename, sample_rate):
    '''
    
    '''
    win_size = 10.2268
    win_samples = np.floor(win_size * sample_rate).astype(int)
    print('window samples: ',win_samples)

    dali_entry = dali_code.get_the_DALI_dataset(dali_path, keep=[song_id])[song_id]
    if is_song_trainable(dali_entry):

        # get song and save to audio_path
        download_song(song_id, dali_info, audio_path, sample_rate)

        # slice up song and save to audio_path, return indices of samples
        song_ndx,filename_list = crop_song(song_id, audio_path, dali_entry, win_samples)
        np.save(audio_path+'/'+song_id+'_crop_indices.npy',song_ndx)

        # slice up the transcript for each cropped version of the song
        dali_annot = dali_entry.annotations['annot']
        transcript_list = dali_helpers.get_cropped_transcripts(dali_annot,song_ndx,sample_rate)

        # save all cropped files in nemo format
        for i in range(len(transcript_list)):
            append_transcript_nemo(nemo_manifest_filename,filename_list[i],win_size,transcript_list[i])

        

        return True

    return False

def main():
    '''
    print("Import a single song and chop into chunks at the line level...")
    '''
    # 1. select a random song from the dali dataset, 
    # 2. create a set overlapping .wav files  from the song
    # 3. create a set of transcripts for each .wav file.
    allsongfilenames = utilities.get_files_path(dali_path, '.gz')
    song_id = ''
    n = len(allsongfilenames)
    num_skipped_after_download = 0

    i = np.random.randint(n)
    song_id = os.path.relpath(allsongfilenames[i], dali_path).split('.')[0]
    entry = dali_code.get_the_DALI_dataset(dali_path, keep=[song_id])[song_id]
    while True:
        if is_song_trainable(dali_data):
            converted = download_convert_song(song_id, entry, dali_info, audio_path)
            if converted:
                break
            num_skipped_after_download += 1
            if num_skipped_after_download >= 10:
                print("Too many songs downloaded but not converted; aborting ... ")
                break


if __name__ == '__main__':
    '''
    choose a random song, crop audio files, and massage transcripts
    '''
    dali_path, audio_path, dali_info = dali_setup()
    allsongfilenames = utilities.get_files_path(dali_path, '.gz')
    n = len(allsongfilenames)

    #number of samples assumes 22kHz (Stoller) - 10.23s
    sample_rate = 22050 

    #choose a random song...
    i = np.random.randint(n)

    song_id = os.path.relpath(allsongfilenames[i], dali_path).split('.')[0]

    dali_entry = dali_code.get_the_DALI_dataset(dali_path, keep=[song_id])[song_id]
    print('choosing index:',i,', title:',dali_entry.info['title'])
    
    #preprocess a song
    nemo_manifest_filename = 'dali_training.json'
    if not preprocess_song(song_id, dali_path, audio_path, dali_info, nemo_manifest_filename, sample_rate):
        print('ERROR PREPROCESSING.')



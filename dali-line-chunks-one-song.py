import matplotlib.pyplot as plt
import numpy as np
import sys, os, time, subprocess
import librosa
import soundfile as sf
import pdb
import re
import DALI as dali_code
from DALI import utilities


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


def download_song(info, dest, song_id):
    basename = dest + '/' + song_id

    # download...
    errors = dali_code.get_audio(info, dest, skip=[], keep=[song_id])
    print(errors)
    i = 0

    # wait for download to finish...
    while utilities.check_file(basename + '.mp3') != True and i < 10:
        time.sleep(1)
        i += 1
        print(i)

    # convert to .wav...
    print('Creating', basename + '.wav')
    subprocess.call(['ffmpeg', '-i', basename + '.mp3', basename + '.wav'])

    # remove mp3...
    print('Removing', basename + '.mp3')
    os.remove(basename + '.mp3')
    return


def save_samples_wav(dest, song_id, line_num, x, line_samples, sr):
    '''
    song_id - string of DALI song ID
    line    - line number
    x - song samples
    line_samples  - tuple (start,end)
    dest - folder to save to
    '''
    n = x.shape[0]
    if line_num < 10:
        line_str = str(line_num).zfill(2)
    elif line_num < 100:
        line_str = str(line_num).zfill(1)
    else:
        print('This thing has more than 100 lines!!')
        line_str = str(line_num)

    basename = dest + '/' + song_id + '_' + line_str
    start = line_samples[0]
    term = line_samples[1]

    print('Writing:', basename + '.wav')
    sf.write(basename + '.wav', x[start:term], sr)
    return (basename + '.wav'), ('audio/' + song_id + '_' + line_str + '.wav')

    # n%window crops the file to a multiple of the window
    # reshape creates a 2-d matrix with rows being number of chunks and 
    # columns being the window (i.e. number of samples)
    # out = np.reshape(x[:-(n%samples)],(np.floor(n/samples).astype(int),samples))


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
    language = dali_data.info['metadata']['language']
    if language != 'english':
        return False
    annot_type = dali_data.annotations['type']
    if annot_type != 'horizontal':
        return False
    annot = dali_data.annotations['annot']['lines']
    if len(annot) > 100:
        return False
    # TODO: Possibly filter out more songs, like those that have long instrumental sections
    # print(dali_data.annotations.keys())
    # print(dali_data.annotations['annot_param'])
    print(dali_data.info)
    # print(dali_data.annotations)
    # print(dali_data)
    # print([(i['time'], i['text']) for i in annot])
    # print([(i['time'], i['text']) for i in dali_code.annot2frames(annot, 1/22050)])
    return True


if __name__ == '__main__':
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

    # ###############################################################
    #
    print("Import a single song and chop into chunks at the line level...")
    #
    # ###############################################################

    # select a random song from the dali dataset
    allsongfilenames = utilities.get_files_path(dali_path, '.gz')
    song_id = ''
    n = len(allsongfilenames)
    num_skipped_after_download = 0
    while True:
        i = np.random.randint(n)
        song_id = os.path.relpath(allsongfilenames[i], dali_path).split('.')[0]
        dali_data = dali_code.get_the_DALI_dataset(dali_path, keep=[song_id])[song_id]
        if is_song_trainable(dali_data):
            converted = download_convert_song(song_id, dali_data, dali_info, audio_path)
            if converted:
                break
            num_skipped_after_download += 1
            if num_skipped_after_download >= 10:
                print("Too many songs downloaded but not converted; aborting ... ")
                break

    """
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
    """

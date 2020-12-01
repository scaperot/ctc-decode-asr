import matplotlib.pyplot as plt
import numpy as np
import sys,os,time,subprocess
import librosa
import soundfile as sf
import pdb
import DALI as dali_code
from DALI import utilities


def download_song(info,dest,song_id):

    basename = dest+'/'+song_id

    #download...
    errors = dali_code.get_audio(info, dest, skip=[], keep=[song_id])
    print(errors)
    i = 0
    
    #wait for download to finish...
    while utilities.check_file(basename+'.mp3') != True and i < 10:
        time.sleep(1)
        i += 1
        print(i)

    #convert to .wav...
    print('Creating',basename+'.wav')
    subprocess.call(['ffmpeg','-i',basename+'.mp3',basename+'.wav'])

    #remove mp3...
    print('Removing',basename+'.mp3')
    os.remove(basename+'.mp3')
    return

def save_samples_wav(dest,song_id,line_num,x,line_samples,sr):
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

    basename = dest+'/'+song_id+'_'+line_str
    start = line_samples[0]
    term  = line_samples[1]

    print('Writing:',basename+'.wav')
    sf.write(basename+'.wav',x[start:term],sr)
    return (basename+'.wav'), ('audio/'+song_id+'_'+line_str+'.wav')

    # n%window crops the file to a multiple of the window
    # reshape creates a 2-d matrix with rows being number of chunks and 
    # columns being the window (i.e. number of samples)
    #out = np.reshape(x[:-(n%samples)],(np.floor(n/samples).astype(int),samples))


if __name__ == '__main__':
    ################################
    # CHECK FOR DALI dataset folder
    ################################
    if os.path.isdir('DALI/'):
        dali_path = os.path.abspath('DALI/')
    else:
        print('DALI dataset not found in',os.path.abspath('.')+'/DALI/')
        sys.exit()
    
    ################################
    # CHECK FOR audio folder
    ################################
    if os.path.isdir('audio/'):
        audio_path = os.path.abspath('audio/')
    else:
        print('audio directory not found, trying to create it.')
        os.makedirs(os.path.abspath('.')+'/audio/')

    csv = open('batch.csv','a')
    
    dali_info = dali_code.get_info(dali_path + '/info/DALI_DATA_INFO.gz')

    # ###############################################################
    #
    print("Import a single song and chop into chunks at the line level...")
    #
    # ###############################################################
    
    #select a random song from the dali dataset
    allsongfilenames = utilities.get_files_path(dali_path,'.gz')
    song_id = ''
    n = len(allsongfilenames)
    i = np.random.randint(n)
    song_id =  os.path.relpath(allsongfilenames[i],dali_path).split('.')[0]

    #download random song to disk
    start = time.time()
    download_song(dali_info,audio_path,song_id)
    terminate = time.time()
    print('Download/converted 1 song in', terminate-start,' seconds.')

    start = time.time()
    #read random song into memory
    x,sr = librosa.load(audio_path+'/'+song_id+'.wav')
    terminate = time.time()
    print('Reloaded 1 song in', terminate-start,' seconds.')

    #get the line information     
    dali_data = dali_code.get_the_DALI_dataset(dali_path,keep=[song_id])
    entry = dali_data[song_id]
    annot = entry.annotations['annot']['lines']
    # magic line from the README.md to fetch all the lines and put them
    # in a nice array.
    lines_time = [i['time'] for i in dali_code.annot2frames(annot,1/sr)]
    line_text  = [i['text'] for i in dali_code.annot2frames(annot,1/sr)]
    nlines = len(lines_time)
    start = time.time()
    for line_num in range(nlines):
        full_filename, rel_filename = save_samples_wav(audio_path,song_id,line_num,x,lines_time[line_num],sr)
        filesize = os.stat(full_filename).st_size
        csv.write(rel_filename+','+str(filesize)+","+line_text[line_num]+'\n')
    csv.close()
    terminate = time.time()
    print('Time to save line files:',terminate-start,'seconds.')

    



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

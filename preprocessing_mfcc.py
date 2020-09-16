import numpy as np
import librosa
import math
import os
import python_speech_features as sf
from sklearn.preprocessing import scale
from glob import glob
from tqdm import tqdm

def audiofile_to_input_vector(wav_filename):
    """
    Returns audio and its transcripts. Audio is preprocessed by MFCC.
    :param wav_filename:
    :param int numcep: feature size vector
    :param int numcontext: not used
    :return: ndarray of shape (numcep, num_vectors)
    """
    # load wav file and downsamples it to 16khz
    signal, sample_rate = librosa.load(wav_filename, sr=None)

    if sum(signal) == 0:
        print(wav_filename)


    # Applying mffc transformation to get feature vector

    mfcc_features = sf.mfcc(signal, 16000)


    # normalization?
    mfcc_features = scale(mfcc_features, axis=1)

    return mfcc_features

def main():

    input_dir = 'test_set'
    output_dir = 'data_preprocessed/' + input_dir
    files = glob(input_dir + '/*')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in tqdm(range(len(files))):

        audio_path = files[i]
        f_name = audio_path.split('/')[-1][:-4]
        data = audiofile_to_input_vector(audio_path)
        l = len(data)
        np.save('%s/%s.npy'%(output_dir, f_name), data)


if __name__ == '__main__':
    main()

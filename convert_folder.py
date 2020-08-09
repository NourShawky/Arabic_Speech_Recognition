from pydub import AudioSegment
from glob import glob
from tqdm import tqdm

input_dir = '/home/nour/Quran_challenge/data/train_set'
output_dir = '/home/nour/Quran_challenge/train_set'
f_list = glob(input_dir + '/*')

for l in tqdm(f_list):

    sound = AudioSegment.from_mp3(l)
    sound.export("%s/%s.wav" % (output_dir, l.split('/')[-1][:-4]), format="wav")

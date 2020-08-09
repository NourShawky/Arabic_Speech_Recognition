import csv
import numpy as np

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0

names_list = []

arabic_letters = ["ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص","ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي","أ","إ","آ","ى","ئ","ء","ة",'َ', 'ّ', 'ِ', 'ْ', 'ُ', 'ٍ', 'ٌ', 'ً']
letters_dict = {}

for i in range(len(arabic_letters)):
    letters_dict[arabic_letters[i]] = i + 1

print(letters_dict)
labels = []
large_files = open('/home/nour/Quran_challenge/large_files_names.txt').readlines()
large_files = [x[:-1] for x in large_files]


def chars_to_index(char_text):
    """
    Convert chars to asci indexes
    :param list char_text: array of characters
    :return: array of ascii indexes
    """
    l = [SPACE_INDEX if x == SPACE_TOKEN else letters_dict.get(x,-1) for x in char_text]
    l = [x for x in l if x != -1 ]
    return np.asarray(l)

def text_to_chars(text):
    """
    Convert text to characters
    :param str text: string text
    :return: array of characters
    """
    # replace space(' ') on two spaces
    refactor_text = text.replace(' ', '  ')

    # splits by words and spaces to array ['hello', '', 'how', '', 'are' ...]
    refactor_text = refactor_text.split(' ')

    # crates array of chars and instead of space('') puts <space> token
    refactor_text = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in refactor_text])

    return refactor_text


with open('/home/nour/Quran_challenge/data/train_transcriptions3.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first = True
    counter = 0
    for row in csv_reader:
        if first:
            first = False

        else:
            n = row[0]
            if n in large_files:
                large_files.remove(n)
                continue

            names_list.append(n)
            l = text_to_chars(row[2])
            l = chars_to_index(l)


            l = np.array(l)
            labels.append(l)


            counter += 1
            print(counter)

names_list = np.array(names_list)
np.save('data/labels_small.npy', labels, allow_pickle=True)
np.save('data/names_small.npy', names_list, allow_pickle=True)

import tensorflow as tf
import numpy as np
from glob import glob
import os
import csv
from tqdm import tqdm


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):

    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)


    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


def load_batch(data):

    data_list = []
    names_list = []

    for n in range(len(data)):
        f = data[n]
        name = f.split('/')[-1][:-4]
        data_list.append(np.load(f))
        names_list.append(name)

    data_list = np.array(data_list)
    x, x_length = pad_sequences(data_list)

    return x, x_length, names_list



def main():

    batch_size = 8
    arabic_letters = ["ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص","ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي","أ","إ","آ","ى","ئ","ء","ة",'َ', 'ّ', 'ِ', 'ْ', 'ُ', 'ٍ', 'ٌ', 'ً']
    letters_dict = {}
    letters_dict[0] = ' '

    for i in range(len(arabic_letters)):
        letters_dict[i + 1] = arabic_letters[i]

    letters_dict[45] = ''

    # Constants
    base_dir = '/home/nour/Quran_challenge/data_preprocessed2/test_set'
    data_list = glob(base_dir + '/*')
    data_list.sort()


    with open('res_file_11.csv', mode='w') as res_file:
        res_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        res_writer.writerow(['audio_id', 'text'])

        with tf.Session() as sess:

            saver = tf.train.import_meta_graph('/home/nour/Quran_challenge/model_small11/backup_model.ckpt.meta')
            saver.restore(sess, '/home/nour/Quran_challenge/model_small11/backup_model.ckpt')

            inputs = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
            seq_len = tf.get_default_graph().get_tensor_by_name('Placeholder_5:0')
            batch_size_placeholder = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
            indices_placeholder = tf.get_default_graph().get_tensor_by_name('CTCBeamSearchDecoder:0')
            values_placeholder = tf.get_default_graph().get_tensor_by_name('CTCBeamSearchDecoder:1')


            num_batch = len(data_list) // batch_size




            for i in tqdm(range(num_batch)):

                data = data_list[i * batch_size : (i + 1) * batch_size]


                val_inputs, val_seq_len, names_list = load_batch(data)

                val_feed = {inputs: val_inputs,
                            batch_size_placeholder: batch_size,
                            seq_len: val_seq_len}

                indices, values = sess.run([indices_placeholder, values_placeholder], feed_dict=val_feed)

                pred_list = [[] for i in range(batch_size)]


                for i in range(len(indices)):
                    pred_list[indices[i][0]].append(letters_dict[values[i]])


                for i in range(batch_size):
                    str1 = ""
                    name = names_list[i]

                    for ele in pred_list[i]:
                        str1 += ele

                    res_writer.writerow([name, str1])



if __name__ == '__main__':

  main()

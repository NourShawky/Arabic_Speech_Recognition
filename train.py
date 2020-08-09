import tensorflow as tf
import numpy as np
import time
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', default='/home/nour/Quran_challenge/data/train_small.npy')
parser.add_argument('--val_file', default='/home/nour/Quran_challenge/data/val_small.npy')
parser.add_argument('--data_dir', default='/home/nour/Quran_challenge/data_preprocessed2/train_set')
parser.add_argument('--log_dir', default='model_small12')
parser.add_argument('--pretrained', default='/home/nour/Quran_challenge/model_small11/backup_model.ckpt')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--opt', default='adam', type=str, help='optimizer type')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')

def sparse_tuple_from(sequences, dtype=np.int32):

    indices = []
    values = []


    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)



    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

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
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
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

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]




def load_batch(data, labels, base_dir):

    data_list = []
    label_list = []


    for n in range(len(data)):
        data_list.append(np.load('%s/%s.npy'%(base_dir, data[n][0])))
        label_list.append(labels[n][0].tolist())

    data_list = np.array(data_list)

    y_sparse = sparse_tuple_from(label_list)
    x, x_length = pad_sequences(data_list)



    return y_sparse, x, x_length


def main(args):

    base_dir = args.data_dir
    training_list = np.load(args.train_file, allow_pickle=True)
    training_data, training_labels = np.hsplit(training_list, 2)

    val_list = np.load(val_file, allow_pickle=True)
    val_data, val_labels = np.hsplit(val_list, 2)

    learning_rate_steps = []

    dropout_keep_prob = args.dropout_keep_prob

    log_dir = args.log_dir

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_file = open('%s/train_log.txt' % (log_dir), 'a')

    def log_func(out_str=''):
      log_file.write(out_str+'\n')
      log_file.flush()
      print(out_str)


    # Some configs
    num_features = 13
    num_classes = 45

    # Hyper-parameters
    num_epochs = args.num_epochs
    num_hidden = 1024
    num_layers = 2
    is_stack = True
    batch_size = args.batch_size
    learning_rate = args.lr
    momentum = 0.9

    num_examples = len(training_data)
    num_batches_per_epoch = num_examples // batch_size

    num_val_examples = len(val_data)
    num_val_batches = num_val_examples // batch_size

    with tf.Graph().as_default() as graph:

        inputs = tf.placeholder(tf.float32, [None, None, num_features])
        batch_size_placeholder = tf.placeholder(tf.int32, shape=None)
        targets = tf.sparse_placeholder(tf.int32)
        seq_len = tf.placeholder(tf.int32, [None])



        cells = []
        for _ in range(num_layers):

            cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
            cells.append(cell)


        stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        outputs, state = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]

        outputs = tf.reshape(outputs, [-1, num_hidden])

        W = tf.Variable(tf.truncated_normal([num_hidden,
                                             num_classes],
                                            stddev=0.1))

        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        logits = tf.matmul(outputs, W) + b

        logits = tf.reshape(logits, [batch_size_placeholder, -1, num_classes])


        logits = tf.transpose(logits, (1, 0, 2))

        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)

        step = tf.Variable(0)

        learning_rate_placeholder = tf.placeholder(tf.float32, shape=None)

        if args.opt == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate_placeholder,
                                                   0.9, use_nesterov=True)

        elif args.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate_placeholder)

        gradients, variables = zip(*optimizer.compute_gradients(cost))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimize = optimizer.apply_gradients(zip(gradients, variables))

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)


        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))
        saver = tf.train.Saver()

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            least_loss = 100000
            least_ler = 100000
            start_batch = 0
            start_epoch = 0

            if args.pretrained:
                saver.restore(session, args.pretrained)
                start_batch = 2497
                start_epoch = 12



            for curr_epoch in range(start_epoch, num_epochs):
                log_func("------ start epoch: %d ------" % curr_epoch)

                training_data, training_labels = unison_shuffled_copies(training_data, training_labels)
                train_cost = train_ler = val_cost = val_ler = 0


                if curr_epoch in learning_rate_steps:
                    learning_rate /= 10

                start = time.time()
                print(learning_rate)

                for batch in range(start_batch, num_batches_per_epoch):


                    data = training_data[batch * batch_size : (batch + 1) * batch_size, :]
                    labels = training_labels[batch * batch_size : (batch + 1) * batch_size, :]

                    train_targets, train_inputs, train_seq_len = load_batch(data, labels, base_dir)



                    feed = {inputs: train_inputs,
                            targets: train_targets,
                            seq_len: train_seq_len,
                            batch_size_placeholder: batch_size,
                            learning_rate_placeholder: learning_rate}

                    batch_cost, batch_ler, _ = session.run([cost, ler, optimize], feed_dict=feed)


                    train_cost += batch_cost
                    train_ler += batch_ler


                    if batch % 1 == 0:
                        log_func("Batch [%d / %d] train_cost: %f, train_ler: %f time: %f" % (batch + 1, num_batches_per_epoch, batch_cost, batch_ler, time.time() - start))
                        start = time.time()
                        save_path = saver.save(session, "%s/backup_model.ckpt" % (log_dir))

                train_cost /= num_batches_per_epoch
                train_ler /= num_batches_per_epoch
                start_batch = 0

                log_func('------------------- Start Validation -----------------------')


                for val_batch in range(num_val_batches):

                    data = val_data[val_batch * batch_size : (val_batch + 1) * batch_size, :]
                    labels = val_labels[val_batch * batch_size : (val_batch + 1) * batch_size, :]

                    val_targets, val_inputs, val_seq_len = load_batch(data, labels, base_dir)

                    val_feed = {inputs: val_inputs,
                                targets: val_targets,
                                batch_size_placeholder: batch_size,
                                seq_len: val_seq_len}


                    val_batch_cost, val_batch_ler = session.run([cost, ler], feed_dict=val_feed)

                    val_cost += val_batch_cost
                    val_ler += val_batch_ler

                    if val_batch % 100 == 0:

                        log_func("Batch [%d / %d] val_cost: %f, val_ler: %f" % (val_batch + 1, num_val_batches, val_batch_cost, val_batch_ler))

                val_cost /= num_val_batches
                val_ler /= num_val_batches

                if val_cost < least_loss and val_ler < least_ler:
                      save_path = saver.save(session, "%s/best_model.ckpt" % (log_dir))
                      log_func("******************************************* Best Model ********************************************")


                log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"

                log_func(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                                 val_cost, val_ler, time.time() - start))


if __name__ == '__main__':

  args = parser.parse_args()
  main(args)

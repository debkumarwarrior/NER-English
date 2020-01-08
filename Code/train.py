from model import BiLSTMModel
import os,argparse,traceback
import sys

from shutil import copyfile as copy_file
from utils import  load_hparams,save_hparams,get_time
from hparams import hparams, hparams_debug_string
import tensorflow as tf
import numpy as np

from data_helpers import DataHelper

tf.logging.set_verbosity(tf.logging.ERROR)

def read_data(file_path):
    import re
    tokens = []
    tags = []

    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            token = re.sub(r'http[s]?://[^ ]{1,}', '<URL>', token)
            token = re.sub(r'^@[A-Za-z]{1,}', '<USR>', token)

            if re.match(r'<usr>', token):
                token = '<USR>'
            if re.match(r'[A-Za-z]{3}.[A-Za-z]{1,}.[A-Za-z]{3}',token):
                token = '<URL>'
            tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags

def read_csv(filename):
    import pandas as pd
    df = pd.read_csv(filename, encoding='utf-8')
    df.fillna(False, inplace=True)
    return df

def read_data_from_csv(df):
    import re
    tokens = []
    tags = []

    sentences = []
    labels = []
    senetence_count = 0

    for i in range(len(df['Tags'])):
        if senetence_count >= 500000:
            continue

        tag = df['Tags'][i]
        token = df['Sentences'][i]

        if tag == False or token == False:
            continue

        if re.match(r'\\n',token):
            tokens.append(sentences)
            tags.append(labels)
            senetence_count += 1

            sentences = []
            labels = []
        else:
            token = re.sub(r'http[s]?://[^ ]{1,}', '<URL>', token)
            token = re.sub(r'^@[A-Za-z]{1,}', '<USR>', token)

            if re.match(r'<usr>', token):
                token = '<USR>'
            if re.match(r'[A-Za-z]{3}.[A-Za-z]{1,}.[A-Za-z]{3}',token):
                token = '<URL>'
            sentences.append(token)
            labels.append(tag)

    return tokens, tags


class Train(object):
    def __str__(self):
        return 'Training data on {}'.format(self.config.ftrain)

    def prepare_dirs(self, hparams):
        dataset_desc = 'NER'
        self.config.model_name = "{}_{}".format(dataset_desc, get_time())
        self.config.model_dir = os.path.join(self.config.log_dir, self.config.model_name)

        for path in [self.config.log_dir, self.config.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        save_hparams(self.config.model_dir, hparams)
        copy_file("hparams.py", os.path.join(self.config.model_dir, "hparams.py"))

    def prepare_data(self):
        self.config.ftrain = os.path.abspath(os.path.join(self.config.data_paths,'train.txt'))
        self.config.fval = os.path.abspath(os.path.join(self.config.data_paths,'validation.txt'))
        self.config.ftest = os.path.abspath(os.path.join(self.config.data_paths,'test.txt'))

        if not os.path.exists(self.config.ftrain) or not os.path.exists(self.config.fval) or not os.path.exists(self.config.ftest):
            raise FileExistsError

        self.train_tokens,self.train_tags = read_data(self.config.ftrain)
        self.validation_tokens,self.validation_tags = read_data(self.config.fval)
        self.test_tokens,self.test_tags = read_data(self.config.ftest)

    def train(self):
        checkpoint_path = os.path.join(self.config.model_dir, 'model.ckpt')
        data_dirs = self.config.data_paths
        data_helper = DataHelper(self.train_tokens, self.validation_tokens, self.train_tags)
        data_helper.dump_data_to_file(self.config.model_dir)

        tf.reset_default_graph()
        print('=' * 50)
        print('=' * 50)
        print(' [*] Checkpoint path: %s' % checkpoint_path)
        print(' [*] Loading training data from: %s' % data_dirs)
        print(' [*] Using model: %s' % self.config.model_dir)
        print(hparams_debug_string())

        token_count = data_helper.gettokencount()
        tag_count = data_helper.gettagcount()
        cap_feat_count = 4
        pad_index = data_helper.word2idx('<PAD>')

        print('Tag count : %d' % tag_count)
        print('Token count : %d' % token_count)
        print('Cap Features count : %d' % cap_feat_count)
        print('PAD index : %d' % pad_index)

        with tf.Graph().as_default() as graph:
            sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

            with tf.Session(config=sess_config) as sess:
                try:

                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=2)
                    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
                    sess_config.gpu_options.allow_growth = True

                    with tf.variable_scope('model'):
                        model = BiLSTMModel()
                        model.init()
                        train_vars = model.build_layers(hparams=hparams, vocabulary_size=token_count,
                                                        n_cap_feats=cap_feat_count,
                                                        n_tags=tag_count)
                        model.compute_predictions()
                        model.compute_loss(n_tags=tag_count, PAD_index=pad_index)
                        model.perform_optimization(global_step)
                        train_stat = model.add_stats()

                        print('No. of trainable variables : %d' % train_vars)

                        sess.run(tf.global_variables_initializer())
                        summary_writer = tf.summary.FileWriter(self.config.model_dir)
                        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

                        start_step = sess.run(global_step)
                        print('-' * 20 + ' Starting new training ' + '-' * 20)
                        learning_rate = self.config.learning_rate

                        for epoch in range(self.config.n_epochs):
                            print('-' * 20 + ' Epoch {}'.format(epoch + 1) + '-' * 20)

                            for x_batch, c_batch, y_batch, lengths in data_helper.batches_generator(self.config.batch_size,
                                                                                           self.train_tokens,
                                                                                           self.train_tags):
                                step, loss = model.train_on_batch(sess, global_step, x_batch, c_batch, y_batch, lengths,
                                                                  learning_rate, self.config.dropout_keep_probability)
                                if step % 100 == 0:
                                    print('[Step : %d] loss : %f' % (step, loss))

                            learning_rate = learning_rate / self.config.learning_rate_decay

                            if (epoch + 1) % self.config.checkpoint_interval == 0:
                                print('Saving checkpoint to : %s-%d' % (checkpoint_path, epoch + 1))
                                saver.save(sess, checkpoint_path, global_step=epoch)

                            if (epoch + 1) % self.config.test_interval == 0:
                                print('Train data evaluation')
                                data_helper.eval_conll(model, sess, self.train_tokens, self.train_tags,
                                                       short_report=True)

                                print('Validation data evaluation:')
                                data_helper.eval_conll(model, sess, self.validation_tokens, self.validation_tags,
                                                       short_report=True)

                except Exception as e:
                    print('Exitin due to exception : [%s]!!!' % e)
                    traceback.print_exc()
                    log.close()
                    return

                print('-' * 20 + ' Training completed! ' + '-' * 20)
                print('Training data evaluation:')
                data_helper.eval_conll(model,sess,self.train_tokens,self.train_tags,short_report=False)

                print('Validation data evaluation:')
                data_helper.eval_conll(model,sess,self.validation_tokens,self.validation_tags,short_report=False)

                print('Training data evaluation:')
                data_helper.eval_conll(model,sess,self.test_tokens,self.test_tags,short_report=True)
        return

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--log_dir', default='log-dir')
        parser.add_argument('--data_paths', required=True)

        parser.add_argument('--batch_size', default=32)
        parser.add_argument('--n_epochs', default=10)
        parser.add_argument('--learning_rate', default=0.005)
        parser.add_argument('--learning_rate_decay', default=np.sqrt(2.0))
        parser.add_argument('--dropout_keep_probability', default=0.9)
        parser.add_argument('--summary_interval', type=int, default=10)
        parser.add_argument('--test_interval', type=int, default=5)
        parser.add_argument('--checkpoint_interval', type=int, default=10)
        parser.add_argument('--test_size', default=0.2)

        self.config = parser.parse_args()
        self.prepare_dirs(hparams=hparams)
        log_path = os.path.join(self.config.model_dir, 'train.log')
        sys.stdout = open(log_path,'w+')

class TrainImpl(Train):
    def __str__(self):
        return 'Train Impl on : {}'.format(self.config.traincsv)

    def prepare_data(self):
        self.config.traincsv = os.path.abspath(os.path.join(self.config.data_paths,'EWNERTC.csv'))

        if not os.path.exists(self.config.traincsv):
            raise FileNotFoundError

        from sklearn.model_selection import train_test_split
        df = read_csv(self.config.traincsv)
        tokens,tags = read_data_from_csv(df)

        self.train_tokens,self.validation_tokens,self.train_tags,self.validation_tags = \
            train_test_split(tokens,tags,test_size=self.config.test_size)

        print('Reading training data files completed.')

if __name__ == '__main__':
    try:
        train = Train()
        train.prepare_data()
        train.train()
    except FileNotFoundError as e:
        print('File not found error : [%s]' % e)
        exit(-1)
    except Exception as e:
        print('Unknown exception : [%s]' % e)
        traceback.print_exc()
        exit(-1)
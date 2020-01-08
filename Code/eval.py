import tensorflow as tf
import argparse,os
from data_helpers import DataHelper
from evaluation import precision_recall_f1

class Evaluator(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--data_paths', required=True)
        parser.add_argument('--checkpoint_path', required=True)

        self.config = parser.parse_args()

    def prepare_date(self):
        from train import read_data_from_csv,read_csv,read_data

        testfile = os.path.join(os.path.abspath(self.config.data_paths),'test.txt')
        self.test_token,self.test_tags = read_data(testfile)

        self.data_helper = DataHelper(None,None,None,isTrain=False)
        self.data_helper.load_data_from_file(self.config.checkpoint_path)

    def evaluate(self):
        self.checkpoint_file = tf.train.latest_checkpoint(self.config.checkpoint_path)

        sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Graph().as_default() as graph:
            with tf.Session(config=sess_config) as sess:
                saver = tf.train.import_meta_graph('{}.meta'.format(self.checkpoint_file))
                saver.restore(sess,self.checkpoint_file)

                word_batch = graph.get_operation_by_name('model/placeholders/word_batch').outputs[0]
                cap_feat_batch = graph.get_operation_by_name('model/placeholders/cap_feat_batch').outputs[0]
                lengths_batch = graph.get_operation_by_name('model/placeholders/lengths').outputs[0]

                predictions = graph.get_operation_by_name('model/output/predictions').outputs[0]

                y_true, y_pred = [], []
                for x_batch, c_batch, y_batch, l_batch in self.data_helper.batches_generator(1, self.test_token,
                                                                                             self.test_tags):
                    tags_idxs_batch = sess.run(predictions,feed_dict ={
                        word_batch : x_batch,
                        cap_feat_batch : c_batch,
                        lengths_batch : l_batch
                    })

                    tags_batch, tokens_batch = [],[]
                    for tag_idxs,token_idxs in zip(tags_idxs_batch,x_batch):
                        tags,tokens = [],[]
                        for tag_idx,token_idx in zip(tag_idxs,token_idxs):
                            tags.append(self.data_helper.idx2tag[tag_idx])
                            tokens.append(self.data_helper.idx2token[token_idx])

                        tags_batch.append(tags)
                        tokens_batch.append(tokens)

                    predicted_tags = []
                    ground_truth_tags = []

                    for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]):
                        if token != '<PAD>':
                            ground_truth_tags.append(self.data_helper.idx2tag[gt_tag_idx])
                            predicted_tags.append(pred_tag)

                    # We extend every prediction and ground truth sequence with 'O' tag
                    # to indicate a possible end of entity.
                    y_true.extend(ground_truth_tags + ['O'])
                    y_pred.extend(predicted_tags + ['O'])

                results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=False)

if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.prepare_date()
    evaluator.evaluate()
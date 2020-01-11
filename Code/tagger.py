import tensorflow as tf
import numpy as np
import argparse
import falcon


from data_helpers import DataHelper

class Tagger:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path', required=True)
        parser.add_argument('--port',type=int,default=8991)
        self.config = parser.parse_args()
        tf.reset_default_graph()

    def create_test_session(self):
        self.checkpoint_file = tf.train.latest_checkpoint(self.config.checkpoint_path)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.import_meta_graph('{}.meta'.format(self.checkpoint_file))
        saver.restore(self.sess, self.checkpoint_file)

        self.graph = tf.get_default_graph()
        self.word_batch= self.graph.get_operation_by_name('model/placeholders/word_batch').outputs[0]
        self.cap_feat_batch = self.graph.get_operation_by_name('model/placeholders/cap_feat_batch').outputs[0]
        self.lengths_batch = self.graph.get_operation_by_name('model/placeholders/lengths').outputs[0]

        self.predictions = self.graph.get_operation_by_name('model/output/predictions').outputs[0]
        self.data_helper = DataHelper(None, None, None, isTrain=False)
        self.data_helper.load_data_from_file(self.config.checkpoint_path)

    def gettags(self, text):
        import re
        text = re.sub(r'([A-Za-z])([!,:;])',r'\1 \2', text)
        tokens = text.strip().split()

        for token in tokens:
            token = re.sub(r'http[s]?://[^ ]{1,}', '<URL>', token)
            token = re.sub(r'^@[A-Za-z]{1,}', '<USR>', token)

            if re.match(r'<usr>', token):
                token = '<USR>'
            if re.match(r'[A-Za-z]{3}.[A-Za-z]{1,}.[A-Za-z]{3}',token):
                token = '<URL>'

        token_idxs = np.array([self.data_helper.words2idxs(tokens)])
        token_capfeats = np.array([self.data_helper.token2cap_feats(tokens)])

        lengths = np.array([len(tokens)])

        print ('Shape of token idxs : {}'.format(token_idxs.shape))
        print ('Shape of token capfeats : {}'.format(token_capfeats.shape))
        print ('Shape of token lengths : {}'.format(lengths.shape))

        tag_idxs = self.sess.run(self.predictions,feed_dict={self.word_batch:token_idxs,
                                                             self.cap_feat_batch: token_capfeats,
                                                             self.lengths_batch:lengths})

        print ('Shape of tag idxs : {}'.format(tag_idxs.shape))

        ret = []
        for words,tags in zip(token_idxs,tag_idxs):
            for word,tag in zip(tokens,self.data_helper.idxs2tags(tags)):
                ret.append([word,tag])
                print('{} --- {}'.format(word,tag))

        return ret

    def on_get(self,req,res):
        if not req.params.get('text'):
            raise falcon.HTTPBadRequest()

        result = self.gettags(req.params.get('text'))
        print(result)

        res.media = result

if __name__ == '__main__':
    from wsgiref import simple_server

    tagger = Tagger()
    tagger.create_test_session()
    api = falcon.API()
    api.add_route('/gettag',tagger)

    print('Serving on port : [%d]' % tagger.config.port)

    simple_server.make_server('0.0.0.0',tagger.config.port,api).serve_forever()






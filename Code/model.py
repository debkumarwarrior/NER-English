import tensorflow as tf
import numpy as np

class BiLSTMModel:
    def init(self):
        with tf.variable_scope('placeholders'):
            # Placeholders for input and ground truth output.
            self.word_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_batch')
            self.cap_feat_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='cap_feat_batch')
            self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                            name='ground_truth_tags')
            # Placeholder for lengths of the sequences.
            self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')

            # Placeholder for a dropout keep probability. If we don't feed
            # a value for this placeholder, it will be equal to 1.0.
            self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])

            # Placeholder for a learning rate (tf.float32).
            self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[],
                                                   name='learning_rate_ph')


    def build_layers(self, hparams, vocabulary_size, n_cap_feats, n_tags):
        with tf.variable_scope('word_embedding'):
            initial_embedding_matrix = np.random.randn(vocabulary_size, hparams.word_dim) / np.sqrt(hparams.word_dim)
            embedding_matrix_variable = tf.Variable(initial_value=initial_embedding_matrix,
                                                    name='embeddings_matrix',
                                                    dtype=tf.float32)
            word_embedding = tf.nn.embedding_lookup(params=embedding_matrix_variable, ids=self.word_batch)

        with tf.variable_scope('cap_embedding'):
            initial_embedding_matrix = np.random.randn(n_cap_feats, hparams.cap_dim) / np.sqrt(hparams.cap_dim)
            embedding_matrix_variable = tf.Variable(initial_value=initial_embedding_matrix,
                                                    name='embeddings_matrix',
                                                    dtype=tf.float32)
            case_embedding = tf.nn.embedding_lookup(params=embedding_matrix_variable, ids=self.cap_feat_batch)

        inputs = tf.concat([word_embedding, case_embedding],axis=-1,name='inputs')
        with tf.variable_scope('encoder'):
            forward_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hparams.word_lstm_dim),
                input_keep_prob=self.dropout_ph,
                output_keep_prob=self.dropout_ph,
                state_keep_prob=self.dropout_ph)
            backward_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hparams.word_lstm_dim),
                input_keep_prob=self.dropout_ph,
                output_keep_prob=self.dropout_ph,
                state_keep_prob=self.dropout_ph)

            (rnn_output_fw, rnn_output_bw), _ =  tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                                                 cell_bw=backward_cell,
                                                                                 inputs=inputs,
                                                                                 sequence_length=self.lengths,
                                                                                 dtype=tf.float32)
            rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

            self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    def compute_predictions(self):
        with tf.variable_scope('output'):
            softmax_output = tf.nn.softmax(self.logits, name='softmax_output')
            self.predictions = tf.argmax(softmax_output, axis=-1, name='predictions')

    def compute_loss(self, n_tags, PAD_index):
        # Create cross entropy function function (tf.nn.softmax_cross_entropy_with_logits_v2)
        ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                          labels=ground_truth_tags_one_hot)

        mask = tf.cast(tf.not_equal(self.word_batch, PAD_index), tf.float32)
        self.loss = tf.reduce_mean(mask * loss_tensor)


    def perform_optimization(self, global_step):
        """Specifies the optimizer and train_op for the model."""
        # Create an optimizer (tf.train.AdamOptimizer)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        clip_norm = tf.cast(1.0, tf.float32)
        self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in
                                                 self.grads_and_vars]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=global_step)

    def train_on_batch(self, session, step, words, capfeats, tags, lengths, learning_rate, dropout_keep_probability):
        feed_dict = {self.word_batch: words,
                     self.cap_feat_batch: capfeats,
                     self.ground_truth_tags: tags,
                     self.learning_rate_ph: learning_rate,
                     self.dropout_ph: dropout_keep_probability,
                     self.lengths: lengths}

        step,loss,_ = session.run([step, self.loss, self.train_op], feed_dict=feed_dict)
        return step,loss

    def predict_for_batch(self, session, words, cap_feats, lengths):
        predictions = session.run(self.predictions, feed_dict={self.word_batch: words,
                                                               self.cap_feat_batch: cap_feats,
                                                               self.lengths: lengths})

        return predictions

    def add_stats(self, scope_name='train'):
        with tf.variable_scope(scope_name) as scope:
            summaries = [
                tf.summary.scalar('loss', self.loss)
            ]

            if scope_name == 'train':
                gradient_norms = [tf.norm(grad) for grad,_ in self.grads_and_vars if self.grads_and_vars is not None]

                summaries.extend([
                    tf.summary.scalar('learning_rate', self.learning_rate_ph),
                    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)),
                ])
        return tf.summary.merge(summaries)
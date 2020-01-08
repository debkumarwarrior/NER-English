import tensorflow as tf

hparams = tf.contrib.training.HParams(
    word_dim = 100,
    word_lstm_dim = 100,
    cap_dim = 25,
    dropout = 0.5,
    lr_method = "sgd-lr_.005"
)

def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
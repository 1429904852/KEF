import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'twitter2017', ['twitter2017', 'twitter2015'])
tf.flags.DEFINE_string('phase', 'bert_train', ['bert_train', 'bert_test'])
tf.flags.DEFINE_string('config_path', 'src/multimodal/config/twitter_config.json', 'json file')

if FLAGS.dataset == "twitter2017":
    if FLAGS.phase == 'bert_train_anp':
        from src.multimodal.trainer import Trainer
        trainer = Trainer(FLAGS)
        trainer.train()
    elif FLAGS.phase == 'bert_test_anp':
        from src.multimodal.test import Test
        Test(FLAGS)

elif FLAGS.dataset == "twitter2015":
    if FLAGS.phase == 'bert_train_anp':
        from src.multimodal.trainer import Trainer
        trainer = Trainer(FLAGS)
        trainer.train()
    elif FLAGS.phase == 'bert_test_anp':
        from src.multimodal.test import Test
        Test(FLAGS)
else:
    raise ValueError(FLAGS.phase)

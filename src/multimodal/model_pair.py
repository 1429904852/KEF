#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof


import os
import tensorflow as tf
from src.bert import optimization, modeling
from src.multimodal.module import multi_head_attention


class BertClassifier(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(config["bert_model_path"], "bert_config.json")
        self.__num_classes = config["num_classes"]
        self.__learning_rate = config["learning_rate"]
        self.sequence_len = config["sequence_length"]
        self.target_length = config["target_sequence_length"]
        self.picture_length = config["picture_length"]
        self.picture_dimension = config["picture_dimension"]
        self.multi_heads = config["multi_heads"]
        self.attention_parameter = config["attention_parameter"]
        self.aux_parameter = config["aux_parameter"]
        self.anp_parameter = config["anp_parameter"]
        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.__target_classes = config["target_classes"]

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')

        self.target_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_ids')
        self.target_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_masks')
        self.target_segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_segment_ids')

        self.picture_ids = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='picture_ids')
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids")

        self.target_label_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_label_ids')
        self.target_label_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_label_masks')
        self.target_label_segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                       name='target_label_segment_ids')

        self.adj_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='adj_ids')
        self.adj_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='adj_masks')
        self.adj_segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='adj_segment_ids')
        self.adj_index_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='adj_index_ids')

        self.noun_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='noun_ids')
        self.noun_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='noun_masks')
        self.noun_segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='noun_segment_ids')
        self.noun_index_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='noun_index_ids')

        self.built_model()
        self.init_saver()

    def built_model(self):
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)

        model = modeling.BertModel(config=bert_config,
                                   is_training=self.__is_training,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_masks,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=False)

        target_model = modeling.BertModel(config=bert_config,
                                          is_training=self.__is_training,
                                          input_ids=self.target_ids,
                                          input_mask=self.target_masks,
                                          token_type_ids=self.target_segment_ids,
                                          use_one_hot_embeddings=False)

        target_label_model = modeling.BertModel(config=bert_config,
                                                is_training=self.__is_training,
                                                input_ids=self.target_label_ids,
                                                input_mask=self.target_label_masks,
                                                token_type_ids=self.target_label_segment_ids,
                                                use_one_hot_embeddings=False)

        anp_adj_model = modeling.BertModel(config=bert_config,
                                           is_training=self.__is_training,
                                           input_ids=self.adj_ids,
                                           input_mask=self.adj_masks,
                                           token_type_ids=self.adj_segment_ids,
                                           use_one_hot_embeddings=False)

        anp_noun_model = modeling.BertModel(config=bert_config,
                                            is_training=self.__is_training,
                                            input_ids=self.noun_ids,
                                            input_mask=self.noun_masks,
                                            token_type_ids=self.noun_segment_ids,
                                            use_one_hot_embeddings=False)

        output_layer = model.get_sequence_output()
        hidden_size = output_layer.shape[-1].value

        output_target = target_model.get_sequence_output()
        output_picture_feature = self.picture_ids

        output_target_label = target_label_model.get_sequence_output()
        output_anp_adj = anp_adj_model.get_sequence_output()
        output_anp_noun = anp_noun_model.get_sequence_output()

        if self.__is_training:
            # I.e., 0.1 dropout
            # 12
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            output_target = tf.nn.dropout(output_target, keep_prob=0.9)
            output_target_label = tf.nn.dropout(output_target_label, keep_prob=0.9)
            output_anp_adj = tf.nn.dropout(output_anp_adj, keep_prob=0.9)
            # [-1, 32, 768]
            output_anp_noun = tf.nn.dropout(output_anp_noun, keep_prob=0.9)

        with tf.name_scope("output"):
            output_weights = tf.get_variable("output_weights",
                                             [self.__num_classes, 2 * hidden_size],
                                             initializer=tf.truncated_normal_initializer(stddev=0.01))
            output_bias = tf.get_variable("output_bias", [self.__num_classes], initializer=tf.zeros_initializer())

            # 方法1
            # 求模
            output_layer_anp_norm = tf.sqrt(tf.reduce_sum(tf.square(output_target), axis=2))
            output_anp_norm = tf.sqrt(tf.reduce_sum(tf.square(output_anp_noun), axis=2))

            # 内积
            output_layer_sen_pivot = tf.reduce_sum(tf.multiply(output_target, output_anp_noun), axis=2)
            # [-1, sen_len]
            cosin = tf.divide(output_layer_sen_pivot, tf.multiply(output_layer_anp_norm, output_anp_norm))
            weight_ = tf.expand_dims(cosin, -1)
            output_recon_target = weight_ * output_anp_noun

            output_target = output_target + self.attention_parameter * output_recon_target

            output_layer_sen = multi_head_attention(output_layer, output_layer, output_target, hidden_size,
                                                        self.multi_heads,
                                                        scope="multihead_1")


            # target attention picture
            inputs_pic = tf.reshape(output_picture_feature, [-1, self.picture_length, self.picture_dimension])
            output_layer_pic = multi_head_attention(inputs_pic, inputs_pic, output_target, hidden_size,
                                                        self.multi_heads,
                                                        scope="multihead_2")
                
            # 方法2
            output_query_adj_total = weight_ * output_anp_adj
            output_layer_pic = output_layer_pic + self.anp_parameter * output_query_adj_total
            
            output_total = tf.concat([output_layer_sen, output_query_adj_total], -1)
            output_first = tf.squeeze(output_total[:, 0:1, :], axis=1)
            # output_last = tf.squeeze(output_total[:, -2:-1, :], axis=1)
            
            if self.__is_training:
                self.output = tf.nn.dropout(output_first, keep_prob=0.5)
            else:
                self.output = output_first
            # self.output = tf.nn.dropout(output_first, keep_prob=1.0)
            
            # output_mean = tf.reduce_mean(output_total, 1)
            # output_max = tf.reduce_max(output_total, 1)
            
            # output = tf.concat([output_last, output_first], 1)
            
            logits = tf.matmul(self.output, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            self.predictions = tf.argmax(logits, axis=-1, name="predictions")

        if self.__is_training:
            with tf.name_scope("loss"):
                # losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label_ids)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label_ids)
                self.loss = tf.reduce_mean(losses, name="loss")

                # 辅助loss
                recon_loss = tf.square(output_recon_target - output_layer_pic)
                recon_loss_ = tf.reduce_mean(tf.reduce_mean(recon_loss, 1), -1)
                self.mse = tf.reduce_mean(recon_loss_)
                self.loss = self.loss_1 + self.aux_parameter * self.mse

            with tf.name_scope('train_op'):
                self.train_op = optimization.create_optimizer(
                    self.loss, self.__learning_rate, self.__num_train_step, self.__num_warmup_step, use_tpu=False)

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {
            self.input_ids: batch["input_ids"],
            self.input_masks: batch["input_masks"],
            self.segment_ids: batch["segment_ids"],
            self.target_ids: batch["target_ids"],
            self.target_masks: batch["target_masks"],
            self.target_segment_ids: batch["target_segment_ids"],
            self.picture_ids: batch["picture_ids"],
            self.label_ids: batch["label_ids"],
            self.target_label_ids: batch["target_label_ids"],
            self.target_label_masks: batch["target_label_masks"],
            self.target_label_segment_ids: batch["target_label_segment_ids"],
            self.adj_ids: batch["adj_ids"],
            self.adj_masks: batch["adj_masks"],
            self.adj_segment_ids: batch["adj_segment_ids"],
            self.adj_index_ids: batch["adj_index_ids"],
            self.noun_ids: batch["noun_ids"],
            self.noun_masks: batch["noun_masks"],
            self.noun_segment_ids: batch["noun_segment_ids"],
            self.noun_index_ids: batch["noun_index_ids"]
        }

        # 训练模型
        _, loss, predictions = sess.run(
            [self.train_op, self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def eval(self, sess, batch):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {
            self.input_ids: batch["input_ids"],
            self.input_masks: batch["input_masks"],
            self.segment_ids: batch["segment_ids"],
            self.target_ids: batch["target_ids"],
            self.target_masks: batch["target_masks"],
            self.target_segment_ids: batch["target_segment_ids"],
            self.picture_ids: batch["picture_ids"],
            self.label_ids: batch["label_ids"],
            self.target_label_ids: batch["target_label_ids"],
            self.target_label_masks: batch["target_label_masks"],
            self.target_label_segment_ids: batch["target_label_segment_ids"],
            self.adj_ids: batch["adj_ids"],
            self.adj_masks: batch["adj_masks"],
            self.adj_segment_ids: batch["adj_segment_ids"],
            self.adj_index_ids: batch["adj_index_ids"],
            self.noun_ids: batch["noun_ids"],
            self.noun_masks: batch["noun_masks"],
            self.noun_segment_ids: batch["noun_segment_ids"],
            self.noun_index_ids: batch["noun_index_ids"]
        }

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        feed_dict = {
            self.input_ids: batch["input_ids"],
            self.input_masks: batch["input_masks"],
            self.segment_ids: batch["segment_ids"],
            self.target_ids: batch["target_ids"],
            self.target_masks: batch["target_masks"],
            self.target_segment_ids: batch["target_segment_ids"],
            self.picture_ids: batch["picture_ids"],
            self.label_ids: batch["label_ids"],
            self.target_label_ids: batch["target_label_ids"],
            self.target_label_masks: batch["target_label_masks"],
            self.target_label_segment_ids: batch["target_label_segment_ids"],
            self.adj_ids: batch["adj_ids"],
            self.adj_masks: batch["adj_masks"],
            self.adj_segment_ids: batch["adj_segment_ids"],
            self.adj_index_ids: batch["adj_index_ids"],
            self.noun_ids: batch["noun_ids"],
            self.noun_masks: batch["noun_masks"],
            self.noun_segment_ids: batch["noun_segment_ids"],
            self.noun_index_ids: batch["noun_index_ids"]
        }

        predict, features = sess.run([self.predictions, self.output], feed_dict=feed_dict)

        return predict, features

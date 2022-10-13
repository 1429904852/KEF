#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import json
import os
import time

import tensorflow as tf
from src.multimodal.model_pair import BertClassifier
from src.bert import modeling
from src.multimodal.data_helper import TrainData
from src.multimodal.metrics import mean
from sklearn.metrics import accuracy_score, f1_score


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)
        self.__bert_checkpoint_path = os.path.join(self.config["bert_model_path"], "bert_model.ckpt")
        self.dataset = args.dataset

        # 加载数据集
        self.data_obj = self.load_data()

        self.t_in_ids, self.t_in_masks, self.t_seg_ids, self.t_tar_ids, self.t_tar_masks, \
        self.t_tar_seg_ids, self.t_pic_ids, self.t_lab_ids, lab_to_idx, self.t_t_l_ids, self.t_t_l_masks, self.t_t_l_seg_ids, \
        self.t_a_ids, self.t_a_masks, self.t_a_seg_ids, self.t_a_index_ids, \
        self.t_n_ids, self.t_n_masks, self.t_n_seg_ids, self.t_n_index_ids = self.data_obj.gen_data(
            self.config["train_data"])

        self.e_in_ids, self.e_in_masks, self.e_seg_ids, self.e_tar_in_ids, self.e_tar_in_masks, self.e_tar_seg_ids, \
        self.e_pic_ids, self.e_lab_ids, lab_to_idx, self.e_t_l_ids, self.e_t_l_masks, self.e_t_l_seg_ids, \
        self.e_a_ids, self.e_a_masks, self.e_a_seg_ids, self.e_a_index_ids, \
        self.e_n_ids, self.e_n_masks, self.e_n_seg_ids, self.e_n_index_ids = self.data_obj.gen_data(
            self.config["eval_data"], is_training=False)

        print("train data size: {}".format(len(self.t_lab_ids)))
        print("eval data size: {}".format(len(self.e_lab_ids)))

        self.label_list = [value for key, value in lab_to_idx.items()]
        print("label numbers: ", len(self.label_list))

        num_train_steps = int(len(self.t_lab_ids) / self.config["batch_size"] * self.config["epochs"])
        num_warmup_steps = int(num_train_steps * self.config["warmup_rate"])
        # 初始化模型对象
        self.model = self.create_model(num_train_steps, num_warmup_steps)

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        data_obj = TrainData(self.config)
        return data_obj

    def create_model(self, num_train_step, num_warmup_step):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        model = BertClassifier(config=self.config, num_train_step=num_train_step, num_warmup_step=num_warmup_step)
        return model

    def train(self):
        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.__bert_checkpoint_path)
            print("init bert model params")
            tf.train.init_from_checkpoint(self.__bert_checkpoint_path, assignment_map)
            print("init bert model params done")
            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0
            start = time.time()
            max_acc = 0
            max_target_acc = 0
            max_f1 = 0
            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                # optimize baseline
                for batch in self.data_obj.next_batch(self.t_in_ids, self.t_in_masks, self.t_seg_ids, self.t_tar_ids,
                                                      self.t_tar_masks, self.t_tar_seg_ids, self.t_pic_ids,
                                                      self.t_lab_ids, self.t_t_l_ids, self.t_t_l_masks,
                                                      self.t_t_l_seg_ids,
                                                      self.t_a_ids, self.t_a_masks, self.t_a_seg_ids,
                                                      self.t_a_index_ids,
                                                      self.t_n_ids, self.t_n_masks, self.t_n_seg_ids,
                                                      self.t_n_index_ids):
                    loss, predictions = self.model.train(sess, batch)

                    acc = accuracy_score(list(batch["label_ids"]), list(predictions))

                    f1 = f1_score(list(batch["label_ids"]), list(predictions), average='macro')
                    print("train: step: {}, loss: {:.4f}, acc: {:.4f}, f1: {:.4f}".format(current_step,
                                                                                          loss, acc, f1))

                    current_step += 1
                    if self.data_obj and current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_accs = []
                        true_label = []
                        pre_lable = []

                        for eval_batch in self.data_obj.next_batch(self.e_in_ids, self.e_in_masks, self.e_seg_ids,
                                                                   self.e_tar_in_ids, self.e_tar_in_masks,
                                                                   self.e_tar_seg_ids, self.e_pic_ids, self.e_lab_ids,
                                                                   self.e_t_l_ids, self.e_t_l_masks, self.e_t_l_seg_ids,
                                                                   self.e_a_ids, self.e_a_masks, self.e_a_seg_ids,
                                                                   self.e_a_index_ids,
                                                                   self.e_n_ids, self.e_n_masks, self.e_n_seg_ids,
                                                                   self.e_n_index_ids):
                            eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_losses.append(eval_loss)

                            e_acc = accuracy_score(list(eval_batch["label_ids"]), list(eval_predictions))

                            true_label += list(eval_batch["label_ids"])
                            pre_lable += list(eval_predictions)

                            eval_accs.append(e_acc)

                        e_f1 = f1_score(true_label, pre_lable, average='macro')
                        print("\n")
                        print("eval: loss: {:.4f}, acc: {:.4f}, f_1: {:.4f}".
                              format(mean(eval_losses), mean(eval_accs), e_f1))

                        if mean(eval_accs) > max_acc:
                            max_acc = mean(eval_accs)
                            max_f1 = e_f1
                            save_path = self.config["ckpt_model_path"]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            path = self.model.saver.save(sess, model_save_path, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))

                        print("topacc {:g}, topf1 {:g}".format(max_acc, max_f1))
                        print("\n")
            end = time.time()
            print("topacc {:g}, topf1 {:g}".format(max_acc, max_f1))
            print("total train time: ", end - start)

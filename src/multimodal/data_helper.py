#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import json
import os
import random
import io
import re
from src.bert import tokenization
import numpy as np


class TrainData(object):
    def __init__(self, config):

        self.__vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.__output_path = config["output_path"]
        self.__pictures_path = config["pictures_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)
        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长
        self._target_sequence_length = config["target_sequence_length"]  # target序列处理为定长
        # self._pivot_sequence_length = config["pivot_sequence_length"]
        self._batch_size = config["batch_size"]

    @staticmethod
    def read_data(file_path):
        """
        读取数据
        :param file_path:
        :return: 返回分词后的文本内容和标签，inputs = [], labels = []
        """
        inputs = []
        target = []
        labels = []
        target_labels = []
        pictures = []

        adj = []
        noun = []

        lines = io.open(file_path, "r", encoding="UTF-8").readlines()
        for i in range(0, len(lines), 7):
            # inputs.append(lines[i].strip())
            inputs.append(TrainData.text_cleaner(lines[i].strip().lower()))
            target.append(lines[i + 1].strip().lower())
            labels.append(lines[i + 2].strip())
            target_labels.append(lines[i + 3].strip())
            pictures.append(lines[i + 4].strip())
            # adj.append(lines[i + 5].strip())
            # noun.append(lines[i + 6].strip())
            # young fancy nice muddy traditional
            adj_a = lines[i + 5].strip().split()[0:3]
            adj.append(" ".join(adj_a))
            noun_n = lines[i + 6].strip().split()[0:3]
            noun.append(" ".join(noun_n))

        return inputs, target, labels, target_labels, pictures, adj, noun

    @staticmethod
    def text_cleaner(text):
        """
        cleaning spaces, html tags, etc
        parameters: (string) text input to clean
        return: (string) clean_text
        """
        text = text.replace(".", "")
        text = text.replace("[", " ")
        text = text.replace(",", " ")
        text = text.replace("]", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("\"", "")
        text = text.replace("-", "")
        text = text.replace("=", "")
        text = text.replace("@", "")
        text = text.replace("#", "")
        rules = [
            {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
            {r'\s+': u' '},  # replace consecutive spaces
            {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
            {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
            {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
            {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
            {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
            {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
            {r'^\s+': u''}  # remove spaces at the beginning
        ]
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
        clean_text = text.lower()
        return clean_text

    def trans_to_index(self, inputs):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []
        for text in inputs:
            text = tokenization.convert_to_unicode(text)
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id, _ = tokenizer.convert_tokens_to_ids(tokens)
            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))

        return input_ids, input_masks, segment_ids

    def trans_to_adj_noun_index(self, inputs):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []
        index_ids = []
        for text in inputs:
            text = tokenization.convert_to_unicode(text)
            # print(text.split())
            # tokens = tokenizer.tokenize(text)
            tokens = []
            for text_i in text.split():
                tmp = tokenizer.tokenize(text_i)
                tokens += tmp
                tokens += ["[SEP]"]
            # tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tokens = ["[CLS]"] + tokens
            input_id, index = tokenizer.convert_tokens_to_ids(tokens)
            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))
            index_ids.append(index)

        return input_ids, input_masks, segment_ids, index_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_len):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_len:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def trans_to_index_input(self, text_as, text_bs):
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []
        for text_a, text_b in zip(text_as, text_bs):
            text_a = tokenization.convert_to_unicode(text_a)
            text_b = tokenization.convert_to_unicode(text_b)
            tokens_a = tokenizer.tokenize(text_a)
            tokens_b = tokenizer.tokenize(text_b)

            # 判断两条序列组合在一起长度是否超过最大长度
            self._truncate_seq_pair(tokens_a, tokens_b, self._sequence_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            input_id, _ = tokenizer.convert_tokens_to_ids(tokens)
            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1))

        return input_ids, input_masks, segment_ids

    def picture_feature(self, pictures):
        pictures_id = []
        for picture in pictures:
            photo_feature_path = self.__pictures_path + picture.split(".")[0] + '.npy'
            photo_features = np.load(photo_feature_path)
            pictures_id.append(photo_features)
        pictures_id = np.array(pictures_id)
        print(pictures_id.shape)
        return pictures_id

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        labels_idx = [label_to_index[label] for label in labels]
        return labels_idx

    def padding(self, input_ids, input_masks, segment_ids, sentence_length):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids = [], [], []
        for input_id, input_mask, segment_id in zip(input_ids, input_masks, segment_ids):
            if len(input_id) < sentence_length:
                pad_input_ids.append(input_id + [0] * (sentence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (sentence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (sentence_length - len(segment_id)))
            else:
                pad_input_ids.append(input_id[:sentence_length])
                pad_input_masks.append(input_mask[:sentence_length])
                pad_segment_ids.append(segment_id[:sentence_length])

        return pad_input_ids, pad_input_masks, pad_segment_ids

    def gen_data(self, file_path, is_training=True):
        """
        生成数据
        :param file_path:
        :param is_training:
        :return:
        """
        # 1, 读取原始数据
        inputs, target, labels, target_labels, pictures, adj, noun = self.read_data(file_path)
        print("read finished")

        if is_training:
            uni_label = list(set(labels))
            label_to_index = dict(zip(uni_label, list(range(len(uni_label)))))
            with io.open(os.path.join(self.__output_path, "label_to_index.json"), "w", encoding="utf-8") as fw:
                fw.write(str(json.dumps(label_to_index, ensure_ascii=False)))
        else:
            with io.open(os.path.join(self.__output_path, "label_to_index.json"), "r", encoding="utf-8") as fr:
                label_to_index = json.load(fr)

        # 2, inputs转索引
        inputs_ids, input_masks, segment_ids = self.trans_to_index_input(inputs, target)
        # inputs_ids, input_masks, segment_ids = self.trans_to_index(inputs)
        inputs_ids, input_masks, segment_ids = self.padding(inputs_ids, input_masks, segment_ids, self._sequence_length)
        print("inputs transform finished")

        # 3, target转索引
        target_ids, target_masks, target_segment_ids = self.trans_to_index(target)
        target_ids, target_masks, target_segment_ids = self.padding(target_ids, target_masks, target_segment_ids,
                                                                    self._target_sequence_length)
        print("target transform finished")

        # 4, target_label转索引
        target_label_ids, target_label_masks, target_label_segment_ids = self.trans_to_index(target_labels)
        target_label_ids, target_label_masks, target_label_segment_ids = self.padding(target_label_ids,
                                                                                      target_label_masks,
                                                                                      target_label_segment_ids,
                                                                                      self._target_sequence_length)
        print("target transform finished")

        # 5, 读入图片特征
        pictures_id = self.picture_feature(pictures)

        # 6, 标签转索引
        labels_ids = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")

        # 7, adj转索引
        adj_ids, adj_masks, adj_segment_ids, adj_index_ids = self.trans_to_adj_noun_index(adj)
        adj_ids, adj_masks, adj_segment_ids = self.padding(adj_ids,
                                                           adj_masks,
                                                           adj_segment_ids,
                                                           self._target_sequence_length)

        # 8, noun转索引
        noun_ids, noun_masks, noun_segment_ids, noun_index_ids = self.trans_to_adj_noun_index(noun)
        noun_ids, noun_masks, noun_segment_ids = self.padding(noun_ids,
                                                              noun_masks,
                                                              noun_segment_ids,
                                                              self._target_sequence_length)

        for i in range(5):
            print("line {}: *****************************************".format(i))
            print("input: ", inputs[i])
            print("input_id: ", inputs_ids[i])
            print("input_mask: ", input_masks[i])
            print("segment_id: ", segment_ids[i])
            print("target_id: ", target_ids[i])
            print("target_mask: ", target_masks[i])
            print("target_segment_id: ", target_segment_ids[i])
            print("label_id: ", labels_ids[i])

            print("target_label_ids: ", target_label_ids[i])
            print("target_label_masks: ", target_label_masks[i])
            print("target_label_segment_ids: ", target_label_segment_ids[i])

            print("adj_ids: ", adj_ids[i])
            print("adj_masks: ", adj_masks[i])
            print("adj_segment_ids: ", adj_segment_ids[i])
            print("adj_index_ids: ", adj_index_ids[i])

            print("noun_ids: ", noun_ids[i])
            print("noun_masks: ", noun_masks[i])
            print("noun_segment_ids: ", noun_segment_ids[i])
            print("noun_index_ids: ", noun_index_ids[i])

        return inputs_ids, input_masks, segment_ids, target_ids, target_masks, target_segment_ids, pictures_id, \
               labels_ids, label_to_index, target_label_ids, target_label_masks, target_label_segment_ids, \
               adj_ids, adj_masks, adj_segment_ids, adj_index_ids, noun_ids, noun_masks, noun_segment_ids, noun_index_ids

    def next_batch(self, input_ids, input_masks, segment_ids, target_ids, target_masks, target_segment_ids,
                   picture_ids, label_ids, target_label_ids, target_label_masks, target_label_segment_ids, adj_ids,
                   adj_masks, adj_segment_ids, adj_index_ids, noun_ids, noun_masks, noun_segment_ids, noun_index_ids):
        """
        生成batch数据
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids:
        :return:
        """
        z = list(zip(input_ids, input_masks, segment_ids, target_ids, target_masks, target_segment_ids,
                     picture_ids, label_ids, target_label_ids, target_label_masks, target_label_segment_ids, adj_ids,
                     adj_masks, adj_segment_ids, adj_index_ids, noun_ids, noun_masks, noun_segment_ids, noun_index_ids))
        random.shuffle(z)
        input_ids, input_masks, segment_ids, target_ids, target_masks, target_segment_ids, \
        picture_ids, label_ids, target_label_ids, target_label_masks, target_label_segment_ids, \
        adj_ids, adj_masks, adj_segment_ids, adj_index_ids, noun_ids, noun_masks, noun_segment_ids, noun_index_ids = zip(
            *z)

        num_batches = len(input_ids) // self._batch_size + (1 if len(input_ids) % self._batch_size else 0)

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size
            batch_input_ids = input_ids[start: end]
            batch_input_masks = input_masks[start: end]
            batch_segment_ids = segment_ids[start: end]

            batch_target_ids = target_ids[start: end]
            batch_target_masks = target_masks[start: end]
            batch_target_segment_ids = target_segment_ids[start: end]

            batch_picture_ids = picture_ids[start: end]
            batch_label_ids = label_ids[start: end]

            batch_target_label_ids = target_label_ids[start: end]
            batch_target_label_masks = target_label_masks[start: end]
            batch_target_label_segment_ids = target_label_segment_ids[start: end]

            batch_adj_ids = adj_ids[start: end]
            batch_adj_masks = adj_masks[start: end]
            batch_adj_segment_ids = adj_segment_ids[start: end]
            batch_adj_index_ids = adj_index_ids[start: end]

            batch_noun_ids = noun_ids[start: end]
            batch_noun_masks = noun_masks[start: end]
            batch_noun_segment_ids = noun_segment_ids[start: end]
            batch_noun_index_ids = noun_index_ids[start: end]

            yield dict(input_ids=batch_input_ids,
                       input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids,
                       target_ids=batch_target_ids,
                       target_masks=batch_target_masks,
                       target_segment_ids=batch_target_segment_ids,
                       picture_ids=batch_picture_ids,
                       label_ids=batch_label_ids,
                       target_label_ids=batch_target_label_ids,
                       target_label_masks=batch_target_label_masks,
                       target_label_segment_ids=batch_target_label_segment_ids,
                       adj_ids=batch_adj_ids,
                       adj_masks=batch_adj_masks,
                       adj_segment_ids=batch_adj_segment_ids,
                       adj_index_ids=batch_adj_index_ids,
                       noun_ids=batch_noun_ids,
                       noun_masks=batch_noun_masks,
                       noun_segment_ids=batch_noun_segment_ids,
                       noun_index_ids=batch_noun_index_ids
                       )

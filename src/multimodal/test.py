#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import json
from src.multimodal.predict import Predictor
from src.multimodal.metrics import mean
import io
from sklearn.metrics import accuracy_score, f1_score
import os
import numpy as np


def Test(FLAGS):
    with io.open(FLAGS.config_path, "r") as fr:
        config = json.load(fr)
    print(config)
    predictor = Predictor(config)

    label_to_index = predictor.label_to_index
    # target_label_to_index = predictor.target_label_to_index

    text, target, labels, target_labels, pictures, adj, noun = predictor.read_data(config['test_data'])

    labels_ids = predictor.trans_label_to_index(labels, label_to_index)

    input_ids, input_masks, segment_ids = predictor.trans_to_index_input(text, target)
    target_ids, target_masks, target_segment_ids = predictor.sentence_to_idx(target)

    target_label_ids, target_label_masks, target_label_segment_ids = predictor.sentence_to_idx(target_labels)

    adj_ids, adj_masks, adj_segment_ids, adj_index_ids = predictor.trans_to_adj_noun_index(adj)
    noun_ids, noun_masks, noun_segment_ids, noun_index_ids = predictor.trans_to_adj_noun_index(noun)

    picture_ids = predictor.picture_feature(pictures)

    eval_accs = []
    true_label, pre_lable = [], []
    # eval_test = []
    features = []
    for test_batch in predictor.next_batch(input_ids, input_masks, segment_ids, target_ids, target_masks,
                                           target_segment_ids, picture_ids, labels_ids, target_label_ids,
                                           target_label_masks, target_label_segment_ids,
                                           adj_ids, adj_masks, adj_segment_ids, adj_index_ids,
                                           noun_ids, noun_masks, noun_segment_ids, noun_index_ids):
        predictions, feature = predictor.predict(test_batch)
        features.append(feature)

        true_label += list(test_batch["label_ids"])
        pre_lable += list(predictions)

        acc = accuracy_score(list(test_batch["label_ids"]), list(predictions))
        eval_accs.append(acc)

    # new_features = np.concatenate(features, 0)
    # new_true_label = np.array(true_label)
    # np.save("/home/zhaof/Multi_model_bert/output/bert_twitter2017/feature_2017.npy", new_features)
    # np.save("/home/zhaof/Multi_model_bert/output/bert_twitter2017/label_2017.npy", new_true_label)

    test_f1 = f1_score(true_label, pre_lable, average='macro')
    print("acc {:g}, f1 {:g}".format(mean(eval_accs), test_f1))

    # output_file = os.path.join(config["output_path"], "label_2017.txt")
    # fp = open(output_file, 'w')
    
    # for t, p in zip(true_label, pre_lable):
    #     fp.write(str(t) + ' ' + str(p) + '\n')

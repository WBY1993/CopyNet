# -*- coding: utf-8 -*-
class Config():
    def __init__(self):
        self.vocab_file = "./data/vocab.pkl"
        self.save_dir = ".//model"
        self.PAD_ID = 0
        self.GO_ID = 1
        self.EOS_ID = 2
        self.NUM_ID = 3
        self.UNK_ID = 4
        self.epoch = 10
        self.batch_size = 16
        self.shuffle_size = 1000
        self.vocab_size = 10000
        self.embedding_size = 400
        self.learning_rate = 1e-3
        self.keep_prob = 0.75
        self.decay_step = 5000
        self.decay_rate = 0.5
        self.grad_clip = 5.0

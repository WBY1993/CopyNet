# -*- coding: utf-8 -*-
import pickle
import jieba


class Preprocess():
    def __init__(self, config):
        self.vocab = {}
        self.vocab_file = config.vocab_file
        self.PAD_ID = config.PAD_ID
        self.GO_ID = config.GO_ID
        self.EOS_ID = config.EOS_ID
        self.NUM_ID = config.NUM_ID
        self.UNK_ID = config.UNK_ID

    def build_vocab(self, file_list):
        vocab_dict = {}
        for file_name in file_list:
            with open(file_name, "r", encoding="utf-8") as file:
                line_num = 0
                for line in file:
                    line_num += 1
                    if line_num % 10000 == 0:
                        print("file_name:%s line:%d" % (file_name, line_num))
                    line = line.strip()
                    for word in jieba.cut(line):
                        if word not in ["PAD_ID", "GO_ID", "EOS_ID", "NUM_ID", "UNK_ID"]:
                            if word not in vocab_dict.keys():
                                vocab_dict[word] = 1
                            else:
                                vocab_dict[word] += 1
        
        vocab_list = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True)
        vocab2id = {"PAD_ID": self.PAD_ID, "GO_ID": self.GO_ID, "EOS_ID": self.EOS_ID, "NUM_ID": self.NUM_ID, "UNK_ID": self.UNK_ID}
        id2vocab = {self.PAD_ID: "PAD_ID", self.GO_ID: "GO_ID", self.EOS_ID: "EOS_ID", self.NUM_ID: "NUM_ID", self.UNK_ID: "UNK_ID"}
        for i,v in enumerate(vocab_list):
            vocab2id[v[0]] = len(vocab2id)
            id2vocab[len(id2vocab)] = v[0]
        self.vocab = {"vocab2id": vocab2id,"id2vocab": id2vocab}
        pickle.dump(self.vocab, open(self.vocab_file, "wb"), protocol=2)
        
    def add_vocab(self, file_list):
        vocab_dict = {}
        for file_name in file_list:
            with open(file_name, "r", encoding="utf-8") as file:
                line_num = 0
                for line in file:
                    line_num += 1
                    if line_num % 10000 == 0:
                        print("file_name:%s line:%d" % (file_name, line_num))
                    line = line.strip()
                    for word in jieba.cut(line):
                        if word not in self.vocab["vocab2id"]:
                            if word not in vocab_dict.keys():
                                vocab_dict[word] = 1
                            else:
                                vocab_dict[word] += 1
        
        vocab_list = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True)
        vocab2id = self.vocab["vocab2id"]
        id2vocab = self.vocab["id2vocab"]
        for i,v in enumerate(vocab_list):
            vocab2id[v[0]] = len(vocab2id)
            id2vocab[len(id2vocab)] = v[0]
        self.vocab = {"vocab2id": vocab2id,"id2vocab": id2vocab}
        pickle.dump(self.vocab, open(self.vocab_file, "wb"), protocol=2)
        
    def load_vocab(self):
        self.vocab = pickle.load(open(self.vocab_file, "rb"))
        
    def convert(self, encoder_file, decoder_file, data_file):
        encoder_len = []
        decoder_len = []
        file_en = open(encoder_file, "r", encoding="utf-8")
        file_de = open(decoder_file, "r", encoding="utf-8")
        line_num = 0
        with open(data_file, "w", encoding="utf-8") as f:
            while True:
                line_num += 1
                if line_num % 10000 == 0:
                    print("converting line:%d" % (line_num))
                encoder_str = file_en.readline()
                decoder_str = file_de.readline()
                if not encoder_str or not decoder_str:
                    break
                encoder_str = encoder_str.strip()
                decoder_str = decoder_str.strip()
                if not encoder_str or not decoder_str:
                    continue
                
                encoder_id = []
                decoder_id = []
                for word in jieba.cut(encoder_str):
                    try:
                        encoder_id.append(self.vocab["vocab2id"][word])
                    except KeyError:
                        if word.replace(".", "", 1).isdigit():
                            encoder_id.append(self.NUM_ID)
                        else:
                            encoder_id.append(self.UNK_ID)
                for word in jieba.cut(decoder_str):
                    try:
                        decoder_id.append(self.vocab["vocab2id"][word])
                    except KeyError:
                        if word.replace(".", "", 1).isdigit():
                            decoder_id.append(self.NUM_ID)
                        else:
                            decoder_id.append(self.UNK_ID)
                            
                encoder_len.append(len(encoder_id))
                decoder_len.append(len(decoder_id))
                print([encoder_id, decoder_id], file=f)
        file_en.close()
        file_de.close()
        print("Encoder length: %d--%d" % (min(encoder_len), max(encoder_len)))
        print("Decoder length: %d--%d" % (min(decoder_len), max(decoder_len)))

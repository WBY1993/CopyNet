# -*- coding: utf-8 -*-
import numpy as np


class Data_loader():
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.PAD_ID = config.PAD_ID
        self.GO_ID = config.GO_ID
        self.EOS_ID = config.EOS_ID
        self.NUM_ID = config.NUM_ID
        self.UNK_ID = config.UNK_ID
        
    def handle(self, encoder_input_data, decoder_input_label):
        outputs = []
        for i in range(len(encoder_input_data)):
            encoder = encoder_input_data[i]
            decoder = decoder_input_label[i]
            visit = np.zeros_like(encoder, dtype=np.int32)
            sample = []
            for j in range(len(decoder)):
                find_same = False
                for k in range(len(encoder)):
                    if decoder[j]==encoder[k] and visit[k]==0:
                        sample.append(self.vocab_size + k)
                        visit[k] = 1
                        find_same = True
                        break
                if not find_same:
                    if decoder[j]<self.vocab_size:
                        sample.append(decoder[j])
                    else:
                        sample.append(self.UNK_ID)
            outputs.append(sample)
        return np.array(outputs)
                
    def replace_with_unk(self, inputs):
        condition = np.less(inputs, self.vocab_size)
        output = np.where(condition, inputs, np.ones_like(inputs)*self.UNK_ID)
        return output
        
    def padding(self, encoder_input, decoder_input):
        input_max_len = max([len(i) for i in encoder_input])
        output_max_len = max([len(i) for i in decoder_input])
        encoder_input_data = []
        decoder_input_data = []
        decoder_input_label = []
        for i in range(len(encoder_input)):
            encoder_input_data.append(np.array(encoder_input[i] + [self.PAD_ID] * (input_max_len-len(encoder_input[i])), dtype=np.int32))
            data = decoder_input[i] + [self.EOS_ID] + [self.PAD_ID] * (output_max_len-len(decoder_input[i]))
            decoder_input_data.append(np.array([self.GO_ID] + data[:-1], dtype=np.int32))
            decoder_input_label.append(np.array(data, dtype=np.int32))
        return encoder_input_data, decoder_input_data, decoder_input_label
            
    def create_batches(self, data_file, batch_size, shuffle_size):
        with open(data_file, "r", encoding="utf-8") as file:
            text_list = []
            while True:
                data = file.readline()
                if data:
                    data = eval(data)
                    text_list.append(data)
                    if len(text_list)>=shuffle_size:
                        np.random.shuffle(text_list)
                        encoder_input = []
                        decoder_input = []
                        for _ in range(batch_size):
                            sample = text_list.pop()
                            encoder_input.append(sample[0])
                            decoder_input.append(sample[1])
                        encoder_input_data, decoder_input_data, decoder_input_label = self.padding(encoder_input, decoder_input)
                        decoder_input_label = self.handle(encoder_input_data, decoder_input_label)
                        encoder_input_data = self.replace_with_unk(encoder_input_data)
                        decoder_input_data = self.replace_with_unk(decoder_input_data)
                        yield encoder_input_data, decoder_input_data, decoder_input_label
                else:
                    np.random.shuffle(text_list)
                    while len(text_list) >= batch_size:
                        encoder_input = []
                        decoder_input = []
                        for _ in range(batch_size):
                            sample = text_list.pop()
                            encoder_input.append(sample[0])
                            decoder_input.append(sample[1])
                        encoder_input_data, decoder_input_data, decoder_input_label = self.padding(encoder_input, decoder_input)
                        decoder_input_label = self.handle(encoder_input_data, decoder_input_label)
                        encoder_input_data = self.replace_with_unk(encoder_input_data)
                        decoder_input_data = self.replace_with_unk(decoder_input_data)
                        yield encoder_input_data, decoder_input_data, decoder_input_label
                    break
    

                
                

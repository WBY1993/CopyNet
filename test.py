# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import configuration
import preprocess
import jieba


class CopyNet:
    def __init__(self, config):
        # initial
        self.batch_size = 1
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.learning_rate = config.learning_rate
        self.GO_ID = config.GO_ID
        self.UNK_ID = config.UNK_ID
        self.EOS_ID = config.EOS_ID
        
        # input_data
        self.build_input()

        # build graph
        self.build_network()
        
    def replace_with_unk(self, inputs):
        condition = tf.less(inputs, self.vocab_size)
        outputs = tf.where(condition, inputs, tf.ones_like(inputs, dtype=tf.int32) * self.UNK_ID)
        return outputs
        
    def build_input(self):
        with tf.variable_scope("input"):
            self.encoder_input_data = tf.placeholder(tf.int32, shape=[None, None], name="encoder_input_data")
            self.encoder_seq_len = tf.reduce_sum(tf.sign(self.encoder_input_data), axis=1)
            self.encoder_max_len = tf.reduce_max(self.encoder_seq_len)
            self.decoder_input_data = tf.placeholder(tf.int32, shape=[None, None], name="decoder_input_data")
            self.decoder_input_label = tf.placeholder(tf.int32, shape=[None, None], name="decoder_input_label")
            self.decoder_seq_len = tf.reduce_sum(tf.sign(self.decoder_input_data), axis=1)
            self.decoder_max_len = tf.reduce_max(self.decoder_seq_len)
            self.target_weight = tf.cast(tf.sequence_mask(self.decoder_seq_len, self.decoder_max_len), dtype=tf.float32)
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    def build_network(self):
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            encoder_input_data_emb = tf.nn.embedding_lookup(embedding, self.replace_with_unk(self.encoder_input_data))
        
        with tf.variable_scope("encoder"):
            en_lstm1 = tf.contrib.rnn.LSTMCell(256)
            en_lstm1 = tf.contrib.rnn.DropoutWrapper(en_lstm1, output_keep_prob=self.keep_prob)
            en_lstm2 = tf.contrib.rnn.LSTMCell(256)
            en_lstm2 = tf.contrib.rnn.DropoutWrapper(en_lstm2, output_keep_prob=self.keep_prob)
            encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([en_lstm1])
            encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([en_lstm2])
            bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
                                                                                   encoder_cell_bw,
                                                                                   encoder_input_data_emb,
                                                                                   sequence_length=self.encoder_seq_len,
                                                                                   dtype=tf.float32)
            encoder_outputs = tf.concat(bi_encoder_outputs, -1)
            encoder_state = []
            for layer_id in range(1):  # layer_num
                encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                encoder_state.append(bi_encoder_state[1][layer_id])  # backward
            encoder_state = tuple(encoder_state)
        
        with tf.variable_scope("decoder"):
            with tf.variable_scope("attention"):
                de_lstm1 = tf.contrib.rnn.LSTMCell(256)
                de_lstm1 = tf.contrib.rnn.DropoutWrapper(de_lstm1, output_keep_prob=self.keep_prob)
                de_lstm2 = tf.contrib.rnn.LSTMCell(256)
                de_lstm2 = tf.contrib.rnn.DropoutWrapper(de_lstm2, output_keep_prob=self.keep_prob)
                decoder_cell = tf.contrib.rnn.MultiRNNCell([de_lstm1, de_lstm2])
                
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(256, encoder_outputs, self.encoder_seq_len)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, 256)
                decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
            with tf.variable_scope("output"):
                weights_copy = tf.get_variable("weights_copy", shape=[512, 256])
                weights_generate = tf.get_variable("weights_generate", shape=[256, self.vocab_size])
                def cond(time, state, copy_score, max_len, sample_word, sample_list):
                    return (time < max_len) & tf.not_equal(sample_word[0], self.EOS_ID)
                def body(time, state, copy_score, max_len, sample_word, sample_list):
                    # selective read
                    decoder_input = tf.cond(sample_word[0]<self.vocab_size, lambda: sample_word, lambda: self.encoder_input_data[:, sample_word[0]-self.vocab_size])
                    decoder_input_emb = tf.nn.embedding_lookup(embedding, decoder_input) # batch * embedding_size
                    condition = tf.cast(tf.equal(self.encoder_input_data, tf.expand_dims(decoder_input, axis=1)), dtype=tf.float32) # batch * en_seqs
                    selective_weights = condition * copy_score # batch * en_seqs
                    selective_weights_sum = tf.expand_dims(tf.reduce_sum(selective_weights, axis=1), axis=1) # batch * 1
                    selective_weights = selective_weights / (selective_weights_sum + 1e-4) # batch * en_seqs
                    decoder_input_sel = tf.expand_dims(selective_weights, axis=2) * encoder_outputs # batch * en_seqs * encoder_output_size
                    decoder_input_sel = tf.reduce_sum(decoder_input_sel, axis=1) # batch * encoder_output_size
                    decoder_input_final = tf.concat([decoder_input_emb, decoder_input_sel], axis=1)
                    output, state = decoder_cell(decoder_input_final, state) # batch * hidden_dim
                    # generate mode
                    generate_score = tf.matmul(output, weights_generate) # batch * vocab_size
                    # copy mode
                    output_expand = tf.expand_dims(output, axis=1) # batch * 1 * hidden_dim
                    copy_score = tf.tensordot(encoder_outputs, weights_copy, axes=1) # batch * en_seqs * hidden_dim
                    copy_score = tf.tanh(copy_score)
                    copy_score = tf.reduce_sum(copy_score * output_expand, axis=2) # batch * en_seqs
                    # move generate score to copy score
                    generate_score_expand = tf.expand_dims(generate_score, axis=1) # batch * 1 * vocab_size
                    mask = tf.one_hot(self.encoder_input_data, self.vocab_size) # batch * en_seqs * vocab_size
                    copy_score = copy_score + tf.reduce_sum(mask * generate_score_expand, axis=2) # batch * en_seqs
                    generate_score = generate_score - tf.reduce_sum(mask * generate_score_expand, axis=1) # batch * vocab_size
                    mix_score = tf.concat([generate_score, copy_score], axis=1) # batch * (vocab_size + en_seqs)
                    softmax = tf.nn.softmax(mix_score)
                    sample_word = tf.argmax(softmax, axis=1) # batch
                    sample_word = tf.cast(sample_word, tf.int32)
                    sample_list = sample_list.write(time, sample_word)
                    return time+1, state, softmax[:, self.vocab_size:], max_len, sample_word, sample_list
                    
                sample_list = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, name="sample_list")
                _, _, _, _, _, sample_list = tf.while_loop(cond, body, loop_vars=[0, decoder_initial_state, tf.zeros_like(self.encoder_input_data, dtype=tf.float32), 100, tf.ones_like(self.encoder_seq_len)*self.GO_ID, sample_list])
                sample_list = sample_list.stack() # de_seqs * batch
                self.sample_list = tf.transpose(sample_list, perm=[1, 0]) # batch * de_seqs
                
def replace_with_unk(inputs, config):
    for i in range(len(inputs)):
        if inputs[i] >= config.vocab_size:
            inputs[i] = config.UNK_ID
    return inputs
        
        
def main():
    conf = configuration.Config()
    pre_model = preprocess.Preprocess(conf)
    pre_model.load_vocab()
    copy_model = CopyNet(conf)   
    saver = tf.train.Saver()
    sess = tf.Session()
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(conf.save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        last_global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, last_global_step is %s' % last_global_step)
    else:
        print('No checkpoint file found')
    file_en = open("data/encoder.val", "r", encoding="utf-8")
    file_de = open("data/decoder.val", "r", encoding="utf-8")
    while True:
        import dataloader
        data_model = dataloader.Data_loader(conf)
        t1 = [[1,2,3,1,2]]
        t2 = [[1,2,3,1,2,10003,5,6]]
        t3 = data_model.handle(t1, t2)
        print(t3)
        return
        encoder = file_en.readline()
        decoder = file_de.readline()
        if not encoder or not decoder:
            break
        encoder = encoder.strip()
        decoder = decoder.strip()
        encoder = jieba.lcut(encoder)
        decoder = jieba.lcut(decoder)
        encoder_input = []
        for word in encoder:
            try:
                encoder_input.append(pre_model.vocab["vocab2id"][word])
            except KeyError:
                if word.replace(".", "", 1).isdigit():
                    encoder_input.append(conf.NUM_ID)
                else:
                    encoder_input.append(conf.UNK_ID)
        encoder_input = replace_with_unk(encoder_input, conf)
            
        encoder_input_data = []
        encoder_input_data.append(encoder_input)
        sample_list = sess.run(copy_model.sample_list, 
                          feed_dict={copy_model.encoder_input_data: encoder_input_data,
                                     copy_model.keep_prob: 1.0})
        
        output = []
        for word_id in sample_list[0]:
            if word_id==conf.EOS_ID:
                break
            try:
                if word_id>=conf.vocab_size:
                    output.append(encoder[word_id-conf.vocab_size])
                else:
                    output.append(pre_model.vocab["id2vocab"][word_id])
            except KeyError:
                output.append("UNK_ID")
        
        print("".join(encoder))
        print("".join(decoder))
        print("".join(output))
        print(sample_list)
        print("#########")
    file_en.close()
    file_de.close()
        
if __name__ == "__main__":
    main()

            
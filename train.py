# -*- coding: utf-8 -*-
import tensorflow as tf
import configuration
import preprocess
import dataloader
import copynet
import os


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
resume = True

def main(unused_argv):
    conf = configuration.Config()
#    ####################Preprocess#######################
#    pre_model = preprocess.Preprocess(conf)
#    #pre_model.build_vocab(["data/encoder.tra", "data/decoder.tra"])
#    #pre_model.add_vocab(["data/encoder.tra", "data/decoder.tra"])
#    pre_model.load_vocab()
#    pre_model.convert("data/encoder.tra", "data/decoder.tra", "data/data.tra")
#    #####################################################

    ####################Initialize#######################
    data_model = dataloader.Data_loader(conf)
    copy_model = copynet.CopyNet(conf)
    #####################################################

    ####################Build Graph#######################
    global_step = tf.Variable(0, trainable=False, name="global_step")
    with tf.variable_scope("optimizer"):
        decayed_lr_dis = tf.train.exponential_decay(learning_rate=conf.learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=conf.decay_step,
                                                    decay_rate=conf.decay_rate,
                                                    staircase=True)
        optimizer = tf.train.AdamOptimizer(decayed_lr_dis)
        var = [v for v in tf.trainable_variables()]
        grad, var = zip(*optimizer.compute_gradients(copy_model.loss, var_list=var))
        grad, _ = tf.clip_by_global_norm(grad, conf.grad_clip)
        trainop = optimizer.apply_gradients(zip(grad, var), global_step)
    summary_op = tf.summary.merge_all()
    ######################################################
        
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(os.path.join(conf.save_dir, "tra"), sess.graph)
    ####################Resume#######################
    if resume:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(conf.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            last_global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, last_global_step is %s' % last_global_step)
        else:
            print('No checkpoint file found')
    #################################################
    
    ####################Start training#######################
    for epoch in range(conf.epoch):
        for encoder_input_data, decoder_input_data, decoder_input_label in data_model.create_batches("data/data.tra", conf.batch_size, conf.shuffle_size):
            print(encoder_input_data[0])
            print(decoder_input_data[0])
            print(decoder_input_label[0])
            print("############")
#            _, tra_loss, summary_str = sess.run([trainop, copy_model.loss, summary_op],
#                                                feed_dict={copy_model.encoder_input_data: encoder_input_data,
#                                                           copy_model.decoder_input_data: decoder_input_data,
#                                                           copy_model.decoder_input_label: decoder_input_label,
#                                                           copy_model.keep_prob: conf.keep_prob})
#            step = global_step.eval(session=sess)
#            if step % 1 == 0:
#                summary_writer.add_summary(summary_str, global_step=step)
#                print("Epoch %d, Train Step %d, loss: %.4f" % (epoch, step, tra_loss))
#        saver.save(sess, os.path.join(conf.save_dir, "model.ckpt"), global_step=step)
    ########################################################
    sess.close()
    

if __name__ == "__main__":
    tf.app.run()

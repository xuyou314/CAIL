import json
from preproces import *
from tqdm import tqdm
import re
import os
import tensorflow as tf
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from model import Model
home = os.getcwd()

os.environ['CUDA_VISIBLE_DEVICES']='2'



flags = tf.flags

flags.DEFINE_string("answer_file", "../result/result.json", "Out file for answer")
flags.DEFINE_string("save_dir", "./model/", "Directory for saving model")

flags.DEFINE_integer("glove_dim", 200, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("hidden", 96, "Hidden size")
flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")


if __name__=="__main__":
    jieba.re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\xd7]+)", re.U)
    test_examples, test_eval = process_file(
        "../data/data.json", "test")
    config = flags.FLAGS
    with open("word_dictionary.json","r") as fh:
        word2idx_dict=json.load(fh)
    with open("char_dictionary.json","r") as fh:
        char2idx_dict=json.load(fh)
    test_meta = build_features(config, test_examples, "test",
                               "test.tfrecords", word2idx_dict, char2idx_dict, is_test=True)
    save("test_eval.json", test_eval, message="test eval")
    save("test_meta.json",test_meta,message="test meta")
    with open("word_emb.json", "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open("char_emb.json", "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open("test_eval.json", "r") as fh:
        eval_file = json.load(fh)
    with open("test_meta.json","r") as fh:
        meta = json.load(fh)
    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_batch = get_dataset("test.tfrecords", get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            losses = []
            answer_dict = {}
            remapped_dict = {}
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, loss, yp1, yp2 = sess.run(
                    [model.qa_id, model.loss, model.yp1, model.yp2])
                answer_dict_, remapped_dict_ = convert_tokens(
                    eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
                answer_dict.update(answer_dict_)
                remapped_dict.update(remapped_dict_)
                losses.append(loss)
            loss = np.mean(losses)
            # metrics = evaluate(eval_file, answer_dict)
            result_list=[]
            for key in remapped_dict.keys():
                item={}
                item['id']=key
                item['answer']=remapped_dict[key]
                result_list.append(item)
            with open(config.answer_file, "w") as fh:
                json.dump(result_list, fh,ensure_ascii=False)
            # print("Exact Match: {}, F1: {}".format(
            #     metrics['exact_match'], metrics['f1']))
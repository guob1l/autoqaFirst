#coding=utf-8
'''
Created on 2017��4��21��

@author: gb
'''
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import data_helpers
import jieba
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_float("checkpoint_every", 300, "Dropout keep probability (default: 0.5)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


##build_dic 
question_data_file='questionfile.txt'
answer_data_file='questionfilee.txt'
answer_data_file_w='questionfileee.txt'
x_text= data_helpers.load_data_and_labels(question_data_file, answer_data_file,answer_data_file_w)
x_question=x_text[0]
x_answer=x_text[1]
x_text_con=x_question+x_answer
max_document_length = max([len(x.split(" ")) for x in x_text_con])
print(max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(x_text_con)
x_question= np.array(list(vocab_processor.fit_transform(x_question)))#question （CNN的输入）
x_answer= np.array(list(vocab_processor.fit_transform(x_answer)))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))





#取参数，参数格式与restore权重格式相同
Wq = tf.Variable(tf.random_uniform([len(vocab_processor.vocabulary_), FLAGS.embedding_dim], -1.0, 1.0),name="Wq")
W_q2 = tf.Variable(tf.truncated_normal([2, FLAGS.embedding_dim, 1, FLAGS.num_filters], stddev=0.1), name="W_q2")
b_q2 = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_filters]), name="b_q2")#filter数量。
W_q3 = tf.Variable(tf.truncated_normal([3, FLAGS.embedding_dim, 1, FLAGS.num_filters], stddev=0.1), name="W_q3")
b_q3 = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_filters]), name="b_q3")#filter数量。
W_q5 = tf.Variable(tf.truncated_normal([5, FLAGS.embedding_dim, 1, FLAGS.num_filters], stddev=0.1), name="W_q5")
b_q5 = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_filters]), name="b_q5")#filter数量。

#测试问句读取
# question='test.txt'
# question = list(open(question, "r").readlines())
# question = [s.strip() for s in question]
# question = [sent.replace('？', '') for sent in question]
# print(question)
##手动输入问题：
print('请输入您的问题进行检索：')
str = input("Enter your input: ")
str= ' '.join(jieba.cut(str))
str=str.strip()
question=[]
question.append(str)

#答案 读取
answer='questionfilee.txt'
answer=list(open(answer,'r').readlines())
answer=[s.strip() for s in answer]
print(answer)

#问句下标表示
# question_index=next(vocab_processor.transform(question)).tolist()
# question_index=np.reshape(question_index, [-1,25])
question_index= np.array(list(vocab_processor.fit_transform(question)))
print(question_index)

#答案下标表示
# answer_input=[]
# for i in range(len(answer)):
#     answer_index=next(vocab_processor.transform(answer)).tolist()
#     answer_input.append(answer_index)
answer_index= np.array(list(vocab_processor.fit_transform(answer)))
print(answer_index)

embedded_chars_q = tf.nn.embedding_lookup(Wq, question_index)
embedded_chars_expanded_q = tf.expand_dims(embedded_chars_q, -1)

embedded_chars_a = tf.nn.embedding_lookup(Wq, answer_index)
embedded_chars_expanded_a = tf.expand_dims(embedded_chars_a, -1)
# print(embedded_chars_expanded_q)
pooled_outputs_q = []
pooled_outputs_a = []
for i, filter_size in enumerate([2,3,5]):
    if filter_size==2:
        W_q = W_q2
        b_q = b_q2
    if filter_size==3:
        W_q=W_q3
        b_q=b_q3
    if filter_size==5:
        W_q=W_q5
        b_q=b_q5
    conv_q = tf.nn.conv2d(                        #卷积operation

                    embedded_chars_expanded_q,           #the sentence of conv2input。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

                    W_q,                                        #[filter_height, filter_width, in_channels, out_channels]

                    strides=[1, 1, 1, 1],

                    padding="VALID",

                    name="conv_q")
    h_q = tf.nn.relu(tf.nn.bias_add(conv_q, b_q), name="relu1")

    pooled_q = tf.nn.max_pool(

                    h_q,

                    ksize=[1, x_question.shape[1] - filter_size + 1, 1, 1],

                    strides=[1, 1, 1, 1],

                    padding='VALID',

                    name="pool_q")
    pooled_outputs_q.append(pooled_q)

    conv_a = tf.nn.conv2d(  # 卷积operation

        embedded_chars_expanded_a,
        # the sentence of conv2input。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

        W_q,  # [filter_height, filter_width, in_channels, out_channels]

        strides=[1, 1, 1, 1],

        padding="VALID",

        name="conv_a")
    h_a = tf.nn.relu(tf.nn.bias_add(conv_a, b_q), name="relu1")

    pooled_a = tf.nn.max_pool(

        h_a,

        ksize=[1, x_question.shape[1] - filter_size + 1, 1, 1],

        strides=[1, 1, 1, 1],

        padding='VALID',

        name="pool_a")
    pooled_outputs_a.append(pooled_a)

# print(pooled_outputs_q)
num_filters_total = 128* 3                                      #filter_size * filter_number
h_pool_q = tf.concat(3,pooled_outputs_q)
h_pool_a = tf.concat(3,pooled_outputs_a)
h_drop_q = tf.reshape(h_pool_q, [-1, num_filters_total])
h_drop_a = tf.reshape(h_pool_a, [-1, num_filters_total])

# print(h_drop_q)
# # h_drop_a=0
# #  
h_drop_q_len = tf.sqrt(tf.reduce_sum(tf.mul(h_drop_q,h_drop_q), 1))#[ 26.56549835 ]

h_drop_a_len= tf.sqrt(tf.reduce_sum(tf.mul(h_drop_a,h_drop_a), 1))#[ 26.56549835  27.79394341  30.14491653  33.30422211  26.13010025...]


h_mul_qa = tf.reduce_sum(tf.mul(h_drop_q, h_drop_a), 1)
cos_12 = tf.div(h_mul_qa , tf.mul(h_drop_q_len, h_drop_a_len), name="scores")
# test=tf.mul(h_drop_q_len, h_drop_a_len)
print(cos_12)
result=tf.argmax(cos_12,0)
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "tmp/model.ckpt")
    # print(sess.run(W_q2))
    print(sess.run(h_drop_q_len))
    print(sess.run(h_drop_a_len))
    # print(sess.run(h_mul_qa))
    print(sess.run(cos_12))
    result_index=sess.run(result)
    print(result_index)
    print('下面为检索到的结果：')
    print(answer[result_index])


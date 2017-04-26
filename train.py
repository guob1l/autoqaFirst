#coding=utf-8
'''
Created on 2017��4��16��

@author: gb
'''
import data_helpers
from tensorflow.contrib import learn
import numpy as np
import tensorflow as tf
import time
import os
import datetime
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
##load data##
print("Loading data...")
question_data_file='questionfile.txt'
answer_data_file='questionfilee.txt'
answer_data_file_w='questionfileee.txt'
x_text= data_helpers.load_data_and_labels(question_data_file, answer_data_file,answer_data_file_w)
x_question=x_text[0]
x_answer=x_text[1]
x_answer_w=x_text[2]

##build_dic
print("build dictionary...")
x_text_con=x_question+x_answer
max_document_length = max([len(x.split(" ")) for x in x_text_con])

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(x_text_con)#答案问句所有词生成一个词典
x_question= np.array(list(vocab_processor.fit_transform(x_question)))
x_answer= np.array(list(vocab_processor.fit_transform(x_answer)))
x_answer_w= np.array(list(vocab_processor.fit_transform(x_answer_w)))

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

# Randomly shuffle data 打乱句子顺序（句子表示为字典数字形式）
np.random.seed(10)

shuffle_indices = np.random.permutation(np.arange(len(x_question)))

x_shuffled_q = x_question[shuffle_indices]
x_shuffled_a=x_answer[shuffle_indices]
x_shuffled_aa=x_answer_w[shuffle_indices]


#sequence_length:句子的长度。所有句子都补充到相同的长度。
#num_classes:输出层类别的数量。這里是2
#vocab_size:词汇表的大小。这个是被需要去定义embedding layer 的大小。shape :[vocab_size,embeding_size]
#embedding_size:
#filter_sizes:卷积器覆盖的词的数量。如【2，3，5】
#num_filters:卷积器的数量。
sequence_length=x_shuffled_q.shape[1]  #matrix列数即句子长度。

num_classes=2

vocab_size=len(vocab_processor.vocabulary_)

embedding_size=FLAGS.embedding_dim #手动定义为128

filter_sizes=list(map(int, FLAGS.filter_sizes.split(",")))

num_filters=FLAGS.num_filters   #128

l2_reg_lambda=FLAGS.l2_reg_lambda
        #输入数据占位定义，原始输入。。。为2维。。
##build Cnn MODEL
input_xq = tf.placeholder(tf.int32, [None, sequence_length], name="input_xq")#问题
input_xa = tf.placeholder(tf.int32, [None, sequence_length], name="input_xa")#正确答案
input_xaa = tf.placeholder(tf.int32, [None, sequence_length], name="input_xaa")#错误答案
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")#drop_out
l2_loss = tf.constant(0.0)



# Embedding layer ## 

Wq = tf.Variable(

    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),   #W is our embedding matrix that we learn during training.

    name="Wq")
                                     
#operation of create the actual embedding 结果为3维:[None, sequence_length, embedding_size].
embedded_chars_q = tf.nn.embedding_lookup(Wq, input_xq)
embedded_chars_a = tf.nn.embedding_lookup(Wq, input_xa)
embedded_chars_aa = tf.nn.embedding_lookup(Wq, input_xaa)  

#CNN输入为4维矩阵。需要执行扩维。
embedded_chars_expanded_q = tf.expand_dims(embedded_chars_q, -1)#shape [None, sequence_length, embedding_size, 1].
embedded_chars_expanded_a = tf.expand_dims(embedded_chars_a, -1)#shape [None, sequence_length, embedding_size, 1].
embedded_chars_expanded_aa = tf.expand_dims(embedded_chars_aa, -1)
# Create a convolution + maxpool layer for each filter size ##
pooled_outputs_q = []
pooled_outputs_a = []
pooled_outputs_aa = []




# Convolution Layer
#size=2
filter_shape = [2, embedding_size, 1, num_filters] #filter卷积窗口的shape
#filter(W)是一个四维Tensor：[filter_height, filter_width, in_channels, out_channels]   
W_q2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_q2")#卷积窗口的大小。
        
b_q2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_q2")#filter数量。
         


conv_q = tf.nn.conv2d(                        #卷积operation

    embedded_chars_expanded_q,           #the sentence of conv2input。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q2,                                        #[filter_height, filter_width, in_channels, out_channels]   

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_q")
conv_a = tf.nn.conv2d(                       #卷积operation

    embedded_chars_expanded_a,           #the sentence of conv2。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q2,

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_a")
conv_aa = tf.nn.conv2d(                       #卷积operation

    embedded_chars_expanded_aa,           #the sentence of conv2。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q2,

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_aa")
# Apply nonlinearity

h_q = tf.nn.relu(tf.nn.bias_add(conv_q, b_q2), name="relu1")
h_a = tf.nn.relu(tf.nn.bias_add(conv_a, b_q2), name="relu2")
h_aa = tf.nn.relu(tf.nn.bias_add(conv_aa,b_q2), name="relu3")
# Maxpooling over the outputs

pooled_q = tf.nn.max_pool(

    h_q,

    ksize=[1, sequence_length - 2 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_q")
pooled_a = tf.nn.max_pool(

    h_a,

    ksize=[1, sequence_length - 2 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_a")
pooled_aa = tf.nn.max_pool(

    h_aa,

    ksize=[1, sequence_length - 2 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_aa")
pooled_outputs_q.append(pooled_q)#循环3次 将pool值组合到一起。
pooled_outputs_a.append(pooled_a)
pooled_outputs_aa.append(pooled_aa)




##size == 3
filter_shape = [3, embedding_size, 1, num_filters] #filter卷积窗口的shape
#filter(W)是一个四维Tensor：[filter_height, filter_width, in_channels, out_channels]   
W_q3 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_q3")#卷积窗口的大小。
        
b_q3 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_q3")#filter数量。
         


conv_q = tf.nn.conv2d(                        #卷积operation

    embedded_chars_expanded_q,           #the sentence of conv2input。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q3,                                        #[filter_height, filter_width, in_channels, out_channels]   

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_q")
conv_a = tf.nn.conv2d(                       #卷积operation

    embedded_chars_expanded_a,           #the sentence of conv2。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q3,

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_a")
conv_aa = tf.nn.conv2d(                       #卷积operation

    embedded_chars_expanded_aa,           #the sentence of conv2。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q3,

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_aa")
# Apply nonlinearity

h_q = tf.nn.relu(tf.nn.bias_add(conv_q, b_q3), name="relu1")
h_a = tf.nn.relu(tf.nn.bias_add(conv_a, b_q3), name="relu2")
h_aa = tf.nn.relu(tf.nn.bias_add(conv_aa,b_q3), name="relu3")
# Maxpooling over the outputs

pooled_q = tf.nn.max_pool(

    h_q,

    ksize=[1, sequence_length - 3 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_q")
pooled_a = tf.nn.max_pool(

    h_a,

    ksize=[1, sequence_length - 3 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_a")
pooled_aa = tf.nn.max_pool(

    h_aa,

    ksize=[1, sequence_length - 3 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_aa")
pooled_outputs_q.append(pooled_q)#循环3次 将pool值组合到一起。
pooled_outputs_a.append(pooled_a)
pooled_outputs_aa.append(pooled_aa)




# Convolution Layer
##size==5
filter_shape = [5, embedding_size, 1, num_filters] #filter卷积窗口的shape
#filter(W)是一个四维Tensor：[filter_height, filter_width, in_channels, out_channels]   
W_q5 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_q5")#卷积窗口的大小。
        
b_q5 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_q5")#filter数量。
         


conv_q = tf.nn.conv2d(                        #卷积operation

    embedded_chars_expanded_q,           #the sentence of conv2input。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q5,                                        #[filter_height, filter_width, in_channels, out_channels]   

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_q")
conv_a = tf.nn.conv2d(                       #卷积operation

    embedded_chars_expanded_a,           #the sentence of conv2。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q5,

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_a")
conv_aa = tf.nn.conv2d(                       #卷积operation

    embedded_chars_expanded_aa,           #the sentence of conv2。input是一个四维Tensor：[batch, in_height, in_width, in_channels]

    W_q5,

    strides=[1, 1, 1, 1],

    padding="VALID",

    name="conv_aa")
# Apply nonlinearity

h_q = tf.nn.relu(tf.nn.bias_add(conv_q, b_q5), name="relu1")
h_a = tf.nn.relu(tf.nn.bias_add(conv_a, b_q5), name="relu2")
h_aa = tf.nn.relu(tf.nn.bias_add(conv_aa,b_q5), name="relu3")
# Maxpooling over the outputs

pooled_q = tf.nn.max_pool(

    h_q,

    ksize=[1, sequence_length - 5 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_q")
pooled_a = tf.nn.max_pool(

    h_a,

    ksize=[1, sequence_length - 5 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_a")
pooled_aa = tf.nn.max_pool(

    h_aa,

    ksize=[1, sequence_length - 5 + 1, 1, 1],

    strides=[1, 1, 1, 1],

    padding='VALID',

    name="pool_aa")
pooled_outputs_q.append(pooled_q)#循环3次 将pool值组合到一起。
pooled_outputs_a.append(pooled_a)
pooled_outputs_aa.append(pooled_aa)   
# Combine all the pooled features

num_filters_total = num_filters * len(filter_sizes)         #final shape[num_filters_total]

h_pool_q = tf.concat(3,pooled_outputs_q)
h_pool_a = tf.concat(3,pooled_outputs_a)
h_pool_aa = tf.concat(3,pooled_outputs_aa)  
h_pool_flat_q = tf.reshape(h_pool_q, [-1, num_filters_total])
h_pool_flat_a = tf.reshape(h_pool_a, [-1, num_filters_total])
h_pool_flat_aa = tf.reshape(h_pool_aa, [-1, num_filters_total])

# Add dropout

with tf.name_scope("dropout"):

    h_drop_q = tf.nn.dropout(h_pool_flat_q, dropout_keep_prob)
    h_drop_a = tf.nn.dropout(h_pool_flat_a, dropout_keep_prob)
    h_drop_aa = tf.nn.dropout(h_pool_flat_aa, dropout_keep_prob)
#predict 
with tf.name_scope("y_prediction"):
    h_drop_q_len = tf.sqrt(tf.reduce_sum(tf.mul(h_drop_q, h_drop_q), 1)) #计算向量长度Batch模式
    h_drop_a_len = tf.sqrt(tf.reduce_sum(tf.mul(h_drop_a, h_drop_a), 1))
    h_drop_aa_len = tf.sqrt(tf.reduce_sum(tf.mul(h_drop_aa, h_drop_aa), 1))
   
    h_mul_qa = tf.reduce_sum(tf.mul(h_drop_q, h_drop_a), 1) #计算向量的点乘Batch模式
    h_mul_qaa = tf.reduce_sum(tf.mul(h_drop_q, h_drop_aa), 1)

    cos_12 = tf.div(h_mul_qa , tf.mul(h_drop_q_len, h_drop_a_len), name="scores") #计算向量夹角Batch模式
    cos_13 = tf.div(h_mul_qaa , tf.mul(h_drop_q_len, h_drop_aa_len), name="scores2")
# CalculateMean cross-entropy loss  
with tf.name_scope("loss"):
    zero = tf.constant(0,  dtype=tf.float32)
    margin = tf.constant(0.05,  dtype=tf.float32)
    losses = tf.maximum(zero, tf.sub(margin, tf.sub(cos_12, cos_13)))
    losss = tf.reduce_sum(losses) + l2_reg_lambda * l2_loss
    print('loss ', losss)
    
# Accuracy
with tf.name_scope("accuracy"):
    correct = tf.equal(zero, losses)
    accuracyy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")   

##train
# Define Training procedure

global_step = tf.Variable(0, name="global_step", trainable=False)

optimizer = tf.train.AdamOptimizer(1e-3)

grads_and_vars = optimizer.compute_gradients(losss)

train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Output directory for models 

timestamp = str(int(time.time())) 

out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

print("Writing to {}\n".format(out_dir))
# Checkpoint directory.if not exist ，we create it

checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))



if not os.path.exists(checkpoint_dir):

    os.makedirs(checkpoint_dir)
#定义saver
saver = tf.train.Saver() #typically saver=tf.train.Saver()
# Write vocabulary
vocab_processor.save(os.path.join(out_dir, "vocab"))

#start train_op
# first need to do is Initialize all variables
sess=tf.Session()
sess.run(tf.global_variables_initializer()) 
#define train function
def train_step(xq_batch,xa_batch ,xaa_batch):
    feed_dict = {

      input_xq: xq_batch,
      input_xa: xa_batch,
      input_xaa: xaa_batch,

      dropout_keep_prob: FLAGS.dropout_keep_prob

    }

    _, step, loss,accuracy = sess.run(

        [train_op, global_step, losss,accuracyy],

        feed_dict)

    time_str = datetime.datetime.now().isoformat()

    print("{}: step {}, loss {:g},accuracy {:g}".format(time_str, step, loss,accuracy))

# Generate batches

batches = data_helpers.batch_iter(

    list(zip(x_shuffled_q,x_shuffled_a,x_shuffled_aa)), FLAGS.batch_size, FLAGS.num_epochs)

# Training loop. For each batch...
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

for batch in batches:

    x_batch_q, x_batch_a,x_batch_aa = zip(*batch)

    train_step(x_batch_q, x_batch_a,x_batch_aa)

    current_step = tf.train.global_step(sess, global_step)

    if current_step % FLAGS.checkpoint_every == 0: 
        path = saver.save(sess,'tmp/model.ckpt')
        print("Saved model checkpoint to {}\n".format(path))   
print(sess.run(Wq))              
              
                

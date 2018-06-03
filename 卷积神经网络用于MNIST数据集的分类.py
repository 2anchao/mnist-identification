# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:24:46 2018

@author: Alex
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
batch_size=100
number_batch=mnist.train.num_examples // batch_size
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean=tf.reduce_mean(var)#求变量的平均值
        tf.summary.scalar("mean",mean)#第一个位置是名称
        with tf.name_scope("stddev"):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev",stddev)#标准差
        tf.summary.scalar("max",tf.reduce_max(var))#最大值
        tf.summary.scalar("min",tf.reduce_min(var))#最小值
        tf.summary.histogram("histogram",var)#直方图
#初始化权值变量
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#初始化偏差值变量
def bias_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#卷积层
def conv2d(x,W):#2d表示是一个二维的卷积操作
    #we can get from help:conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    #x=tensor:`[batch, in_height, in_width, in_channels]`四维的值
    #W=filter:`[filter_height, filter_width, in_channels, out_channels]`
    #stride[0]=stride[3]=1,stride[1]代表x方向的步长，stride[2]代表y方向的步长
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
#池化层
def max_pool(x):
    #ksize代表filter大小
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
with tf.name_scope("input"):
    x=tf.placeholder(tf.float32,[None,784])
    y=tf.placeholder(tf.float32,[None,10])
with tf.name_scope("convolution_layer"):
#改变x的格式转化为2D的向量[batch,in_height,in_width,in_channels]
    x_image=tf.reshape(x,[-1,28,28,1])
    #初始化第一个卷积层的权值和偏执值
    with tf.name_scope("c_layer1_W_b"):
        W_conv1=weight_variable([5,5,1,32])#5*5*1的filer，输出32个channel，表示32个卷积核从1个平面抽取特征
        variable_summaries(W_conv1)
        b_conv1=bias_variable([32])#每一个卷积核一个偏置值
        variable_summaries(b_conv1)
    #把x_image和权值向量进行卷积，在加上偏置值，然后应用于relu激活函数
    with tf.name_scope("c_layer1_output"):
        h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
        h_pool1=max_pool(h_conv1)
        
    with tf.name_scope("c_layer2_W_b"):
        #第二个卷积层
        W_conv2=weight_variable([5,5,32,64])#表示64个卷积核从32个平面抽取特征
        variable_summaries(W_conv2)
        b_conv2=bias_variable([64])
        variable_summaries(b_conv2)
    with tf.name_scope("c_layer2_output"):
        h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
        h_pool2=max_pool(h_conv2)
#28*28的照片第一次卷积后还是28*28，第一次池化后变为14*14
#第二次卷积后为14*14，第二次池化后变为7*7
#得到64张7*7的特征图
with tf.name_scope("full_connection_layer1"):
    with tf.name_scope("f_layer1_W_b"):
        #初始化第一个全连接层的权值
        W_fcl=weight_variable([7*7*64,1024])#上一层有64*7*7个特征，把这些特征连接到我们的1024个神经元
        variable_summaries(W_fcl)
        b_fcl=bias_variable([1024])#1024个节点
        variable_summaries(b_fcl)
    with tf.name_scope("f_layer1_output"):
#把池化层2的输出扁平化成1维
        h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
        #求第一个全连接层的输出
        h_fcl=tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl)+b_fcl)
#keep_drop用来表示神经元的输出概率
keep_drop=tf.placeholder(tf.float32)
h_fcl_drop=tf.nn.dropout(h_fcl,keep_drop)
with tf.name_scope("full_connection_layer2"):
    with tf.name_scope("f_layer2_W_b"):
        #初始化第二个全连接层
        W_fcl2=weight_variable([1024,10])
        variable_summaries(W_fcl2)
        b_fcl2=bias_variable([10])
        variable_summaries(b_fcl2)
    with tf.name_scope("f_layer2_output"):
        #第二个全连接层的输出
        prediction=tf.nn.softmax(tf.matmul(h_fcl,W_fcl2)+b_fcl2)
with tf.name_scope("loss_function"):
    #交叉熵损失函数
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar("cross_entropy",cross_entropy)
    #将我们的loss函数最小化
    ir=tf.Variable(0.001,tf.float32)
    optimizer=tf.train.AdamOptimizer(ir)
    train=optimizer.minimize(cross_entropy)
with tf.name_scope("accuracy"):
    #查看准确率
    #第一步是将结果放在一个布尔列表中
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    #第二步计算概率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#tf.cast是进行类型转化
merged=tf.summary.merge_all()#统计
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('logs/',sess.graph)#生成我们graph的图
    for cycle in range(5):
        sess.run(tf.assign(ir,0.001*(0.95**cycle)))
        for batch in range(number_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            summary,_=sess.run([merged,train],feed_dict={x:batch_xs,y:batch_ys,keep_drop:1.0})
        writer.add_summary(summary,cycle)#把我们的summary和周期写到文件logs里面
            
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_drop:1.0})
        print("训练周期是："+str(cycle)+"正确率是："+str(acc))
        


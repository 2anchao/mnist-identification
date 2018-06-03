# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:53:59 2018

@author: Alex
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)#one_hot是把标签转化为0，1的列矩阵
#每个批次的大小，批次就是代表我们训练时图片不是一张一张来训练的，批次的大小我们可以自己定义
batch_size=100
#计算一共有多少个批次
number_batch=mnist.train.num_examples // batch_size
#命名空间
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
        
with tf.name_scope("input"):
    #定义两个placehoder
    x=tf.placeholder(tf.float32,[None,784],name='x-input')#None代表的是行，和我们的批次有关
    y=tf.placeholder(tf.float32,[None,10],name="y-input")#标签
    keep_prob=tf.placeholder(tf.float32)#我们在这里加一个placeholder，用来传入我们dropout所需要的参数
    ir=tf.Variable(0.001,dtype=tf.float32)#定义一个学习率的变量
#为了凸显我们dropout的作用，我们设计一个复杂的神经网络结构，实际上实现手写数字不需要这么复杂的神经网络
        
with tf.name_scope("layer"):
    
    with tf.name_scope("wight1"):
        W1=tf.Variable(tf.truncated_normal([784,1000],stddev=0.1),name="W1")#tf.truncated_normal是加入随机的截断正态分布，标准差是0.1
        variable_summaries(W1)
    with tf.name_scope("biases1"):
        b1=tf.Variable(tf.zeros([1,1000])+0.1,name="b1")
        variable_summaries(b1)
    with tf.name_scope("Z1"):
        Z1=tf.matmul(x,W1)+b1
    with tf.name_scope("output1"):
        A1=tf.nn.relu(Z1)
    with tf.name_scope("A1_drop"):
        A1_drop=tf.nn.dropout(A1,keep_prob)#这里我们用前面定义好的placeholder来传入我们想要工作的神经元个数，设置为0.5，就是50%的神经元在工作
    
    with tf.name_scope("wight2"):
        W2=tf.Variable(tf.truncated_normal([1000,500],stddev=0.1),name="W2")
        variable_summaries(W2)
    with tf.name_scope("biases2"):
        b2=tf.Variable(tf.zeros([1,500])+0.1,name="b2")
        variable_summaries(b2)
    with tf.name_scope("Z2"):
        Z2=tf.matmul(A1,W2)+b2
    with tf.name_scope("output2"):
        A2=tf.nn.tanh(Z2)
    with tf.name_scope("A2_drop"):
        A2_drop=tf.nn.dropout(A2,keep_prob)
    
    with tf.name_scope("wight3"):
        W3=tf.Variable(tf.truncated_normal([500,10],stddev=0.1),name="W3")
        variable_summaries(W3)
    with tf.name_scope("biases3"):
        b3=tf.Variable(tf.zeros([1,10])+0.1,name="b3")
        variable_summaries(b3)
    with tf.name_scope("Z3"):
        Z3=tf.matmul(A2,W3)+b3
    with tf.name_scope("output"):
        predict=tf.nn.softmax(Z3)
'''#定义二次代价函数，输出的正确率是0.9348
loss=tf.reduce_mean(tf.square(y-predict))
optimizer=tf.train.GradientDescentOptimizer(0.2)
train=optimizer.minimize(loss)'''
#定义一个分析参数的函数
with tf.name_scope("train_layer"):
    with tf.name_scope("loss_function"):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predict))#labels是我们加的标签，logits是我们得到的预测值
        tf.summary.scalar("loss",loss)
    #我们将最后用的代价函数从二次代价函数改为和softmax激活函数所匹配的的最大似然函数
    with tf.name_scope("optimizer"):
        optimizer=tf.train.AdamOptimizer(ir)#1e-2=0.01的学习率
    #optimizer=tf.train.GradientDescentOptimizer(0.2)#在这里我们更换优化器
    with tf.name_scope("minimize_loss"):
        train=optimizer.minimize(loss)#我们可以看到正确率大约从93%提高到了96%
with tf.name_scope("initial"):
    init=tf.global_variables_initializer()#初始化变量
#求准确率的方法,结果存放在一个布尔型列表中
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_predict"):
        correct_predict=tf.equal(tf.argmax(y,1),tf.argmax(predict,1))#tf.equal是用来判断两个值是否相等。不想等返回false。tf.argmax用来返回y或者predict中的最大值的索引，也就是位置
#求准确率
    with tf.name_scope("accuracy"):
        accuracy=tf.reduce_mean(tf.cast(correct_predict,tf.float32))#tf.cast是进行类型转化
        tf.summary.scalar("accuracy",accuracy)
merged=tf.summary.merge_all()#统计

with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('logs/',sess.graph)#生成我们graph的图
    for cycle in range(5):#循环31个周期
        sess.run(tf.assign(ir,0.001*(0.95**cycle)))#随着迭代次数增加，减小learning rate，使我们更好的找到loss的最小值
        for batch in range(number_batch):#将所有的批次循环一遍
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)#先获得一个批次，mnist.train.next_batch是将图片的数据保存到batch_xs，将图片的标签保存到batch_ys
            summary,_=sess.run([merged,train],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        writer.add_summary(summary,cycle)#把我们的summary和周期写到文件logs里面
        learning_rate=sess.run(ir)
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})#加入测试集
        train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("循环周期是："+str(cycle)+"测试集正确率是："+str(test_acc)+"训练集正确率是："+str(train_acc)+"学习率是:"+str(learning_rate))
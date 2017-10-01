import numpy as np
import json
from sklearn.manifold import TSNE
import pickle
import tensorflow as tf
import os, os.path
import pickle
import matplotlib.pyplot as plt
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML


train_size = 48000
test_size = 5000
batch_size = 512
N0 = 64 #image height or width
N1 = 25 #face grid size
n_channel = 3
threshold = 2.0
goodUntil = 20
display_step = train_size/batch_size
epoch = 50
training_itr = display_step * epoch
decay = 0.85
schedule = train_size/batch_size
iter_display = display_step/10


print('TensorFlow version: {0}'.format(tf.__version__))

def normalization(x):
    x = x/255.0
    x_mean = np.mean(x, axis=0)
    x = np.subtract(x, x_mean)
    x = x/np.std(x)
    return x

def importTrainingData():
    npzfile = np.load("../train_and_val.npz")
    train_eye_left = npzfile["train_eye_left"]
    train_eye_right = npzfile["train_eye_right"]
    train_face = npzfile["train_face"]
    train_face_mask = npzfile["train_face_mask"]
    train_y = npzfile["train_y"]

    train_eye_left = normalization(train_eye_left)
    train_eye_right = normalization(train_eye_right)
    train_face = normalization(train_face)

    return train_eye_left, train_eye_right, train_face, train_face_mask, train_y

def importTestingData():
    npzfile = np.load("../train_and_val.npz")
    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_face_mask = npzfile["val_face_mask"]
    val_y = npzfile["val_y"]

    val_eye_left = normalization(val_eye_left)
    val_eye_right = normalization(val_eye_right)
    val_face = normalization(val_face)

    return val_eye_left, val_eye_right, val_face, val_face_mask, val_y

def convLayer(x, W, b, s = 1, name="conv", padding="VALID"):
    with tf.name_scope(name) as scope:
    	x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=padding)
    	x = tf.nn.bias_add(x, b)
    	return x

def activation(x, name="relu"):
    return tf.nn.relu(x, name=name)

def maxPoolLayer(x, d=2, s=2, name="maxPool", padding="VALID"):
    return tf.nn.max_pool(x, ksize=[1, d, d, 1], strides=[1, s, s, 1], padding=padding, name=name)

def conv_net_left_eye(x, weights, biases):
	#convolution layer 1, pooling layer 1
	convEL1 = convLayer(x, weights['CEL1'], biases['BCEL1'], s=1, name="EL_Conv1")
	convEL1 = activation(convEL1, name="EL_relu1")
	convEL1 = maxPoolLayer(convEL1, d=2, s=2, name="EL_pool1")

	#convolution layer 2, pooling layer 2
	convEL2 = convLayer(convEL1, weights['CEL2'], biases['BCEL2'], s=1, name="EL_Conv2")
	convEL2 = activation(convEL2, name="EL_relu2")
	convEL2 = maxPoolLayer(convEL2, d=2, s=2, name="EL_pool2")

	#convolution layer 3
	convEL3 = convLayer(convEL2, weights['CEL3'], biases['BCEL3'], s=1, name="EL_Conv3")
	convEL3 = activation(convEL3, name="EL_relu3")
	#fully connected layer of left eye
	fcEL = tf.reshape(convEL3, [-1,7744])
	with tf.name_scope("EL_FC") as scope:
		fcEL = tf.matmul(fcEL, weights['FCEL'])
		fcEL = fcEL + biases['BFCEL']
        fcEL = activation(fcEL, name="ELFC_relu1")

	return fcEL

def conv_net_right_eye(x, weights, biases):
	#convolution layer 1, pooling layer 1
	convER1 = convLayer(x, weights['CER1'], biases['BCER1'], s=1, name="ER_Conv1")
	convER1 = activation(convER1, name="ER_relu1")
	convER1 = maxPoolLayer(convER1, d=2, s=2, name="ER_pool1")

	#convolution layer 2, pooling layer 2
	convER2 = convLayer(convER1, weights['CER2'], biases['BCER2'], s=1, name="ER_Conv2")
	convER2 = activation(convER2, name="ER_relu2")
	convER2 = maxPoolLayer(convER2, d=2, s=2, name="ER_pool2")

	#convolution layer 3
	convER3 = convLayer(convER2, weights['CER3'], biases['BCER3'], s=1, name="ER_Conv3")
	convER3 = activation(convER3, name="ER_relu3")

	#fully connected layer of right eye
	fcER = tf.reshape(convER3, [-1, 7744])
	with tf.name_scope("ER_FC") as scope:
		fcER = tf.matmul(fcER, weights['FCER'])
		fcER = fcER + biases['BFCER']
        fcER = activation(fcER, name="ER_FC_relu1")
	return fcER

def conv_net_eye(x, weights, biases): #fully connected layers for eye images
    with tf.name_scope("Eyes_FC1") as scope:
        convE1 = tf.add(tf.matmul(x, weights['CE1']), biases['BE1'])
        convE1 = activation(convE1)
    with tf.name_scope('Eyes_FC2'):
        convE2 = tf.add(tf.matmul(convE1, weights['CE2']), biases['BE2'])
        convE2 = activation(convE2)
    return convE2

def conv_net_face(x, weights, biases):
	#convolution layer 1, pooling layer 1
	convF1 = convLayer(x, weights['CF1'], biases['BCF1'], s=1, name="F_Conv1")
	convF1 = activation(convF1, name="F_relu1")
	convF1 = maxPoolLayer(convF1, d=2, s=2, name="F_pool1")

	#convolution layer 2, pooling layer 2
	convF2 = convLayer(convF1, weights['CF2'], biases['BCF2'], s=1, name="F_Conv2")
	convF2 = activation(convF2, name="F_relu2")
	convF2 = maxPoolLayer(convF2, d=2, s=2, name="F_pool2")

	#convolution layer 3
	convF3 = convLayer(convF2, weights['CF3'], biases['BCF3'], s=1, name="F_Conv3")
	convF3 = activation(convF3, name="F_relu3")

	#fully connected layer 1 of face
	fcF1 = tf.reshape(convF3, [-1, 7744])
	with tf.name_scope("F_FC1") as scope:
		fcF1 = tf.matmul(fcF1, weights['FCF1'])
		fcF1 = fcF1 + biases['BFCF1']
        fcF1 = activation(fcF1)
	#fully connected layer 2 of right eye
	with tf.name_scope("F_FC2") as scope:
		fcF2 = tf.matmul(fcF1, weights['FCF2'])
		fcF2 = fcF2 + biases['BFCF2']   
        fcF2 = activation(fcF2)

	return fcF2 


def conv_net_face_grid(x, weights, biases):
	x = tf.reshape(x, [-1,625])
	with tf.name_scope("F_Mask1") as scope:
		layer1 = tf.add(tf.matmul(x, weights['FG1']), biases['BFG1'])
        layer1 = activation(layer1)
	with tf.name_scope("F_Mask2") as scope:
		layer2 = tf.add(tf.matmul(layer1, weights['FG2']), biases['BFG2'])
        layer2 = activation(layer2)

	return layer2


def conv_net_four_path(eye_left, eye_right, face, face_mask, weights, biases):
    ELout = conv_net_left_eye(eye_left, weights['WEL'], biases['BEL'])
    ERout = conv_net_right_eye(eye_right, weights['WER'], biases['BER'])
    eyeOut = tf.concat([ELout, ERout], 1)
    eyeOut = conv_net_eye(eyeOut, weights['WE'], biases['BE'])

    FaceOut = conv_net_face(face, weights['WF'], biases['BF'])
    FaceGridOut = conv_net_face_grid(face_mask, weights['WFG'], biases['BFG'])

    fc0 = tf.concat([eyeOut, FaceOut, FaceGridOut], 1)
    with tf.name_scope("FC1") as scope:
        fc1 = tf.add(tf.matmul(fc0, weights['WOUT']['fc1']), biases['BOUT']['Bfc1'])
        fc1 = activation(fc1, name="FC_relu1")
    with tf.name_scope("FC2"):
        fc2 = tf.add(tf.matmul(fc1, weights['WOUT']['fc2']), biases['BOUT']['Bfc2'])
        fc2 = activation(fc2, name="FC_relu2")
    with tf.name_scope("Output") as scope:
        out = tf.add(tf.matmul(fc2, weights['WOUT']['out']), biases['BOUT']['Bout'])

    return out

def conv_net_two_path(eye_left, eye_right, face, face_mask, weights, biases):
    ELout = conv_net_left_eye(eye_left, weights['WEL'], biases['BEL'])
    ERout = conv_net_right_eye(eye_right, weights['WER'], biases['BER'])
    eyeOut = tf.concat([ELout, ERout], 1)
    eyeOut = conv_net_eye(eyeOut, weights['WE'], biases['BE'])

    with tf.name_scope("FC1") as scope:
        fc1 = tf.add(tf.matmul(eyeOut, weights['WOUT2PATH']['fc1']), biases['BOUT']['Bfc1'])
        fc1 = activation(fc1, name="FC_relu1")
    with tf.name_scope("FC2"):
        fc2 = tf.add(tf.matmul(fc1, weights['WOUT2PATH']['fc2']), biases['BOUT']['Bfc2'])
        fc2 = activation(fc2, name="FC_relu2")
    with tf.name_scope("Output") as scope:
        out = tf.add(tf.matmul(fc2, weights['WOUT2PATH']['out']), biases['BOUT']['Bout'])

    return out

def conv_net_one_path(eye_left, eye_right, face, face_mask, weights, biases):
    ELout = conv_net_left_eye(eye_left, weights['WEL'], biases['BEL'])

    with tf.name_scope("FC1") as scope:
        fc1 = tf.add(tf.matmul(ELout, weights['WOUT1PATH']['fc1']), biases['BOUT']['Bfc1'])
        fc1 = activation(fc1, name="FC_relu1")
    with tf.name_scope("FC2"):
        fc2 = tf.add(tf.matmul(fc1, weights['WOUT1PATH']['fc2']), biases['BOUT']['Bfc2'])
        fc2 = activation(fc2, name="FC_relu2")
    with tf.name_scope("Output") as scope:
        out = tf.add(tf.matmul(fc2, weights['WOUT1PATH']['out']), biases['BOUT']['Bout'])

    return out

def trainCNN(init_rate, path_choice, tb_writer):
    global N0, N1, n_channel, train_size, batch_size, decay, schedule, training_itr, display_step, threshold, goodUntil, iter_display

    eye_left = tf.placeholder(tf.float32, [None, N0, N0 ,n_channel], name='eye_left')
    eye_right = tf.placeholder(tf.float32, [None, N0, N0 ,n_channel], name='eye_right')
    face = tf.placeholder(tf.float32, [None, N0, N0 ,n_channel], name='face')
    face_mask = tf.placeholder(tf.float32, [None, N1, N1], name='face_mask')

    y = tf.placeholder(tf.float32, [None, 2], name='location')
    
    eta = tf.placeholder(tf.float32)
    weightsEL = {
        'CEL1': tf.get_variable("CEL1",shape=[5,5,3,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'CEL2': tf.get_variable("CEL2",shape=[5,5,32,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'CEL3': tf.get_variable("CEL3",shape=[3,3,32,64], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'FCEL': tf.get_variable("FCEL",shape=[7744,256],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN"))
    }
    weightsER = {
        'CER1': tf.get_variable("CER1",shape=[5,5,3,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'CER2': tf.get_variable("CER2",shape=[5,5,32,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'CER3': tf.get_variable("CER3",shape=[3,3,32,64], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'FCER': tf.get_variable("FCER",shape=[7744, 256],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN"))
    }
    weightsE = {
        'CE1': tf.get_variable("CE1",shape=[512, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'CE2': tf.get_variable("CE2",shape=[256, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN"))    
    }
    weightsF = {
        'CF1': tf.get_variable("CF1",shape=[5,5,3,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'CF2': tf.get_variable("CF2",shape=[5,5,32,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'CF3': tf.get_variable("CF3",shape=[3,3,32,64], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'FCF1': tf.get_variable("FCF1",shape=[7744,256],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'FCF2': tf.get_variable("FCF2",shape=[256,512],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
    }
    weightsFG = {
        'FG1': tf.get_variable("FG1",shape=[625, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'FG2': tf.get_variable("FG2",shape=[256, 128], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN"))
    }
    weightsOut = {
        'fc1': tf.get_variable("fc1",shape=[896, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'fc2': tf.get_variable("fc2",shape=[256, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'out': tf.get_variable("out",shape=[256, 2], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN"))    
    }

    weightsOut2Path = {
        'fc1': tf.get_variable("fc2p1",shape=[256, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'fc2': tf.get_variable("fc2p2",shape=[256, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'out': tf.get_variable("out2p",shape=[256, 2], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN"))    
    }

    weightsOut1Path = {
        'fc1': tf.get_variable("fc1p1",shape=[256, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'fc2': tf.get_variable("fc1p2",shape=[256, 256], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'out': tf.get_variable("out1p",shape=[256, 2], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN"))    
    }

    biasesEL = {
        'BCEL1': tf.Variable(tf.constant(0.005, shape=[32]),name="BCEL1"),
        'BCEL2': tf.Variable(tf.constant(0.005, shape=[32]), name="BCEL2"),
        'BCEL3': tf.Variable(tf.constant(0.005, shape=[64]), name="BCEL13"),
        'BFCEL': tf.Variable(tf.constant(0.005, shape=[256]), name="BFCEL")
    }
    biasesER = {
        'BCER1': tf.Variable(tf.constant(0.005, shape=[32]),name="BCEL1"),
        'BCER2': tf.Variable(tf.constant(0.005, shape=[32]), name="BCEL2"),
        'BCER3': tf.Variable(tf.constant(0.005, shape=[64]), name="BCEL13"),
        'BFCER': tf.Variable(tf.constant(0.005, shape=[256]), name="BFCEL")
    }
    biasesE = {
        'BE1': tf.Variable(tf.constant(0.005, shape=[256]),name="BE1"),
        'BE2': tf.Variable(tf.constant(0.005, shape=[256]),name="BE2")
    }
    biasesF = {
        'BCF1': tf.Variable(tf.constant(0.005, shape=[32]),name="BCF1"),
        'BCF2': tf.Variable(tf.constant(0.005, shape=[32]), name="BCF2"),
        'BCF3': tf.Variable(tf.constant(0.005, shape=[64]), name="BCF3"),
        'BFCF1': tf.Variable(tf.constant(0.005, shape=[256]), name="BFCF1"),
        'BFCF2': tf.Variable(tf.constant(0.005, shape=[512]), name="BFCF2")
    }
    biasesFG = {
        'BFG1': tf.Variable(tf.constant(0.005, shape=[256]),name="BFG1"),
        'BFG2': tf.Variable(tf.constant(0.005, shape=[128]), name="BFG2")
    }
    biasesOut = {
        'Bfc1': tf.Variable(tf.constant(0.005, shape=[256]),name="Bout1"),
        'Bfc2': tf.Variable(tf.constant(0.005, shape=[256]), name="Bout2"),
        'Bout': tf.Variable(tf.constant(0.005, shape=[2]), name="Bout2")
    }
    weights = { 'WEL' : weightsEL, 
                'WER': weightsER,
                'WE': weightsE,
                'WF': weightsF,
                'WFG': weightsFG,
                'WOUT': weightsOut,
                'WOUT2PATH': weightsOut2Path,
                'WOUT1PATH': weightsOut1Path
    }

    biases = {  'BEL' : biasesEL, 
                'BER': biasesER,
                'BE': biasesE,
                'BF': biasesF,
                'BFG': biasesFG,
                'BOUT': biasesOut
    }

    print '================Run: #path='+path_choice+', rate='+str(init_rate)+'================'
    print 'Building graph...' 
    if path_choice=='1':
        predict_op = conv_net_one_path(eye_left, eye_right, face, face_mask, weights, biases)
    elif path_choice=='2':
         predict_op = conv_net_two_path(eye_left, eye_right, face, face_mask, weights, biases)
    elif path_choice=='4':
         predict_op = conv_net_four_path(eye_left, eye_right, face, face_mask, weights, biases)
    else:
        print 'Path number error'

    #saving: preparing model
    tf.get_collection("validation_nodes")
    tf.add_to_collection("validation_nodes", eye_left)
    tf.add_to_collection("validation_nodes", eye_right)
    tf.add_to_collection("validation_nodes", face)
    tf.add_to_collection("validation_nodes", face_mask)
    tf.add_to_collection("validation_nodes", predict_op)
    
    cost = tf.reduce_mean(tf.reduce_sum(tf.pow(predict_op-y, 2), axis=1))/2.0
    with tf.name_scope("train") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost)
    ave_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(predict_op-y, 2), axis=1)))

    tf.summary.scalar('Loss', cost)
    tf.summary.scalar('Error', ave_error)

    init = tf.global_variables_initializer()
    
    print 'Loading data...'
    train_eye_left, train_eye_right, train_face, train_face_mask, train_y = importTrainingData()
    test_eye_left, test_eye_right, test_face, test_face_mask, test_y = importTestingData()

    train_err_list = []
    test_err_list = []
    loss_list = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #merged = tf.summary.merge_all()
        merged_summary = tf.summary.merge_all()
        tb_writer.add_graph(sess.graph)

        sess.run(init)

        step = 1
        good = 0
        rate = init_rate
        print "\n================================================================"
        print 'Start training... batch size: '+str(batch_size), ', learning rate: '+ str(rate)
        print "================================================================"
        while step < training_itr:
            choices = np.random.choice(train_size, batch_size, replace = False)
            batch_x_EL = train_eye_left[choices]
            batch_x_ER = train_eye_right[choices]
            batch_x_F = train_face[choices]
            batch_x_FG = train_face_mask[choices]
            batch_y = train_y[choices]

            if step % iter_display == 0:
                print "Iteration:", step
                s = sess.run(merged_summary, feed_dict={eye_left: batch_x_EL, eye_right: batch_x_ER, face: batch_x_F, face_mask: batch_x_FG, y: batch_y})
                tb_writer.add_summary(s, step)
            
            sess.run(optimizer, feed_dict={eye_left: batch_x_EL, eye_right: batch_x_ER, face: batch_x_F, face_mask: batch_x_FG, y: batch_y, eta: rate})
            if step % display_step == 0:
                loss, train_err = sess.run([cost, ave_error], feed_dict={eye_left: batch_x_EL, eye_right: batch_x_ER, face: batch_x_F, face_mask: batch_x_FG, y: batch_y})
                test_err = sess.run(ave_error, feed_dict={eye_left: test_eye_left[0:1000], eye_right: test_eye_right[0:1000], face: test_face[0:1000], face_mask: test_face_mask[0:1000], y: test_y[0:1000]})
                test_err += sess.run(ave_error, feed_dict={eye_left: test_eye_left[1000:2000], eye_right: test_eye_right[1000:2000], face: test_face[1000:2000], face_mask: test_face_mask[1000:2000], y: test_y[1000:2000]})
                test_err += sess.run(ave_error, feed_dict={eye_left: test_eye_left[2000:3000], eye_right: test_eye_right[2000:3000], face: test_face[2000:3000], face_mask: test_face_mask[2000:3000], y: test_y[2000:3000]})
                test_err += sess.run(ave_error, feed_dict={eye_left: test_eye_left[3000:4000], eye_right: test_eye_right[3000:4000], face: test_face[3000:4000], face_mask: test_face_mask[3000:4000], y: test_y[3000:4000]})
                test_err += sess.run(ave_error, feed_dict={eye_left: test_eye_left[4000:5000], eye_right: test_eye_right[4000:5000], face: test_face[4000:5000], face_mask: test_face_mask[4000:5000], y: test_y[4000:5000]})
                test_err /= 5.0
                print "============================================="
                print "Epoch "+str(step/display_step) + ": Loss= "+"{:.6f}".format(loss) + ", train error= "+"{:.5f}".format(train_err) + ", test error= "+"{:.5f}".format(test_err)
                print "============================================="
                train_err_list.append(train_err)
                test_err_list.append(test_err)
                loss_list.append(loss)
                if test_err < threshold:
                    good = good + 1
                    if good >= goodUntil:
                        break
            if step % schedule == 0:
                if rate < 0.0001:
                    if test_err_list[-1] > 2.5:
                        print 'learning rate remains: '+str(rate)
                    else:
                        rate = rate*decay
                        print 'learning rate decreased to: '+ str(rate)
                else:
                    rate = rate*decay
                    print 'learning rate decreased to: '+ str(rate)

            step += 1
        print "Done!"
        print "Test error:"
        test_err = sess.run(ave_error, feed_dict={eye_left: test_eye_left[0:1000], eye_right: test_eye_right[0:1000], face: test_face[0:1000], face_mask: test_face_mask[0:1000], y: test_y[0:1000]})
        test_err += sess.run(ave_error, feed_dict={eye_left: test_eye_left[1000:2000], eye_right: test_eye_right[1000:2000], face: test_face[1000:2000], face_mask: test_face_mask[1000:2000], y: test_y[1000:2000]})
        test_err += sess.run(ave_error, feed_dict={eye_left: test_eye_left[2000:3000], eye_right: test_eye_right[2000:3000], face: test_face[2000:3000], face_mask: test_face_mask[2000:3000], y: test_y[2000:3000]})
        test_err += sess.run(ave_error, feed_dict={eye_left: test_eye_left[3000:4000], eye_right: test_eye_right[3000:4000], face: test_face[3000:4000], face_mask: test_face_mask[3000:4000], y: test_y[3000:4000]})
        test_err += sess.run(ave_error, feed_dict={eye_left: test_eye_left[4000:5000], eye_right: test_eye_right[4000:5000], face: test_face[4000:5000], face_mask: test_face_mask[4000:5000], y: test_y[4000:5000]})
        test_err /= 5.0
        print "{:.6f}".format(test_err)

        #save model
        #save_path = saver.save(sess, 'rate='+str(init_rate)+', #path='+path_choice+"/my_model")
        save_path = saver.save(sess, 'my_model_'+'rate='+str(init_rate)+'_#path='+path_choice)

        sess.close()
        #save accuracy lists and loss list, for plotting
        with open('train_err','wb') as fp:
            pickle.dump(train_err_list, fp)
        with open('test_err','wb') as fp:
            pickle.dump(test_err_list, fp)
        with open('loss','wb') as fp:
            pickle.dump(loss_list, fp)


def plotAcc(save_dir):
    print "Plotting accuracy figures..."
    with open ('train_err', 'rb') as fp:
        train_err_list = pickle.load(fp)
    with open ('test_err', 'rb') as fp:
        test_err_list = pickle.load(fp)
    with open ('loss', 'rb') as fp:
        loss_list = pickle.load(fp)
    train_err = train_err_list
    test_err = test_err_list
    loss = loss_list

    t = list(range(len(loss)))
    fig, ax1 = plt.subplots()
    ax1.plot(t, loss,'r')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='r')
    ax1.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    ax2.plot(t, train_err,'g')
    ax2.plot(t, test_err,'b')
    ax2.set_ylabel('error / cm', color='k')
    ax2.tick_params('y', colors='k')
    fig.tight_layout()
    plt.legend(['Training Error', 'Testing Error'], loc = 1)
    fig.savefig(save_dir+'_loss_and_error.pdf')
    plt.close(fig)

    
def main():
    #delete certain values in the lists to run a specific configuration
    for rate in [0.0002, 0.001, 0.005]: #0.001 is the best
        for path_choice in ['1', '2', '4']: #'4' is the best
            hparam_str = 'rate='+str(rate)+', #path='+path_choice
            writer = tf.summary.FileWriter("tb_vis/"+hparam_str)
            trainCNN(rate, path_choice, writer)
            plotAcc('rate='+str(rate)+'_#path='+path_choice)
            tf.reset_default_graph()


if __name__ == '__main__':
    main()

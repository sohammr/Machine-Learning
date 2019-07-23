import pandas as pd
import numpy as np
import argparse
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import nn_ops, gen_nn_ops
from tensorflow.python.framework import ops
from sklearn.metrics import f1_score
tf.random.set_random_seed(1234)
np.random.seed(0)


def main():

	np.random.seed(1234)
	
	parser=argparse.ArgumentParser()
	parser.add_argument('--lr')
	parser.add_argument('--batch_size')
	parser.add_argument('--epochs')
	parser.add_argument('--init')
	parser.add_argument('--save_dir')
	parser.add_argument('--train')
	parser.add_argument('--val')
	parser.add_argument('--test')
	parser.add_argument('--dataAugment')
	parser.add_argument('--early_stop')
	
	args=parser.parse_args()
	lr = float(args.lr)
	global early_stop
	early_stop = '0'	#Default value
	early_stop = args.early_stop
	batch_size = int(args.batch_size)
	epochs = int(args.epochs)
	global init
	init = args.init
	#Default dataAugment = 1, can be over written by user
	dataAugment = '1'
	dataAugment = args.dataAugment
	train_path = args.train
	test_path = args.test
	val_path = args.val
	save_dir = args.save_dir


	train0 = pd.read_csv(train_path)
	test = pd.read_csv(test_path)
	valid = pd.read_csv(val_path)

	global X_train
	X_train = train0.iloc[:,1:12289]
	X_test = test.iloc[:,1:12289]
	global X_val
	X_val = valid.iloc[:,1:12289]
	y_train = train0.iloc[:,12289]
	y_val = valid.iloc[:,12289]
	X_train = (X_train - X_train.mean())/(X_train.std())
	X_test = (X_test - X_train.mean())/(X_train.std())
	X_val = (X_val - X_train.mean())/(X_train.std())
	X_train=np.asarray(X_train)
	X_train=np.reshape(X_train,(8769,64,64,3))
	X_test=np.asarray(X_test)
	X_test=np.reshape(X_test,(972,64,64,3))
	X_val=np.asarray(X_val)
	X_val=np.reshape(X_val,(956,64,64,3))
	y_train=np.asarray(y_train)
	y_val=np.asarray(y_val)
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(X_val.shape)

	global img_size
	img_size=64

	global y_val_oh
	y_val_oh = np.array( [ OneHotIt(y) for y in y_val  ])
	global y_train_oh
	y_train_oh = np.array( [ OneHotIt(y) for y in y_train  ])

	num_channels=3
	num_classes = 20
	filter_size_conv1=5
	num_filters_conv1=32

	filter_size_conv2=5
	num_filters_conv2=32

	filter_size_conv3=3
	num_filters_conv3=64

	filter_size_conv4=3
	num_filters_conv4=64

	filter_size_conv5=3
	num_filters_conv5=64

	filter_size_conv6=3
	num_filters_conv6=128

	fc_layer_size = 256

	x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
	 
	y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
	y_true_cls = tf.argmax(y_true, dimension=1)


	layer_conv1,pre_relu1 = create_convolutional_layer(input=x,
				   num_input_channels=num_channels,
				   conv_filter_size=filter_size_conv1,
				   num_filters=num_filters_conv1,use_pooling=False)
	 
	layer_conv2,pre_relu2 = create_convolutional_layer(input=layer_conv1,
				   num_input_channels=num_filters_conv1,
				   conv_filter_size=filter_size_conv2,
				   num_filters=num_filters_conv2)

	 
	layer_conv3,pre_relu3= create_convolutional_layer(input=layer_conv2,
				   num_input_channels=num_filters_conv2,
				   conv_filter_size=filter_size_conv3,
				   num_filters=num_filters_conv3,use_pooling=False)

	layer_conv4,pre_relu4= create_convolutional_layer(input=layer_conv3,
				   num_input_channels=num_filters_conv3,
				   conv_filter_size=filter_size_conv4,
				   num_filters=num_filters_conv4)


	layer_conv5,pre_relu5= create_convolutional_layer(input=layer_conv4,
				   num_input_channels=num_filters_conv4,
				   conv_filter_size=filter_size_conv5,
				   num_filters=num_filters_conv5,use_pooling=False)

	layer_conv6,pre_relu6= create_convolutional_layer(input=layer_conv5,
				   num_input_channels=num_filters_conv5,
				   conv_filter_size=filter_size_conv6,
				   num_filters=num_filters_conv6,pad='VALID')

			  
	layer_flat = create_flatten_layer(layer_conv6)
	 
	layer_fc1 = create_fc_layer(input=layer_flat,
						 num_inputs=layer_flat.get_shape()[1:4].num_elements(),
						 num_outputs=fc_layer_size,
						 use_relu=True)

	layer_fc2 = create_fc_layer(input=layer_fc1,
						 num_inputs=fc_layer_size,
						 num_outputs=num_classes,
						 use_relu=False)

	batch_norm = tf.layers.batch_normalization(layer_fc2, training=True)


	session = tf.Session()
	session.run(tf.global_variables_initializer())

	y_pred = tf.nn.softmax(batch_norm,name='y_pred')
	y_pred_cls = tf.argmax(y_pred, dimension=1)

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=batch_norm,
														labels=y_true)
	cost = tf.reduce_mean(cross_entropy)

	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

	correct_prediction = tf.equal(y_pred_cls, y_true_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver()
	session.run(tf.global_variables_initializer())

	def show_progress(epoch, feed_dict_train, feed_dict_validate,train_loss, val_loss):
		acc = session.run(accuracy, feed_dict=feed_dict_train)
		val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
		
		y_true_v = session.run(y_true_cls, feed_dict=feed_dict_validate)
		y_pred_v = session.run(y_pred_cls, feed_dict=feed_dict_validate)
		f1score = f1_score(y_true_v, y_pred_v, average = None).mean()  

		msg = "Training Epoch {0} --- Training Accuracy: {1:.3f},  Training Loss: {2:.3f}, Validation Accuracy: {3:.3f},  Validation Loss: {4:.3f},  Validation f1: {5:.3f}"
		print(msg.format(epoch + 1, acc, train_loss,val_acc, val_loss, f1score))

	def train(num_epochs,batch_size, save_dir):

		n_batches = int(X_train.shape[0]/batch_size)

		max_f1 = 0.3
		#Patience parameter
		p = 5
		val_acc_list = []
		count = 0

		for j in range(num_epochs):
			
			for i in range(n_batches):

				x_batch = X_train[i*batch_size:(i+1)*batch_size]
				y_true_batch =y_train_oh[i*batch_size:(i+1)*batch_size]         
				# feed_dict_tr = {x: x_batch, y_true: y_true_batch}         
				session.run(optimizer, feed_dict={x: x_batch, y_true: y_true_batch} )

				# feed_dict_val = {x: X_val, y_true: y_val_oh}
			val_loss = session.run(cost, feed_dict={x: X_val, y_true: y_val_oh})
			train_loss = session.run(cost, feed_dict={x: x_batch, y_true: y_true_batch})
			epoch = j   
			show_progress(epoch, {x: x_batch, y_true: y_true_batch},  {x: X_val, y_true: y_val_oh},train_loss, val_loss)
					

	train(epochs,batch_size,save_dir)

	var = [v for v in tf.trainable_variables() if v.name == 'Variable:0'][0]
	filters = session.run(var)

	min = filters.min(axis = tuple([0,1,2]))
	max = filters.max(axis = tuple([0,1,2]))
	filters = (filters - min)/(max-min)

	def rgb2gray(rgb):
		return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

	fig, axes = plt.subplots(4, 8)
	for i, ax in enumerate(axes.flat):

		img = rgb2gray(filters[:,:,:,i])
		thres = 0.6*np.ones(img.shape)
		img = np.minimum(img,thres)
		ax.imshow(img,cmap=plt.get_cmap('gray'))
		ax.set_xticks([])
		ax.set_yticks([])

	fig.savefig(save_dir+'conv1_filters.png')


	#Guided backprop
	grad = tf.placeholder(tf.float32, [1,7,7,128])

	grad_pre_relu6 = tf.where(0. < grad, gen_nn_ops.relu_grad(grad,pre_relu6), tf.zeros(grad.get_shape()))
	grad_conv5=tf.gradients(pre_relu6,layer_conv5,grad_ys=grad_pre_relu6)[0]

	grad_pre_relu5 = tf.where(0. < grad_conv5, gen_nn_ops.relu_grad(grad_conv5,pre_relu5), tf.zeros((1,16,16,64)))
	grad_conv4=tf.gradients(pre_relu5,layer_conv4,grad_ys=grad_pre_relu5)[0]

	grad_pre_relu4 = tf.where(0. < grad_conv4, gen_nn_ops.relu_grad(grad_conv4,pre_relu4), tf.zeros((1,16,16,64)))
	grad_conv3=tf.gradients(pre_relu4,layer_conv3,grad_ys=grad_pre_relu4)[0]

	grad_pre_relu3 = tf.where(0. < grad_conv3, gen_nn_ops.relu_grad(grad_conv3,pre_relu3), tf.zeros((1,32,32,64)))
	grad_conv2=tf.gradients(pre_relu3,layer_conv2,grad_ys=grad_pre_relu3)[0]

	grad_pre_relu2 = tf.where(0. < grad_conv2, gen_nn_ops.relu_grad(grad_conv2,pre_relu2), tf.zeros((1,32,32,32)))
	grad_conv1=tf.gradients(pre_relu2,layer_conv1,grad_ys=grad_pre_relu2)[0]

	grad_pre_relu1 = tf.where(0. < grad_conv1, gen_nn_ops.relu_grad(grad_conv1,pre_relu1), tf.zeros((1,64,64,32)))
	grad_guided_bp=tf.gradients(pre_relu1,x,grad_ys=grad_pre_relu1)[0]


	X_train_1=np.reshape(X_train[3],(1,64,64,3))
	neuron_conv6=np.zeros((1,7,7,128))

	feed_dict_1={x:X_train_1}
	conv6_neurons=session.run(layer_conv6,feed_dict_1)

	a=np.argsort(conv6_neurons,axis=None)
	neuronlist=[]
	for i in range(10):
		l=np.unravel_index(a[-(i*100+1)],conv6_neurons.shape)
		neurons=np.zeros((1,7,7,128))
		neurons[l[0]][l[1]][l[2]][l[3]]=conv6_neurons[l[0]][l[1]][l[2]][l[3]]
		neuronlist.append(neurons)

	fig2, axes = plt.subplots(5,2)
	for i, ax in enumerate(axes.flat):
		im=session.run(grad_guided_bp,feed_dict={x:X_train_1,grad:neuronlist[i]})
		im=(im-im.min())/(im.max()-im.min())
		img = im[0]
		ax.imshow(img)
		ax.set_xticks([])
		ax.set_yticks([])

	fig2.savefig(save_dir+'guided_backprop.png')

def OneHotIt(y):
	y_new = np.zeros(20)
	y_new[y] = 1
	return y_new

def create_weights(shape):
	if init == '1':
		initializer = tf.contrib.layers.xavier_initializer(seed = 0)
	if init == '2':
		initializer = tf.contrib.layers.variance_scaling_initializer(seed = 0)
	return tf.Variable(initializer(shape))

def create_biases(size):
	return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
			   num_input_channels, 
			   conv_filter_size,        
			   num_filters, use_pooling=True , pad='SAME'):  
	
	## We shall define the weights that will be trained using create_weights function.
	weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
	## We create biases using the create_biases function. These are also trained.
	biases = create_biases(num_filters)
 
	## Creating the convolutional layer
	pre_relu = tf.nn.conv2d(input=input,
					 filter=weights,
					 strides=[1, 1, 1, 1],
					 padding=pad)
 
	pre_relu += biases
  
	if use_pooling:
		pre_relu = tf.nn.max_pool(value=pre_relu,
							   ksize=[1, 2, 2, 1],
							   strides=[1, 2, 2, 1],
							   padding='SAME')
	layer = tf.nn.relu(pre_relu)
 
	return layer,pre_relu


def create_flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer = tf.reshape(layer, [-1, num_features])
 
	return layer

def create_fc_layer(input,          
			 num_inputs,    
			 num_outputs,
			 use_relu=True):

	weights = create_weights(shape=[num_inputs, num_outputs])
	biases = create_biases(num_outputs)
 
	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
 
	return layer

if __name__=='__main__':
	main()
import pandas as pd
import numpy as np
import argparse
import math
import tensorflow as tf
import matplotlib.pyplot as plt
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

	if dataAugment == '1':
		flipped_images = flip_images(X_train)
		scaled_imgs = central_scale_images(X_train, [0.6,0.75,1.2])

		X_train=np.concatenate((X_train,flipped_images),axis=0)
		X_train=np.concatenate((X_train,scaled_imgs),axis=0)
		print(X_train.shape)

		y1=np.repeat(y_train,3)
		y2=np.repeat(y_train,3)
		y_train=np.concatenate((y_train,y1),axis=0)
		y_train=np.concatenate((y_train,y2),axis=0)
		print(y_train.shape)

		from sklearn.utils import shuffle
		X_train , y_train = shuffle(X_train,y_train, random_state = 0)

	global y_val_oh
	y_val_oh = np.array( [ OneHotIt(y) for y in y_val  ])
	global y_train_oh
	y_train_oh = np.array( [ OneHotIt(y) for y in y_train  ])

	num_channels=3
	num_classes = 20
	filter_size_conv1=5
	num_filters_conv1=64

	filter_size_conv2=5
	num_filters_conv2=64

	filter_size_conv3=5
	num_filters_conv3=128

	filter_size_conv4=5
	num_filters_conv4=128

	filter_size_conv5=3
	num_filters_conv5=256

	filter_size_conv6=3
	num_filters_conv6=256

	fc_layer_size1 = 512

	x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
	 
	y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
	y_true_cls = tf.argmax(y_true, dimension=1)


	layer_conv1 = create_convolutional_layer(input=x,
				 num_input_channels=num_channels,
				 conv_filter_size=filter_size_conv1,
				 num_filters=num_filters_conv1,use_pooling=False)
	 
	layer_conv2 = create_convolutional_layer(input=layer_conv1,
				 num_input_channels=num_filters_conv1,
				 conv_filter_size=filter_size_conv2,
				 num_filters=num_filters_conv2)

	 
	layer_conv3= create_convolutional_layer(input=layer_conv2,
				 num_input_channels=num_filters_conv2,
				 conv_filter_size=filter_size_conv3,
				 num_filters=num_filters_conv3,use_pooling=False)

	layer_conv4= create_convolutional_layer(input=layer_conv3,
				 num_input_channels=num_filters_conv3,
				 conv_filter_size=filter_size_conv4,
				 num_filters=num_filters_conv4)


	layer_conv5= create_convolutional_layer(input=layer_conv4,
				 num_input_channels=num_filters_conv4,
				 conv_filter_size=filter_size_conv5,
				 num_filters=num_filters_conv5,use_pooling=False)

	layer_conv6= create_convolutional_layer(input=layer_conv5,
				 num_input_channels=num_filters_conv5,
				 conv_filter_size=filter_size_conv6,
				 num_filters=num_filters_conv6,pad='VALID')

			
	layer_flat = create_flatten_layer(layer_conv6)
	 
	layer_fc1 = create_fc_layer(input=layer_flat,
						 num_inputs=layer_flat.get_shape()[1:4].num_elements(),
						 num_outputs=fc_layer_size1,
						 use_relu=True)


	layer_fc2 = create_fc_layer(input=layer_fc1,
						 num_inputs=fc_layer_size1,
						 num_outputs=num_classes,
						 use_relu=False)

	session = tf.Session()
	session.run(tf.global_variables_initializer())

	w_fc1 = session.graph.get_tensor_by_name('Variable_12:0')
	b_fc1 = session.graph.get_tensor_by_name('Variable_13:0')

	y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
	y_pred_cls = tf.argmax(y_pred, dimension=1)

	unreg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
	l2_loss = 0.01 * (tf.nn.l2_loss(w_fc1)+ tf.nn.l2_loss(b_fc1))

	loss = tf.add(unreg_loss, l2_loss, name='loss')

	cost = tf.reduce_mean(loss)

	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

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
		train_loss_list = []
		val_loss_list = []

		for j in range(num_epochs):
			
			for i in range(n_batches):

				x_batch = X_train[i*batch_size:(i+1)*batch_size]
				y_true_batch =y_train_oh[i*batch_size:(i+1)*batch_size]         
				# feed_dict_tr = {x: x_batch, y_true: y_true_batch}         
				session.run(optimizer, feed_dict={x: x_batch, y_true: y_true_batch} )

				if i %100== 0: 
					# feed_dict_val = {x: X_val, y_true: y_val_oh}
					val_loss = session.run(cost, feed_dict={x: X_val, y_true: y_val_oh})

					train_loss = session.run(cost, feed_dict={x: x_batch, y_true: y_true_batch})
					val_acc = session.run(accuracy, feed_dict={x: X_val, y_true: y_val_oh})

					#Early stopping criteria
					if early_stop == '1':
						val_acc_list.append(val_acc)
						if len(val_acc_list)>5:
							if val_acc_list[count-5]>val_acc_list[count]:
								break
					count = count + 1
					
					epoch = j   
					show_progress(epoch, {x: x_batch, y_true: y_true_batch},  {x: X_val, y_true: y_val_oh},train_loss, val_loss)

					y_true_v = session.run(y_true_cls, feed_dict={x: X_val, y_true: y_val_oh})
					y_pred_v = session.run(y_pred_cls, feed_dict={x: X_val, y_true: y_val_oh})
					f1score = f1_score(y_true_v, y_pred_v, average = None).mean()
					if f1score > max_f1:
						print('Saving the model')
						max_f1 = f1score
						# feed_dict_test={x:X_test}
						y_pred_test=session.run(y_pred_cls,feed_dict={x:X_test})
						y_pred_test = pd.DataFrame({'id': range(len(y_pred_test)) , 'label': y_pred_test})
						y_pred_test.to_csv(save_dir+'cnnpred.csv', index = False)

						saver.save(session, save_dir+'model')

			train_loss2 = session.run(cost, feed_dict={x: X_train[:3000], y_true: y_train_oh[:3000]})
			val_loss2 = session.run(cost, feed_dict={x: X_val, y_true: y_val_oh})
			train_loss_list.append(train_loss2)
			val_loss_list.append(val_loss2)

		fig = plt.figure()
		plt.plot(train_loss_list, c = 'blue', label = 'Train Loss')
		plt.plot(val_loss_list, c = 'red', label = 'Validation Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('Loss')
		plt.legend()
		fig.savefig(save_dir+'loss_plot.png')

	train(epochs,batch_size,save_dir)

	"""## **Test Predictions**"""

	# feed_dict_test={x:X_test}
	# y_pred_test=session.run(y_pred_cls,feed_dict=feed_dict_test)
	# y_pred_test = pd.DataFrame({'id': range(len(y_pred_test)) , 'label': y_pred_test})
	# y_pred_test.to_csv('vrm_cnnpred2.csv', index = False)


"""## **Data Augmentation**"""

def flip_images(X_imgs):
	X_flip = []
	tf.reset_default_graph()
	X = tf.placeholder(tf.float32, shape = (img_size, img_size, 3))
	tf_img1 = tf.image.flip_left_right(X)
	tf_img2 = tf.image.flip_up_down(X)
	tf_img3 = tf.image.transpose_image(X)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for img in X_imgs:
			flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
			X_flip.extend(flipped_imgs)
	X_flip = np.array(X_flip, dtype = np.float32)
	return X_flip

def central_scale_images(X_imgs, scales):
	# Various settings needed for Tensorflow operation
	boxes = np.zeros((len(scales), 4), dtype = np.float32)
	for index, scale in enumerate(scales):
		x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
		x2 = y2 = 0.5 + 0.5 * scale
		boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
	box_ind = np.zeros((len(scales)), dtype = np.int32)
	crop_size = np.array([img_size, img_size], dtype = np.int32)
	
	X_scale_data = []
	tf.reset_default_graph()
	X = tf.placeholder(tf.float32, shape = (1, img_size, img_size, 3))
	
	tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for img_data in X_imgs:
			batch_img = np.expand_dims(img_data, axis = 0)
			scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
			X_scale_data.extend(scaled_imgs)
	
	X_scale_data = np.array(X_scale_data, dtype = np.float32)
	return X_scale_data


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
	
	weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
	biases = create_biases(num_filters)

	layer = tf.nn.conv2d(input=input,
					 filter=weights,
					 strides=[1, 1, 1, 1],
					 padding=pad)
 
	layer += biases
	
	if use_pooling:
		layer = tf.nn.max_pool(value=layer,
								 ksize=[1, 2, 2, 1],
								 strides=[1, 2, 2, 1],
								 padding='SAME')

	layer = tf.layers.batch_normalization(layer, training=True)

	layer = tf.nn.relu(layer)
	return layer

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
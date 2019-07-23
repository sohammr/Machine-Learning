import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core

def main():

	tf.random.set_random_seed(1234)
	np.random.seed(0)

	parser=argparse.ArgumentParser()
	parser.add_argument('--lr')
	parser.add_argument('--batch_size')
	parser.add_argument('--epochs')
	parser.add_argument('--init')
	parser.add_argument('--dropout_prob')
	parser.add_argument('--decode_method')
	parser.add_argument('--save_dir')
	parser.add_argument('--train')
	parser.add_argument('--val')
	parser.add_argument('--test')
	
	args=parser.parse_args()
	save_dir = args.save_dir
	train = pd.read_csv(args.train)
	test = pd.read_csv(args.test)
	valid = pd.read_csv(args.val)
	X_train_list = []
	y_train_list = []
	X_val_list = []
	y_val_list = []
	X_test_list = []
	eng_vocab = np.array([])
	hin_vocab = np.array([])
	test_vocab = np.array([])

	for i in range(train.shape[0]):
		X_train_list.append(np.array(train.iloc[i,1].split(' ')))  
		eng_local_vocab = np.unique(np.array(train.iloc[i,1].split(' ')))
		eng_vocab = np.unique(np.concatenate([eng_vocab, eng_local_vocab]))
		y_train_list.append(np.array(train.iloc[i,2].split(' ')))
		hin_local_vocab = np.unique(np.array(train.iloc[i,2].split(' ')))
		hin_vocab = np.unique(np.concatenate([hin_vocab, hin_local_vocab]))
		
	for i in range(valid.shape[0]):
		X_val_list.append(np.array(valid.iloc[i,1].split(' ')))
		y_val_list.append(np.array(valid.iloc[i,2].split(' ')))
		
	for i in range(test.shape[0]):
		X_test_list.append(np.array(test.iloc[i,1].split(' ')))

	eng_vocab = np.concatenate([np.array(['*']),eng_vocab], axis = 0)
	hin_vocab = np.concatenate([np.array(['*','$']),hin_vocab], axis = 0)

	X_train = np.array(X_train_list)
	X_val = np.array(X_val_list)
	X_test = np.array(X_test_list)
	ind = 0
	for i in X_test:
		for j in range(len(i)):
			if i[j]=='Ü':
				X_test[ind][j] = 'U'
		ind = ind+1

	def pad(X_train, max_length):
		X_train_pad = []
		for i in X_train:
			if len(i)<max_length:
				i = np.concatenate([i, np.repeat('*',max_length-len(i))])
				X_train_pad.append(i)
			else:
				X_train_pad.append(i)
		return np.array(X_train_pad)

	def code(X_train_pad, vocab_dict):
		X_train_coded = []
		for i in X_train_pad:
			k = np.array([vocab_dict[j] for j in i])
			X_train_coded.append(k)
		return np.array(X_train_coded)

	eng_vocab_size = len(eng_vocab)
	max_length = 0
	max_word = 'None'
	for i in X_train:
		if len(i)>max_length:
			max_length = len(i)
			max_word = i
			
	#Padding
	X_train_pad = pad(X_train, max_length)
	X_val_pad = pad(X_val, max_length)
	X_test_pad = pad(X_test, max_length)
	codes = np.arange(eng_vocab_size)
	vocab_dict = {i:j for i,j in zip(eng_vocab,codes)}
	X_train_coded = code(X_train_pad, vocab_dict)
	X_val_coded = code(X_val_pad, vocab_dict)
	X_test_coded = code(X_test_pad, vocab_dict)

	def code_y(y_target_pad, hin_vocab_dict):
		y_target_coded = []
		for i in y_target_pad:
			k = np.array([hin_vocab_dict[j] for j in i])
			y_target_coded.append(k)
		return np.array(y_target_coded)

	y_train = np.array(y_train_list)
	hin_vocab_size = len(hin_vocab)
	max_decod_length = 0
	max_word = 'None'
	for i in y_train:
		if len(i)>max_decod_length:
			max_decod_length = len(i)
			max_word = i

	#Padding
	y_train_pad = []
	for i in y_train:
		if len(i)<max_length:
			i = np.concatenate([i, np.repeat('*',max_length-len(i)+1)])
		i = np.concatenate([np.array(['$']),i])
		y_train_pad.append(i)
	y_train_pad = np.array(y_train_pad)

	y_target_pad = []
	for i in y_train:
		if len(i)<max_length:
			i = np.concatenate([i, np.repeat('*',max_length-len(i)+1)])
		i =  np.concatenate([i, np.array(['*'])])
		y_target_pad.append(i)
	y_target_pad = np.array(y_target_pad)

	codes = np.arange(hin_vocab_size)
	hin_vocab_dict = {i:j for i,j in zip(hin_vocab,codes)}
	y_train_coded = code_y(y_train_pad, hin_vocab_dict)
	y_target_coded = code_y(y_target_pad, hin_vocab_dict)

	y_val = np.array(y_val_list)
	#Padding
	y_val_pad = []
	for i in y_val:
		if len(i)<max_length:
			i = np.concatenate([i, np.repeat('*',max_length-len(i)+1)])
		i = np.concatenate([np.array(['$']),i])
		y_val_pad.append(i)
	y_val_pad = np.array(y_val_pad)

	y_target_pad_val = []
	for i in y_val:
		if len(i)<max_length:
			i = np.concatenate([i, np.repeat('*',max_length-len(i)+1)])
		i =  np.concatenate([i, np.array(['*'])])
		y_target_pad_val.append(i)
	y_target_pad_val = np.array(y_target_pad_val)

	codes = np.arange(hin_vocab_size)
	hin_vocab_dict = {i:j for i,j in zip(hin_vocab,codes)}
	y_val_coded = code_y(y_val_pad, hin_vocab_dict)
	y_target_val_coded = code_y(y_target_pad_val, hin_vocab_dict)

	hparams = tf.contrib.training.HParams(
			batch_size=int(args.batch_size),#128,
			encoder_length=max_length,
			decoder_length=max_decod_length+1,
			num_units=256,
			src_vocab_size=len(eng_vocab),
			embedding_size=256,
			tgt_vocab_size=len(hin_vocab),
			learning_rate = float(args.lr),#0.001,
			keep_prob=1-float(args.dropout_prob),#1,
			max_gradient_norm = 5.0,
			beam_width =9,
			use_attention = True,
			early_stop_loss = False,
			early_stop_acc = True,
	)
	tgt_sos_id = hin_vocab_dict['$'].astype(np.int32)
	tgt_eos_id = vocab_dict['*'].astype(np.int32)
	tf.reset_default_graph()

	"""## Encoder"""

	encoder_inputs = tf.placeholder(tf.int32, shape=(hparams.encoder_length, None), name="encoder_inputs")
	b_size=tf.shape(encoder_inputs)[1]

	if args.init == '1':
		initializer = tf.contrib.layers.xavier_initializer(seed = 0)
	else:
		initializer = tf.contrib.layers.variance_scaling_initializer(seed = 0)
	embedding_encoder  = tf.Variable(initializer([hparams.src_vocab_size, hparams.embedding_size]))

	encoder_emb_inputs = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)

	# LSTM cell.
	tr=True
	# Forward direction cell
	lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units, forget_bias=1.0)
	if tr==True:
		lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=hparams.keep_prob)
	# Backward direction cell
	lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units, forget_bias=1.0)
	if tr==True:
		lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=hparams.keep_prob)

	(out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, encoder_emb_inputs, time_major=True, dtype=tf.float32)

	encoder_outputs = tf.concat((out_fw, out_bw), -1)
	bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
	bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
	encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)

	"""## Decoder"""

	decoder_inputs = tf.placeholder(tf.int32, shape=(hparams.decoder_length, None), name="decoder_inputs")
	decoder_lengths = tf.placeholder(tf.int32, shape=(None,), name="decoder_length")

	embedding_decoder  = tf.Variable(initializer([hparams.tgt_vocab_size, hparams.embedding_size]))

	decoder_emb_inputs = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

	projection_layer = layers_core.Dense(hparams.tgt_vocab_size, use_bias=False)
	helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, decoder_lengths, time_major=True)

	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units*2)
	if tr==True:
		decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=hparams.keep_prob)

	"""## Attention Mechanism"""

	if hparams.use_attention:
		attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
		attention_mechanism = tf.contrib.seq2seq.LuongAttention((hparams.num_units)*2, attention_states,memory_sequence_length=None)
		decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=(hparams.num_units)*2)
		initial_state = decoder_cell.zero_state(b_size, tf.float32).clone(cell_state=encoder_state)
	else:
		initial_state = encoder_state

	decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state,output_layer=projection_layer)
	final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
	logits = final_outputs.rnn_output
	target_labels = tf.placeholder(tf.int32, shape=(None, hparams.decoder_length))
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_labels, logits=logits)
	cost = tf.reduce_mean(loss)
	optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate).minimize(cost)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	def show_progress(epoch, train_loss, val_loss,acc):
			msg = "Training Epoch {0} --- Training Loss: {1:.3f} --- Validation Loss: {2:.3f} --- Validation Accuracy: {3:.3f}"
			print(msg.format(epoch + 1,train_loss, val_loss,acc))

	# Inference
	inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder,tf.fill([b_size], tgt_sos_id), tgt_eos_id)
	inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, inference_helper, initial_state,output_layer=projection_layer)
	source_sequence_length = hparams.encoder_length
	maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
	outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, maximum_iterations=maximum_iterations)
	translations = outputs.sample_id

	def train(num_epochs,batch_size):
		n_batches = int(X_train_coded.shape[0]/batch_size)+1
		n_batches_val = int(X_val_coded.shape[0]/batch_size)+1
		train_loss=[];val_loss=[];val_acc_list=[]
		max_val_acc = 0
		for j in range(num_epochs):
			train_loss_b=[];val_loss_b=[]
			tr=True
			for i in range(n_batches):
				feed_dict_tr = {encoder_inputs: X_train_coded[batch_size*i:batch_size*(i+1)].T,
								target_labels: y_target_coded[batch_size*i:batch_size*(i+1)],
								decoder_inputs: y_train_coded[batch_size*i:batch_size*(i+1)].T,
								decoder_lengths: np.ones(( X_train_coded[batch_size*i:batch_size*(i+1)].T.shape[1]), dtype=int) * hparams.decoder_length}
				_, train_loss_batch = sess.run([optimizer, cost], feed_dict=feed_dict_tr)
				train_loss_b.append(train_loss_batch)
				
			tr=False
			for k in range(n_batches_val):    
				feed_dict_val = {
							encoder_inputs: X_val_coded[batch_size*k:batch_size*(k+1)].T,
							target_labels: y_target_val_coded[batch_size*k:batch_size*(k+1)],
							decoder_inputs: y_val_coded[batch_size*k:batch_size*(k+1)].T,
							decoder_lengths: np.ones(( X_val_coded[batch_size*k:batch_size*(k+1)].T.shape[1]), dtype=int) * hparams.decoder_length}
				val_loss_batch = sess.run(cost, feed_dict=feed_dict_val)
				val_loss_b.append(val_loss_batch)
			tl=np.mean(np.array(train_loss_b))
			train_loss.append(tl)
			vl=np.mean(np.array(val_loss_b))
			val_loss.append(vl)
			epoch = j   
			
			y_pred = []
			for n in range(n_batches_val):
				feed_dict = {encoder_inputs: X_val_coded[n*batch_size:(n+1)*batch_size].T}
				y_pred_local= sess.run([translations], feed_dict=feed_dict)[0]
				y_pred.append(y_pred_local)
			inv_dict = {v: k for k, v in hin_vocab_dict.items()}
			y_pred_uncoded = []
			for m in y_pred:
				for i in m:
					k = np.array([inv_dict[j] for j in i])
					y_pred_uncoded.append(k)
			y_pred_uncoded = np.array(y_pred_uncoded)
			
			sub = []
			for i in y_pred_uncoded:
				temp = str()
				for j in i:
					if j== '*':
						break
					else:
						temp = temp + j
						temp = temp + ' '
				sub.append(temp[:-1])
			submit = []
			for i in range(len(y_pred_uncoded)):
				submit.append([sub[i]])
				
			sum=0
			for i in range(len(submit)):
				if submit[i][0]==valid['HIN'][i]:
					sum=sum+1
			val_acc=sum/(len(submit))

			if val_acc > max_val_acc:
				print('Saving weights')
				max_val_acc = val_acc
				saver = tf.train.Saver()
				saver.save(sess, args.save_dir+'/best_model')
			
			p = 7
			#Early stopping criteria
			if hparams.early_stop_acc:
				val_acc_list.append(val_acc)
				if len(val_acc_list)>p:
					if val_acc_list[epoch-p]>val_acc_list[epoch]:
						break
			#Early stopping criteria
			if hparams.early_stop_loss:
				if len(val_loss)>p:
					if val_loss[epoch-p]<val_loss[epoch]:
						break
			
			show_progress(epoch, tl, vl,val_acc) 
		return train_loss,val_loss

	print('Now training')
	train_loss,val_loss=train(int(args.epochs),hparams.batch_size)

	# saver = tf.train.Saver()
	# saver.save(sess, args.save_dir+'/model')

	## Test
	to_pred = X_test_coded
	y_pred = []
	batch_size = hparams.batch_size
	n_batches = int(len(to_pred)/batch_size)
	for i in range(n_batches):
		feed_dict = {encoder_inputs: to_pred[i*batch_size:(i+1)*batch_size].T,}
		y_pred_local = sess.run([translations], feed_dict=feed_dict)[0]
		y_pred.append(y_pred_local)
		
	remainder = to_pred[(i+1)*batch_size:]
	new_batch = np.concatenate([remainder, np.zeros([batch_size - remainder.shape[0], max_length])])
	feed_dict = {encoder_inputs: new_batch.T,}
	y_pred_local = sess.run([translations], feed_dict=feed_dict)[0]
	y_pred.append(y_pred_local)

	inv_dict = {v: k for k, v in hin_vocab_dict.items()}

	y_pred_uncoded = []
	for m in y_pred:
		for i in m:
			k = np.array([inv_dict[j] for j in i])
			y_pred_uncoded.append(k)
	y_pred_uncoded = np.array(y_pred_uncoded)
	y_pred_uncoded = y_pred_uncoded[:to_pred.shape[0]]

	sub = []
	for i in y_pred_uncoded:
		temp = str()
		for j in i:
			if j== '*':
				break
			else:
				temp = temp + j
				temp = temp + ' '
		sub.append(temp[:-1])

	submit = []
	for i in range(len(y_pred_uncoded)):
		submit.append([i, sub[i]])

	submitdf = pd.DataFrame(submit,columns = ['id','HIN'])
	submitdf.to_csv(save_dir+'/sub2.csv', index = False)

if __name__=='__main__':
	main()
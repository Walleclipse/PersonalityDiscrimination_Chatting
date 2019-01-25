from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F
from mxnet.contrib import text
import numpy as np
import time
import threading
import os

NUM_ATTRIBUTES = 4
NUM_CLASSES = 2
UNK_ID = 1

from allennlp.data import vocabulary
import pickle
from collections import Counter
# from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn import metrics
from sklearn.model_selection import train_test_split
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()  # using ELMo pretrained embedding

#options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
#weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)
batch_size = 64
max_l_word = 500
seq_len = 20
EPOCHS = 10

vocab_size = 10003


def get_data(all_data):
	all_set = []
	seq_list = []
	data_x = []
	data_y = []
	for index, row in all_data.iterrows():
		posts = []
		for raw in row['posts']:
			if len(raw) >= 5:
				posts += raw
		if len(posts) >= 100:
			all_set.append([posts, list(row['type'])])
			data_x.append(posts)
			data_y.append(list(row['type']))
			seq_list.extend(row['posts'])
	
	print(len(all_set), len(seq_list))
	
	# all_set:[ [[post1, post2, ... post30],[I,J,K,L]],  [[post1, post2, ... post30],[I,J,K,L]], ]
	def count_token(train_tokenized, token_counter):
		for i, sample in enumerate(train_tokenized):
			if i % 10000 == 0:
				print(i)
			for token in sample:
				if token not in token_counter:
					token_counter[token] = 1
				else:
					token_counter[token] += 1
		return token_counter
	
	token_counter = Counter()
	token_counter = count_token(seq_list, token_counter)
	print(len(token_counter))
	return all_set, seq_list, token_counter



import copy


def data_loader(data_set,vocab, batch_size=8, max_l_word=300):
	def pad_unk(x, emb_size=50):
		if len(x) > max_l_word:
			b_ind = np.random.randint(0, len(x) - max_l_word)
			return x[b_ind:b_ind + max_l_word]
		else:
			# return x + [[UNK_ID] * emb_size] * (max_l_word - len(x))
			_ad = [[UNK_ID] * emb_size] * (max_l_word - len(x))
			return np.concatenate((x, np.array(_ad)), axis=0)
	
	def pad_unk2(x):
		if len(x) > max_l_word:
			b_ind = np.random.randint(0, len(x) - max_l_word)
			return x[b_ind:b_ind + max_l_word]
		else:
			return x + [UNK_ID] * (max_l_word - len(x))
	
	data = data_set.copy()
	L = len(data_set)
	max_l_word=300
	while True:
		np.random.shuffle(data)
		batch_start = 0
		batch_end = batch_size
		while batch_end < L:
			batch_data = data[batch_start:batch_end]
			elmo_emb = [];
			labels = [];
			feat0 = [];
			for row in batch_data:
				xx, yy = row
				if len(xx) >max_l_word:
					b_ind = np.random.randint(0, len(xx) - max_l_word)
					xx= xx[b_ind:b_ind + max_l_word]
				vector = elmo.embed_sentence(xx)[2]
				vector = pad_unk(vector, emb_size=1024)
				# print('vect',vector.shape)
				elmo_emb.append(vector)
				f0 = vocab.to_indices(xx)
				f0 = pad_unk2(f0)
				feat0.append(f0)
				label_IE = 0 if yy[0] == 'I' else 1
				label_NS = 0 if yy[1] == 'N' else 1
				label_TF = 0 if yy[2] == 'T' else 1
				label_JP = 0 if yy[3] == 'J' else 1
				labels.append([label_IE, label_NS, label_TF, label_JP])
			# feat = batch_to_ids(feat).cpu().numpy()
			# print(feat.shape)
			elmo_emb = np.array(elmo_emb)
			# print('feat',features.shape)
			# print('elmo',elmo(features)['elmo_representations'].shape)
			feat0 = np.array(feat0)
			labels = np.array(labels)
			yield (elmo_emb, feat0, labels)
			batch_start += batch_size
			batch_end += batch_size


def tile(a, dim, n_tile):
	init_dim = a.size(dim)
	repeat_idx = [1] * a.dim()
	repeat_idx[dim] = n_tile
	a = a.repeat(*(repeat_idx))
	order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
	return torch.index_select(a, dim, order_index)


class FeedForwardNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
		super(FeedForwardNetwork, self).__init__()
		self.dropout_rate = dropout_rate
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)
	
	def forward(self, x):
		x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
		x_proj = self.linear2(x_proj)
		return x_proj


class SeqAttnPooling(nn.Module):
	"""Self attention over a sequence:

	* o_i = softmax(function(Wx_i)) for x_i in X.
	"""
	
	def __init__(self, input_size, hidden_size=128):
		super(SeqAttnPooling, self).__init__()
		self.FFN = FeedForwardNetwork(input_size, hidden_size, 1)
	
	def forward(self, x, x_mask):
		"""
		Args:
			x: batch * len * dim
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			alpha: batch * len
		"""
		scores = self.FFN(x).squeeze(2)
		scores.data.masked_fill_(x_mask.data, -float('inf'))
		alpha = F.softmax(scores, dim=-1)
		self.alpha = alpha
		return alpha.unsqueeze(1).bmm(x).squeeze(1)


class SFU(nn.Module):
	"""Semantic Fusion Unit
	The ouput vector is expected to not only retrieve correlative information from fusion vectors,
	but also retain partly unchange as the input vector
	"""
	
	def __init__(self, input_size, fusion_size, dropout_rate=0.):
		super(SFU, self).__init__()
		self.linear_r = nn.Linear(input_size + fusion_size, input_size)
		self.linear_g = nn.Linear(input_size + fusion_size, input_size)
		self.dropout_rate = dropout_rate
	
	def forward(self, x, fusions):
		r_f = torch.cat([x, fusions], 2)
		if self.dropout_rate:
			r_f = F.dropout(r_f, p=self.dropout_rate, training=self.training)
		r = torch.tanh(self.linear_r(r_f))
		g = torch.sigmoid(self.linear_g(r_f))
		o = g * r + (1 - g) * x
		return o


class BRNN(nn.Module):
	""" Bi-directional RNNs.
	"""
	
	def __init__(self, input_size, hidden_size, num_layers,
				 dropout_rate=0, rnn_type=nn.LSTM):
		super(BRNN, self).__init__()
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		self.rnns = nn.ModuleList()
		if type(rnn_type) is str:
			rnn_type = nn.GRU if rnn_type == 'gru' else nn.LSTM
		self.num_units = []
		for i in range(num_layers):
			input_size = input_size if i == 0 else 2 * hidden_size
			self.num_units.append(input_size)
			self.rnns.append(rnn_type(input_size, hidden_size,
									  num_layers=1,
									  bidirectional=True))
	
	def forward(self, x):
		# Transpose batch and sequence dims
		x = x.transpose(0, 1)
		batch_size = x.size(1)
		
		# Encode all layers
		outputs = [x]
		for i in range(self.num_layers):
			rnn_input = outputs[-1]
			
			# Apply dropout to hidden input
			if self.dropout_rate > 0:
				rnn_input = F.dropout(rnn_input,
									  p=self.dropout_rate,
									  training=self.training)
			
			# Forward
			rnn_output = self.rnns[i](rnn_input)[0]
			outputs.append(rnn_output)
		
		output = outputs[-1]
		# Transpose back
		output = output.transpose(0, 1)
		return output


class SelfAttnMatch(nn.Module):
	"""Given sequences X and Y, match sequence Y to each element in X.

	* o_i = sum(alpha_j * x_j) for i in X
	* alpha_j = softmax(x_j * x_i)
	"""
	
	def __init__(self, input_size, identity=False, diag=True):
		super(SelfAttnMatch, self).__init__()
		if not identity:
			self.linear = nn.Linear(input_size, input_size)
		else:
			self.linear = None
		self.diag = diag
	
	def forward(self, x, x_mask):
		"""
		Args:
			x: batch * len1 * dim1
			x_mask: batch * len1 (1 for padding, 0 for true)
		Output:
			matched_seq: batch * len1 * dim1
		"""
		# Project vectors
		if self.linear:
			x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
			x_proj = F.relu(x_proj)
		else:
			x_proj = x
		
		# Compute scores
		scores = x_proj.bmm(x_proj.transpose(2, 1))
		if not self.diag:
			x_len = x.size(1)
			for i in range(x_len):
				scores[:, i, i] = 0
		
		# Mask padding
		x_mask = x_mask.unsqueeze(1).expand(scores.size())
		scores.data.masked_fill_(x_mask.data, -float('inf'))
		
		# Normalize with softmax
		alpha = F.softmax(scores, dim=2)
		# Take weighted average
		matched_seq = alpha.bmm(x)
		return matched_seq


class embedding(nn.Module):
	"""An Embedding layer."""
	
	def __init__(self, vocab_size, embedding_dim1=None, **kwargs):
		super(embedding, self).__init__(**kwargs)
		self.vocab_size = vocab_size
		self.embedding_dim1 = embedding_dim1 if embedding_dim1 else 64
		self.embedding = nn.Embedding(self.vocab_size,
									  self.embedding_dim1)
	
	def forward(self, elmo_emb, inp):
		# x1=input1
		x1 = elmo_emb
		x2 = self.embedding(inp)
		x = torch.cat((x1, x2), dim=-1)
		return x


class TextEncoder(nn.Module):
	"""
	Bidirectional Encoder
	can be used for Language Model and also for text classification or others
	input is batch of sentence ids [batch_size, num_steps]
	output is [batch_size, num_steps, 2 * hidden_dim]
	for text classification you can use pooling to get [batch_size, dim] as text resprestation
	for language model you can just add fc layer to convert 2 * hidden_dim to vocab_size -1 and calc cross entropy loss
	Notice you must outputs hidden_dim(forward) and hidden_dim(back_ward) concated at last dim as 2 * hidden dim, so MUST be bidirectional
	"""
	
	def __init__(self, vocab_size, embedding_dim, hidden_size=200, num_layers=2):
		super(TextEncoder, self).__init__()
		self.vocab_size = vocab_size
		
		self.embedding = embedding(vocab_size,
								   embedding_dim - 1024)  # embed1, embed2
		
		self.encode = BRNN(
			input_size=embedding_dim,
			hidden_size=hidden_size,
			num_layers=num_layers,
			rnn_type=nn.LSTM,
		)
		self.output_size = 2 * hidden_size
	
	def forward(self, elmo_emb, inp):
		x = self.embedding(elmo_emb, inp)
		x = self.encode(x)
		return x


class ModelBase(nn.Module):
	def __init__(self):
		super(ModelBase, self).__init__()
		
		self.num_units = 200
		self.dropout_rate = 1 - 0.7
		self.aug = False
		self.encode = TextEncoder(vocab_size=vocab_size, embedding_dim=1100, hidden_size=self.num_units, num_layers=2)
		
		doc_hidden_size = self.encode.output_size
		self.pooling = SeqAttnPooling(input_size=doc_hidden_size)
		# input dim not as convinient as tf..
		pre_logits_dim = doc_hidden_size
		
		self.num_classes = NUM_CLASSES
		# exclusive fc
		# self.logits = nn.ModuleList([copy.deepcopy(nn.Linear(pre_logits_dim, self.num_classes - 1)) for _ in range(NUM_ATTRIBUTES)])
		self.logits = nn.Linear(pre_logits_dim, 4)
	
	def unk_aug(self, x, x_mask=None):
		"""
		randomly make 10% words as unk
		"""
		if not self.aug:  # or epoch() < 2:
			return x
		
		if x_mask is None:
			x_mask = x > 0
		x_mask = x_mask.long()
		
		ratio = np.random.uniform(0, 0.02)
		mask = torch.cuda.FloatTensor(x.size(0), x.size(1)).uniform_() > ratio
		mask = mask.long()
		rmask = UNK_ID * (1 - mask)
		
		x = (x * mask + rmask) * x_mask
		return x


class haha_net(ModelBase):
	def __init__(self):
		super(haha_net, self).__init__()
		
		doc_hidden_size = self.encode.output_size
		
		self.self_aligner = SelfAttnMatch(doc_hidden_size, identity=True, diag=False)
		self.self_SFU = SFU(doc_hidden_size, 3 * doc_hidden_size)
		# aggregating
		self.aggregate_rnn = BRNN(
			input_size=doc_hidden_size,
			hidden_size=self.num_units,
			num_layers=1,
			dropout_rate=self.dropout_rate,
			rnn_type=nn.LSTM,
		)
	
	def forward(self, elmo_emb, inp, training=False):
		x = inp
		# print('inp',elmo_emb.size(),inp.size())
		# print(x.shape)
		x_mask = x.eq(0)
		batch_size = x.size(0)
		max_c_len = x.size(1)
		
		# x = self.unk_aug(x, x_mask)
		
		x_mask = torch.zeros_like(x, dtype=torch.uint8)
		x = self.encode(elmo_emb, inp)
		# print('encode',x.size())
		c_check = x
		c_bar = c_check
		#c_tilde = self.self_aligner.forward(c_bar, x_mask)
		# print('align',c_tilde.size())
		#c_hat = self.self_SFU.forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
		#c_check = self.aggregate_rnn.forward(c_hat)
		# print('rnn shape',c_check.size())
		x = c_check
		x = self.pooling(x, x_mask)
		# print('pooling',x.size())
		self.feature = x
		x = F.sigmoid(self.logits(x))
		# print('fc',x.size())
		x = x.view([-1, NUM_ATTRIBUTES])
		
		return x



stack = []

def create_data_producer(train_iter):
	producer = threading.Thread(target=addstack, args=(train_iter))
	producer.setDaemon(True)
	producer.start()


def addstack(train_iter):
	while True:
		if len(stack) >= 10:
			time.sleep(0.01)
			continue
		train_x1, train_x2, train_y = train_iter.__next__()
		# print("pre-process one batch data costs {0:.4f} s".format(time() - orig_time))
		stack.append((train_x1, train_x2, train_y))


def getbatch():
	while True:
		if len(stack) == 0:
			continue
		return stack.pop(0)


# evaluate function
def evaluate(model, test_x1, test_x2, test_y, loss_fn):
	x1 = torch.tensor(test_x1, dtype=torch.float).to(device)
	x2 = torch.tensor(test_x2, dtype=torch.long).to(device)
	y = test_y
	output = model(x1, x2)
	output = output.cpu()
	loss = loss_fn(output, torch.tensor(y, dtype=torch.float)).detach().numpy()
	output = output.detach().numpy()
	res = {'loss': loss}
	for ii in range(4):
		temp_y = y[:, ii];
		temp_pred = output[:, ii]
		res['f1_' + str(ii + 1)] = metrics.f1_score(temp_y, temp_pred > 0.5)
		res['auc_' + str(ii + 1)] = metrics.roc_auc_score(temp_y, temp_pred)
	return res


# train function
def train(model, test_iter, n_iters):
	eval_every = int(n_iters / 10)
	
	print('number of iteration per batch:{0}, evaluation every {1} iterations.'.format(n_iters, eval_every))
	
	# init
	loss_fn = torch.nn.BCELoss()
	optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)  # 0.001
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.7)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	
	st = time.time()
	best_auc = 0
	for epoch in range(1, EPOCHS + 1):
		for i in range(n_iters):
			train_x1, train_x2, train_y = getbatch()
			# print('data batch',train_x1.shape, train_x2.shape, train_y.shape)
			train_x1 = torch.tensor(train_x1, dtype=torch.float).to(device)
			train_x2 = torch.tensor(train_x2, dtype=torch.long).to(device)
			train_y = torch.tensor(train_y, dtype=torch.float).to(device)
			output = model(train_x1, train_x2)
			# print('output',output.size())
			loss = loss_fn(output, train_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print(epoch, i, loss.cpu().detach().numpy())
			if i % eval_every == 0:
				print("\n epoch: {0}, iter: {1} train_loss: {2}, time_cost: {3}".format(
					epoch, i, loss.cpu().detach().numpy(), time.time() - st))
				st = time.time()
				test_x1, test_x2, test_y = test_iter.__next__()
				res = evaluate(model, test_x1, test_x2, test_y, loss_fn)
				print("test loss: {0} f1_1: {1}, f1_2: {2}, f1_3: {3}, f1_4: {4}".format(
					res['loss'], res['f1_1'], res['f1_2'], res['f1_3'], res['f1_4']))
				avg_auc = np.mean([res['auc_1'], res['auc_2'], res['auc_3'], res['auc_4']])
				print("avg_auc: {0} auc_1: {1}, auc_2: {2}, auc_3: {3}, auc_4: {4}".format(
					avg_auc, res['auc_1'], res['auc_2'], res['auc_3'], res['auc_4']))
				if best_auc < avg_auc:
					best_auc = avg_auc
					torch.save(model, 'model/best_torch_model.pkl')
				st = time.time()
	return model

def main():
	all_data = pickle.load(open('../data/data_url_punc.pkl', 'rb'))
	print(all_data.head())
	all_set, seq_list, token_counter = get_data(all_data)
	
	vocab = text.vocab.Vocabulary(token_counter, unknown_token='<unk>', most_freq_count=10000,
								  reserved_tokens=['<pad>'])
	vocab_size = len(vocab)
	print(vocab_size)
	
	train_set, test_set = train_test_split(all_set, test_size=0.1)
	n_iters = int(len(train_set) / batch_size)
	train_iter = data_loader(train_set,vocab, batch_size=32)
	test_iter = data_loader(test_set,vocab, batch_size=64)

	create_data_producer(train_iter)
	print('producing data...')
	net = haha_net().to(device)
	#net = torch.load('torch_model.pkl').to(device)
	model = train(net, train_iter, n_iters)
	
	
if __name__ == '__main__':
	main()

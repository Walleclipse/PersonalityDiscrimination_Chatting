# coding: utf-8

# In[8]:


import collections
import io
import math
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
import pickle

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
ctx = mx.gpu()

# In[9]:


def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
	all_tokens.extend(seq_tokens)
	seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
	all_seqs.append(seq_tokens)


def build_data(all_seqs, vocab):
	indices = [vocab.to_indices(seq) for seq in all_seqs]
	return nd.array(indices)


def build_vocab(all_tokens):
	vocab = text.vocab.Vocabulary(collections.Counter(all_tokens),
								  reserved_tokens=[PAD, BOS, EOS])
	return vocab



# In[10]:

def get_data_set():
	corpus = pickle.load(open('sampled_chat_data.pkl', 'rb'))
	max_seq_len = 30
	# in和out分别是input和output的缩写
	in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
	MBTI = []
	lines = corpus
	for line in lines:
		in_seq, out_seq, MBTI_cur = line[0], line[1], line[2]
		in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
		if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
			continue  # 如果加上EOS后长于max_seq_len，则忽略掉此样本
		process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
		process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
		MBTI.append(MBTI_cur)
	MBTI = nd.array(MBTI)
	all_vocab = build_vocab(in_tokens + out_tokens)
	pickle.dump(all_vocab, open('all_vocab.pkl', 'wb'))
	in_data = build_data(in_seqs, all_vocab)
	out_data = build_data(out_seqs, all_vocab)
	dataset = gdata.ArrayDataset(in_data, out_data, MBTI)
	
	# In[32]:
	print('process success')
	return all_vocab, dataset

class Encoder(nn.Block):
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
				 drop_prob=0, **kwargs):
		super(Encoder, self).__init__(**kwargs)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
	
	def forward(self, inputs, state):
		# 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
		embedding = self.embedding(inputs).swapaxes(0, 1)
		return self.rnn(embedding, state)
	
	def begin_state(self, *args, **kwargs):
		return self.rnn.begin_state(*args, **kwargs)

def attention_model(attention_size):
	model = nn.Sequential()
	model.add(nn.Dense(attention_size, activation='tanh', use_bias=False,
					   flatten=False),
			  nn.Dense(1, use_bias=False, flatten=False))
	return model



def attention_forward(model, enc_states, dec_state):
	# 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
	dec_states = nd.broadcast_axis(
		dec_state.expand_dims(0), axis=0, size=enc_states.shape[0])
	enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
	e = model(enc_and_dec_states)  # 形状为(时间步数, 批量大小, 1)
	alpha = nd.softmax(e, axis=0)  # 在时间步维度做softmax运算
	return (alpha * enc_states).sum(axis=0)  # 返回背景变量



class Decoder(nn.Block):
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
				 attention_size, drop_prob=0, **kwargs):
		super(Decoder, self).__init__(**kwargs)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.embedding_1 = nn.Embedding(2, 8)
		self.embedding_2 = nn.Embedding(2, 8)
		self.embedding_3 = nn.Embedding(2, 8)
		self.embedding_4 = nn.Embedding(2, 8)
		self.attention = attention_model(attention_size)
		self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
		self.out = nn.Dense(vocab_size, flatten=False)
	
	def forward(self, cur_input, state, enc_states, character):
		# 使用注意力机制计算背景向量
		c = attention_forward(self.attention, enc_states, state[0][-1])
		# 将嵌入后的输入和背景向量在特征维连结
		input_and_c = nd.concat(self.embedding(cur_input), c, self.embedding_1(character[:, 0]),
								self.embedding_2(character[:, 1]), self.embedding_3(character[:, 2]),
								self.embedding_4(character[:, 3]), dim=1)
		# 为输入和背景向量的连结增加时间步维，时间步个数为1
		output, state = self.rnn(input_and_c.expand_dims(0), state)
		# 移除时间步维，输出形状为(批量大小, 输出词典大小)
		output = self.out(output).squeeze(axis=0)
		return output, state
	
	def begin_state(self, enc_state):
		# 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
		return enc_state


def batch_loss(encoder, decoder, X, Y, M, loss,all_vocab):
	batch_size = X.shape[0]
	enc_state = encoder.begin_state(batch_size=batch_size, ctx=ctx)
	enc_outputs, enc_state = encoder(X, enc_state)
	# 初始化解码器的隐藏状态
	dec_state = decoder.begin_state(enc_state)
	# 解码器在最初时间步的输入是BOS
	dec_input = nd.array([all_vocab.token_to_idx[BOS]] * batch_size, ctx=ctx)
	# 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失
	mask, num_not_pad_tokens = nd.ones(shape=(batch_size,), ctx=ctx), 0
	l = nd.array([0], ctx=ctx)
	for y in Y.T:
		dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs, M)
		l = l + (mask * loss(dec_output, y.as_in_context(ctx))).sum()
		dec_input = y  # 使用强制教学
		num_not_pad_tokens += mask.sum().asscalar()
		# 当遇到EOS时，序列后面的词将均为PAD，相应位置的掩码设成0
		mask = mask * (y != all_vocab.token_to_idx[EOS])
	return l / num_not_pad_tokens



def train(encoder, decoder, dataset, lr, batch_size, num_epochs,all_vocab):
	encoder.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
	decoder.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
	enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam',
								{'learning_rate': lr})
	dec_trainer = gluon.Trainer(decoder.collect_params(), 'adam',
								{'learning_rate': lr})
	loss = gloss.SoftmaxCrossEntropyLoss()
	data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
	min_loss = 9999
	for epoch in range(num_epochs):
		l_sum = 0.0
		iters = 0
		for X, Y, M in data_iter:
			if iters % 50 == 0:
				print(iters)
			iters += 1
			with autograd.record():
				l = batch_loss(encoder, decoder, X.as_in_context(ctx), Y.as_in_context(ctx), M.as_in_context(ctx), loss,all_vocab)
			l.backward()
			enc_trainer.step(1)
			dec_trainer.step(1)
			l_sum += l.asscalar()
		print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
		cur_loss = l_sum / len(data_iter)
		if cur_loss < min_loss:
			encoder.save_parameters('encoder.model')
			decoder.save_parameters('decoder.model')


# In[ ]:

def main():
	embed_size, num_hiddens, num_layers = 64, 64, 2
	attention_size, drop_prob, lr, batch_size, num_epochs = 32, 0.5, 0.01, 400, 50
	
	all_vocab, dataset = get_data_set()
	encoder = Encoder(len(all_vocab), embed_size, num_hiddens, num_layers,
					  drop_prob)
	decoder = Decoder(len(all_vocab), embed_size, num_hiddens, num_layers,
					  attention_size, drop_prob)
	train(encoder, decoder, dataset, lr, batch_size, num_epochs,all_vocab)


if __name__ == '__main__':
	main()

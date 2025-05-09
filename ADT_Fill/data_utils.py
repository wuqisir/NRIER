import numpy as np
import pandas as pd
import scipy.sparse as sp
from copy import deepcopy
import random
import torch.utils.data as data


class load_all(object):
	def __init__(self, dataset, data_path, batch_size):
		self.dataset = dataset
		self.batch_size = batch_size
		self.train_rating = data_path + 'trainset_100k.csv'
		# self.train_rating = data_path + 'TRAINSET_1M.csv'
		# self.train_rating = data_path + 'trainset(AA).csv'
		# self.train_rating = data_path + 'adressa.train.rating'
		# self.train_rating = data_path + 'trainset_YM.csv'
		# self.valid_rating = data_path + '{}.valid.rating'.format(dataset)
		self.test_negative = data_path + 'testset_100k.csv'  # test.negtive
		# self.test_negative = data_path + 'TESTSET_1M.csv'#test.negtive
		# self.test_negative = data_path + 'adressa.test.negative'
		# self.test_negative = data_path + 'testset(AA).csv'
		# self.test_negative = data_path + 'testset_YM.csv'
	def get_all_data(self):
		################# load training data #################
		train_data = pd.read_csv(
			self.train_rating,
			sep=',', header=None, names=['user', 'item', 'score'],
			usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

		if self.dataset == "adressa":
			self.user_num = 212231
			self.item_num = 6596
		else:
			self.user_num = train_data['user'].max()  # 说明原数据中已经经过id-1处理
			self.item_num = train_data['item'].max()

		train_data['user'] = train_data['user'] - 1
		train_data['item'] = train_data['item'] - 1
		train_data['score'].apply(lambda x: 1 if x >= 3 else 0)

		train_data_change = train_data.values.tolist()  # 转换为[user_id,item_id,score]的形式
		# train_data_group_by_user=train_data.groupby(['user'])
		# load ratings as a dok matrix, Set the data to a sparse matrix
		self.train_mat = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
		train_data_list = []
		train_data_noisy = []
		train_interaction_dict = {}
		for x in train_data_change:
			train_data_list.append([x[0], x[1]])
			train_data_noisy.append(x[2])
			train_interaction_dict.setdefault(x[0], {}).setdefault(x[1], x[2])
			self.train_mat[x[0],x[1]]=1

		# user_pos indicates whether there is interaction
		self.user_pos = {}

		for x in train_data_list:
			if x[0] in self.user_pos:
				self.user_pos[x[0]].append(x[1])
			else:
				self.user_pos[x[0]] = [x[1]]
		self.all_pos = {}
		for x in train_data_list:
			# if x[0] in self.all_pos:
			# 	self.all_pos[x[0]].append(x[1])
			# else:
			# 	self.all_pos[x[0]] = [x[1]]
			self.all_pos.setdefault(x[0],list()).append(x[1])
		################# load testing data #################
		test_mat = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)

		self.test_data_pos = {}
		with open(self.test_negative, 'r', encoding='utf-8') as fd:
			line = fd.readline().encode('utf-8').decode('utf-8-sig')
			while line != None and line != '':
				arr = line.split(',')
				if self.dataset == "adressa":
					u = eval(arr[0])[0]
					i = eval(arr[0])[1]
				else:
					u = int(arr[0]) - 1
					i = int(arr[1]) - 1
				# if u in self.test_data_pos:
				# 	self.test_data_pos[u].append(i)
				# else:
				# 	self.test_data_pos[u] = [i]
				self.test_data_pos.setdefault(u, list()).append(i)
				test_mat[u, i] = 1.0
				line = fd.readline().encode('utf-8').decode('utf-8-sig')

		return train_data_list, self.test_data_pos, self.user_pos, self.all_pos, self.user_num, self.item_num, self.train_mat, train_data_noisy


class NCFData(data.Dataset):
	def __init__(self, features,
				 num_item, train_mat=None, num_ng=0, is_training=0, noisy_or_not=None):  # noisy_or_not是列表不是布尔类型数据
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		if is_training == 0:
			self.noisy_or_not = noisy_or_not
		else:
			self.noisy_or_not = [0 for _ in range(len(features))]
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]

	def ng_sample(self):
		assert self.is_training != 2, 'no need to sampling when testing'
		# Take a negative sample for each interaction
		self.features_ng = []
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])
		# In the training set, there are interactions as pos and negative samples as ng
		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.noisy_or_not_fill = self.noisy_or_not + [1 for _ in range(len(self.features_ng))]
		self.features_fill = self.features_ps + self.features_ng
		assert len(self.noisy_or_not_fill) == len(self.features_fill)
		self.labels_fill = labels_ps + labels_ng

	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training != 2 \
			else self.features_ps
		labels = self.labels_fill if self.is_training != 2 \
			else self.labels
		noisy_or_not = self.noisy_or_not_fill if self.is_training != 2 \
			else self.noisy_or_not

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		noisy_label = noisy_or_not[idx]

		return user, item, label, noisy_label

def mean_adj_single(adj):  # Defining internal functions
	# D^-1 * A
	rowsum = np.array(adj.sum(1))
	for inx,i in enumerate(rowsum):
		if i[0]==0:
			random_user = random.randint(0, 942)
			#Randomly select one from the users to interact with the item
			adj[inx,random_user]=1

	rowsum = np.array(adj.sum(1))
	# rowsum = np.where(rowsum == 0, 1, rowsum)
	d_inv = np.power(rowsum, -1).flatten()
	d_inv[np.isinf(d_inv)] = 0.
	d_mat_inv = sp.diags(d_inv)

	norm_adj = d_mat_inv.dot(adj)
	# norm_adj = adj.dot(d_mat_inv)
	print('generate single-normalized adjacency matrix.')
	return norm_adj.tocoo()

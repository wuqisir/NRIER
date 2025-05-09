import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# from tensorboardX import SummaryWriter

import model
import evaluate
from data_utils import *
from loss import *
from parameter import args

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(2023) # cpu
torch.cuda.manual_seed(2023) #gpu
np.random.seed(2023) #numpy
random.seed(2023) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
    np.random.seed(2023 + worker_id)


# writer = SummaryWriter() # for visualization

# define drop rate schedule
def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate



class EarlyStopper:
	def __init__(self, num_trials):
		self.num_trials = num_trials
		self.trial_counter = 0
		self.best_metric = dict()

	def is_continuable(self, model, metric,args):
		if len(self.best_metric)==0:
			self.best_metric=metric
			return True
		elif metric['recall'][0] > self.best_metric['recall'][0]:
			self.best_metric = metric
			self.trial_counter = 0
			if not os.path.exists(model_path):
				os.mkdir(model_path)
			torch.save(model, '{}{}_{}-{}_test.pth'.format(model_path, args.model, args.drop_rate, args.num_gradual))
			return True
		elif self.trial_counter + 1 < self.num_trials:
			self.trial_counter += 1
			return True
		else:
			return False

########################### Test #####################################
def take_test(model, test_data_pos, user_pos):
	top_k = args.top_k
	model.eval()
	result=dict()
	precision, recall, F1, NDCG, MRP = evaluate.test_all_users(model, 512, item_num, test_data_pos, user_pos, top_k)

	# print("################### TEST ######################")
	# print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
	# print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))
	result['precision'] = precision
	result['recall'] = recall
	result['F1'] = F1
	result['NDCG'] = NDCG
	result['MRP'] = MRP

	return result

if __name__=="__main__":
	data_path = '../data/{}/'.format(args.dataset)
	model_path = './models/{}/'.format(args.dataset)
	print("arguments: %s " % (args))
	print("config model", args.model)
	print("config data path", data_path)
	print("config model path", model_path)

	############################## PREPARE DATASET ##########################
	data_utils=load_all(args.dataset, data_path,args.batch_size)
	early_stopper = EarlyStopper(20)
	train_data, test_data_pos, user_pos, all_pos,user_num, item_num, train_mat, train_data_noisy = data_utils.get_all_data()
	# train_data, valid_data, test_data_pos, user_pos, user_num, item_num, train_mat, train_data_noisy = data_utils.load_all(
	# 	args.dataset, data_path)
	#Calculate the degree of the interaction matrix
	user_mean_adj=mean_adj_single(train_mat.tolil())
	item_mean_adj=mean_adj_single(train_mat.T.tolil())

	# construct the train and test datasets
	train_dataset = NCFData(
		train_data, item_num, train_mat, args.num_ng, 0, train_data_noisy)
	# valid_dataset = data_utils.NCFData(
	# 	valid_data, item_num, train_mat, args.num_ng, 1)

	train_loader = data.DataLoader(train_dataset,
								   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
								   worker_init_fn=worker_init_fn)

	print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num,
																							len(train_data),
																							len(test_data_pos)))

	########################### CREATE MODEL #################################
	if args.model == 'NeuMF-pre':  # pre-training. Not used in our work.
		GMF_model_path = model_path + 'GMF_0.1-6000_test.pth'
		MLP_model_path = model_path + 'MLP_0.1-6000_test.pth'
		NeuMF_model_path = model_path + 'NeuMF-end_0.1-6000_test.pth'
		assert os.path.exists(GMF_model_path), 'lack of GMF model'
		assert os.path.exists(MLP_model_path), 'lack of MLP model'
		GMF_model = torch.load(GMF_model_path)
		MLP_model = torch.load(MLP_model_path)
	else:
		GMF_model = None
		MLP_model = None

	model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
					  args.dropout, args.model, GMF_model, MLP_model)

	model.cuda()
	BCE_loss = nn.BCELoss()

	if args.model == 'NeuMF-pre':
		optimizer = optim.SGD(model.parameters(), lr=args.lr,weight_decay=1e-5)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
	########################### TRAINING #####################################
	count, best_hr = 0, 0
	best_loss = 1e9

	for epoch in range(args.epochs):
		model.train() # Enable dropout (if have).
		all_loss = 0
		start_time = time.time()
		train_loader.dataset.ng_sample()

		for user, item, label, noisy_or_not in train_loader:
			user = user.cuda()
			item = item.cuda()
			label = label.float().cuda()

			model.zero_grad()
			prediction = model(user, item)
			#更改损失
			# loss=BCE_loss(prediction,label)
			loss = model.create_denoise_loss(user,item,
											 prediction,label,
											 drop_rate_schedule(count),
											 train_mat.todense(),
											 user_mean_adj.todense(),
											 item_mean_adj.todense(),
											 args.temp_rate[0])
			loss.backward()
			optimizer.step()
			all_loss += loss
			# if count % args.eval_freq == 0 and count != 0:c
			# 	print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
			# 	best_loss = eval(model, valid_loader, best_loss, count)
			# 	model.train()

			count += 1
			t2 = time.time()
		# print(t2-t1)
		if (epoch + 1) % 1 != 0:
			# if args.verbose > 0 and epoch % args.verbose == 0:
			perf_str = 'Epoch %d [%.1fs]: train==[%.5f]' % (
				epoch, time.time() - start_time, all_loss)
			print(perf_str)
			continue

		print("############################## Training End. ##############################")
		# test_model = torch.load('{}{}_{}-{}.pth'.format(model_path, args.model, args.drop_rate, args.num_gradual))
		# test_model.cuda()
		ret = take_test(model, test_data_pos, user_pos)

		perf_str = 'Epoch %d : train==[%.5f], recall=[%.5f, %.5f], ' \
				   'precision=[%.5f, %.5f],F1=[%.5f, %.5f] ndcg=[%.5f, %.5f]' % \
				   (epoch, all_loss, ret['recall'][0], ret['recall'][-1],
					ret['precision'][0], ret['precision'][-1], ret['F1'][0], ret['F1'][-1], ret['NDCG'][0], ret['NDCG'][-1])
		print(perf_str)

		if early_stopper.is_continuable(model, ret, args):
			continue
		else:
			# Output the best evaluation metric
			final_perf = "Best results\trecall=[%s], precision=[%s], ndcg=[%s]" % \
						 (['%.5f' % r for r in early_stopper.best_metric['recall']],
						  '\t'.join(['%.5f' % r for r in early_stopper.best_metric['precision']]),
						  '\t'.join(['%.5f' % r for r in early_stopper.best_metric['NDCG']]))
			break


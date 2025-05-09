import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from user_based import user_based

class NCF(nn.Module):
	def __init__(self, user_num, item_num, factor_num, num_layers,
					dropout, model, GMF_model, MLP_model):
		super(NCF, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
		"""
		self.user_num=user_num
		self.item_num=item_num
		self.dropout = dropout
		self.model = model
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model

		self.embed_user_GMF = nn.Embedding(user_num, factor_num)
		self.embed_item_GMF = nn.Embedding(item_num, factor_num)
		self.embed_user_MLP = nn.Embedding(
				user_num, factor_num)
		self.embed_item_MLP = nn.Embedding(
				item_num, factor_num)


		MLP_modules = []
		for i in range(len(num_layers)-1):
			# input_size = factor_num * (2 ** (num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(num_layers[i], num_layers[i+1]))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		if self.model in ['MLP', 'GMF']:
			predict_size = factor_num
		else:
			predict_size = factor_num * 2
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

		self.user_list=torch.tensor([i for i in range(self.user_num)]).cuda()
		self.item_list=torch.tensor([i for i in range(self.item_num)]).cuda()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		if not self.model == 'NeuMF-pre':
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)
			nn.init.kaiming_uniform_(self.predict_layer.weight,
									a=1, nonlinearity='sigmoid')

			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()
		else:
			# embedding layers
			self.embed_user_GMF.weight.data.copy_(
							self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(
							self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(
							self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(
							self.MLP_model.embed_item_MLP.weight)

			# mlp layers
			for (m1, m2) in zip(
				self.MLP_layers, self.MLP_model.MLP_layers):
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([
				self.GMF_model.predict_layer.weight,
				self.MLP_model.predict_layer.weight], dim=1)
			precit_bias = self.GMF_model.predict_layer.bias + \
						self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)

	# Generate denoising loss + contrastive learning loss
	def create_denoise_loss(self,user_id,item_id,prediction,label,drop_rate,train_mat,user_adj,item_adj,temp_rate):
		user_id=user_id.cpu()
		item_id=item_id.cpu()
		train_mat=train_mat.A
		loss = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')

		loss_mul = loss * label
		ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
		loss_sorted = loss[ind_sorted]
		# print(loss[ind_sorted[-1]].item())
		#drop_rate
		remember_rate = 1 - drop_rate
		num_remember = int(remember_rate * len(loss_sorted))
		ind_update = ind_sorted[:num_remember]
		ind_delete = ind_sorted[num_remember:]
		del_num =len(ind_delete)


		loss_update = F.binary_cross_entropy_with_logits(prediction[ind_update], label[ind_update])
		deleted_user_id=user_id[ind_delete]
		deleted_item_id=item_id[ind_delete]

		user_embedding = self.embed_user_MLP(self.user_list)
		item_embedding = self.embed_item_MLP(self.item_list)
		user_embedd = torch.matmul(torch.tensor(user_adj).cuda(), item_embedding)
		user_based_embedd = user_embedd.cpu().detach().numpy()
		item_embedd = torch.matmul(torch.tensor(item_adj).cuda(),user_embedding)
		item_based_embedd = item_embedd.cpu().detach().numpy()
		# similarity = cosine_similarity(user_based_embedd)
		# similarity = similarity - np.eye(self.n_user, self.n_user)
		# similarity=(similarity+1)/2
		div_loss_list = torch.zeros(len(ind_delete))
		if len(deleted_user_id)==0:
			print("No elimination operation is performed in the first round")
		else:
			#Calculate the similarity between the removed user and other users
			delect_dict = {}
			refind_dict = {}
			for i in range(len(deleted_user_id)):
				# if deleted_user_id[i] in delect_dict:
				# 	delect_dict[deleted_user_id[i]].append(deleted_item_id[i])
				# else:
				# 	delect_dict[deleted_user_id[i]] = [deleted_item_id[i]]
				delect_dict.setdefault(deleted_user_id[i],list()).append(deleted_item_id[i])
			for u in delect_dict.keys():
				delete_item_num=len(delect_dict[u])
				del_user_embedding=user_based_embedd[u]
				dot_product = np.dot(item_based_embedd, del_user_embedding)
				# Calculates the norm of a vector
				norm_vector1 = np.linalg.norm(del_user_embedding)
				norm_vectors2 = np.linalg.norm(item_based_embedd, axis=1)

				cosine_similarity = dot_product / (norm_vector1 * norm_vectors2)
				#The initial calculated cosine similarity has negative number
				cosine_similarity = (cosine_similarity+1)/2

				interacted_items = train_mat[u].nonzero()[0]

				cosine_similarity=np.nan_to_num(cosine_similarity, nan=0)
				cosine_similarity[interacted_items]=0

				ind_sort=np.argsort(cosine_similarity)
				recommend_item=ind_sort[::-1][:delete_item_num]
				if u in refind_dict.keys():
					refind_dict[u].append(recommend_item)
				else:
					refind_dict[u] = recommend_item

			for i, u in enumerate(delect_dict.keys()):
				# Find all the removed and re-found user_id and item_id and their corresponding embeddings
				sub_item_id = refind_dict[u]
				if len(sub_item_id) == 0:
					continue
				sub_i_embeddings =item_embedd[sub_item_id.copy()]
				# print(sub_i_embeddings.requires_grad)
				del_item_id = delect_dict[u]
				del_item_id=np.array(del_item_id)
				del_i_embeddings = item_embedd[del_item_id.copy()]

				# interact_item = train_mat[u]
				# interact_item[delect_dict[u]] = 0
				# denoise_item = interact_item.nonzero()[0]
				# # denoise_i_list.append(list(denoise_item))
				# denoise_item_embeddings = item_embedding[denoise_item]
				u1 = torch.tensor([u])
				del_u_embedding = user_embedd[u1]
				# Calculate the cosine similarity between positive and negative samples
				norm_denoise_embedd = del_u_embedding.norm(2, dim=1, keepdim=True).T
				pos_similarity = torch.mm(sub_i_embeddings, del_u_embedding.T) / torch.mm(
					sub_i_embeddings.norm(2, dim=1, keepdim=True), norm_denoise_embedd)
				neg_similarity = torch.mm(del_i_embeddings, del_u_embedding.T) / torch.mm(
					del_i_embeddings.norm(2, dim=1, keepdim=True), norm_denoise_embedd)
				# Calculate div loss
				pos_similarity = pos_similarity / temp_rate
				neg_similarity = neg_similarity / temp_rate
				exp_pos_similarity = torch.exp(pos_similarity)
				exp_neg_similarity = torch.exp(neg_similarity)
				pos_sum_similarity = torch.sum(exp_pos_similarity, dim=1)
				neg_sum_similarity = torch.sum(exp_neg_similarity, dim=1)
				u_div_loss = pos_sum_similarity / (pos_sum_similarity + neg_sum_similarity)
				# print(u_div_loss)
				u_div_loss = torch.log(u_div_loss)
				u_div_loss.view(-1)
				div_loss_list[i] = u_div_loss
		div_loss_without_nan = torch.where(torch.isnan(div_loss_list), torch.tensor(0.0), div_loss_list)
		div_loss = -1 * torch.sum(div_loss_without_nan)
		# print(div_loss.device)
		# div_loss =1* div_loss
		return loss_update+0.1*div_loss



	def forward(self, user, item):
		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		# prediction=torch.sigmoid(prediction.view(-1))
		return prediction.view(-1)

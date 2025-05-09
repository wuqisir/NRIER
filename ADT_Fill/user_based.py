from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def user_based(user_id,similarity,neighbor_num,train_mat):
    #先根据相似度，找到目标用户的neighbor_num个邻居
    user_similarity=similarity[user_id]
    #这里先把邻居的数量·固定为u=10
    neighbor_id=np.argsort(user_similarity)[::-1][:10]
    interacted_items=train_mat[user_id].nonzero()[0]
    neigh_interact=train_mat[neighbor_id]
    pre_rating=predict_rating(user_id, neighbor_id, similarity, train_mat)
    #从预测的所有项目评分中剔除用户已经交互过的项目
    pre_rating[interacted_items]-=99
    recommend_item_id=np.argsort(pre_rating)[::-1][:neighbor_num]
    # for i in recommend_item_id:
    #     while i in interacted_items:
    #         #删除已经交互过的项目，
    #         np.delete(recommend_item_id,np.where(recommend_item_id==i)[0])
    return np.array(recommend_item_id)

#计算预测评分时，由于使用的是隐式反馈，所以如果只填充一个的话，预测分数是一样的
def predict_rating(user_id,neighbors, user_similarity, user_item_matrix):
    total_similarity = np.sum(user_similarity[user_id,neighbors])
    if total_similarity==0:
        print(user_similarity[user_id,neighbors])
        total_similarity=1e+10
    # test1=user_similarity[user_id, neighbors]
    # test2=user_item_matrix[neighbors]
    predicted_ratings = np.dot(user_similarity[user_id, neighbors], user_item_matrix[neighbors]) / total_similarity
    return predicted_ratings


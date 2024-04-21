import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class SVD(object):
    def __init__(self, alpha, reg_p, reg_q, number_LatentFactors=10, number_epochs=10,
                 columns=["uid", "iid", "rating"]):
        self.alpha = alpha
        self.reg_p = reg_p
        self.reg_q = reg_q
        self.number_LatentFactors = number_LatentFactors
        self.number_epochs = number_epochs
        self.columns = columns

    def fit(self, dataset, valset):
        self.dataset = pd.DataFrame(dataset)
        self.valset = valset
        self.user_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.item_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.global_mean = self.dataset[self.columns[2]].mean()

        self.P, self.Q = self.sgd()

    def _init_matrix(self):
        P = dict(zip(
            self.user_ratings.index,
            np.random.rand(len(self.user_ratings), self.number_LatentFactors).astype(np.float32)))

        Q = dict(zip(
            self.item_ratings.index,
            np.random.rand(len(self.item_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def predict(self, uid, iid):
        # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
        if uid not in self.user_ratings.index or iid not in self.item_ratings.index:
            # print("用户<%d>或物品<%d>不存在，故返回全局平均值作为预测" % (uid, iid))
            return self.global_mean

        pu = self.P[uid]
        qi = self.Q[iid]

        return np.dot(pu, qi)

    def test(self, testset):
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating

    def accuracy(self, predict_results):
        def rmse_mae(predict_results):
            '''
            rmse和mae评估指标
            :param predict_results:
            :return: rmse, mae
            '''
            length = 1
            _rmse_sum = 0
            _mae_sum = 0
            for uid, iid, real_rating, pred_rating in predict_results:
                length += 1
                _rmse_sum += (pred_rating - real_rating) ** 2
                _mae_sum += abs(pred_rating - real_rating)
            return round(np.sqrt(_rmse_sum / length), 4), round(_mae_sum / length, 4)

        return rmse_mae(predict_results)

    def sgd(self):
        P, Q = self._init_matrix()
        rmse_list = []
        mae_list = []

        for i in range(self.number_epochs):
            print("epoch%d" % i)

            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                pu = P[uid]
                qi = Q[iid]
                err = np.float32(r_ui - np.dot(pu, qi))

                pu += self.alpha * (err * qi - self.reg_p * pu)
                qi += self.alpha * (err * pu - self.reg_q * qi)

                P[uid] = pu
                Q[iid] = qi
            self.P = P
            self.Q = Q

            pred_results = self.test(self.valset)
            rmse, mae = self.accuracy(pred_results)
            rmse_list.append(rmse)
            mae_list.append(mae)
            print("rmse: ", rmse, "mae: ", mae)

        x = range(1, self.number_epochs + 1)
        plt.plot(x, rmse_list)
        # plt.xticks(x)
        plt.show()
        return P, Q

if __name__ == '__main__':
    trainset = pd.read_csv('ml-100k/u1.base', sep='\t', names=["userId", "movieId", "rating"], usecols=range(3))
    valset = pd.read_csv('ml-100k/u1.test', sep='\t', names=["userId", "movieId", "rating"], usecols=range(3))

    algo = SVD(0.005, 0.01, 0.01, 2, 200, ["userId", "movieId", "rating"])
    algo.fit(trainset, valset)
    pred_results = algo.test(valset)

    rmse, mae = algo.accuracy(pred_results)
    print("rmse: ", rmse, "mae: ", mae)
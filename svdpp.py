import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class BiasSvd(object):
    def __init__(self, alpha, reg_p, reg_q, reg_bu, reg_bi, number_LatentFactors=10, number_epochs=10,
                 columns=["userId", "movieId", "rating"]):
        self.alpha = alpha  # 学习率
        self.reg_p = reg_p
        self.reg_q = reg_q
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs
        self.columns = columns

    def fit(self, dataset, valset):
        '''
        fit dataset
        :param dataset: uid, iid, rating
        :return:
        '''

        self.dataset = pd.DataFrame(dataset)
        self.valset = valset

        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.globalMean = self.dataset[self.columns[2]].mean()

        self.P, self.Q, self.bu, self.bi, self.Y = self.sgd()

    def _init_matrix(self):
        '''
        初始化P和Q矩阵，同时为设置0，1之间的随机值作为初始值
        :return:
        '''
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def predict(self, uid, iid):

        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.P[uid]
        q_i = self.Q[iid]
        Y = self.Y

        _sum_yj = np.zeros([1, self.number_LatentFactors])
        jids = self.users_ratings.loc[uid]['movieId'][0]
        Nu = len(jids)
        for jid in jids:
            _sum_yj += Y[jid]

        return self.globalMean + self.bu[uid] + self.bi[iid] + np.dot(p_u + np.sqrt(1 / Nu) * _sum_yj, q_i)

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
            return np.sqrt(_rmse_sum / length), _mae_sum / length

        return rmse_mae(predict_results)

    def sgd(self):
        '''
        使用随机梯度下降，优化结果
        :return:
        '''
        P, Q = self._init_matrix()

        # 初始化bu、bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        Y = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))

        rmse_list = []
        mae_list = []

        for i in range(self.number_epochs):
            print("iter%d" % i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):

                jids = self.users_ratings.loc[uid]['movieId'][0]
                Nu = len(jids)
                _sum_yj = np.zeros([self.number_LatentFactors])

                for jid in jids:
                    _sum_yj += Y[jid]

                # sum_v_yj =

                v_pu = P[uid]
                v_qi = Q[iid]
                err = np.float32(
                    r_ui - self.globalMean - bu[uid] - bi[iid] - np.dot(v_pu + np.sqrt(1 / Nu) * _sum_yj, v_qi))
                for jid in jids:
                    Y[jid] += self.alpha * (err * np.sqrt(1 / Nu) * v_qi - 0.01 * Y[jid])

                v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (err * (v_pu + np.sqrt(1 / Nu) * _sum_yj) - self.reg_q * v_qi)

                P[uid] = v_pu
                Q[iid] = v_qi

                bu[uid] += self.alpha * (err - self.reg_bu * bu[uid])
                bi[iid] += self.alpha * (err - self.reg_bi * bi[iid])

                error_list.append(err ** 2)
            print(np.sqrt(np.mean(error_list)))
            self.P = P
            self.Q = Q
            self.bu = bu
            self.bi = bi
            self.Y = Y

            pred_results = self.test(self.valset)
            rmse, mae = self.accuracy(pred_results)
            rmse_list.append(rmse)
            mae_list.append(mae)
            print("rmse: ", rmse, "mae: ", mae)

        x = range(1, self.number_epochs + 1)
        plt.plot(x, rmse_list)
        plt.xticks(x)
        plt.show()
        return P, Q, bu, bi, Y

if __name__ == '__main__':
    trainset = pd.read_csv('ml-100k/u1.base', sep='\t', names=["userId", "movieId", "rating"], usecols=range(3))
    valset = pd.read_csv('ml-100k/u1.test', sep='\t', names=["userId", "movieId", "rating"], usecols=range(3))

    algo = BiasSvd(0.01, 0.01, 0.01, 0.01, 0.01, 2, 20)
    algo.fit(trainset, valset)

    # print(trainset[trainset["userId"] == 1])
    # print( len(bsvd.users_ratings.loc[1]['movieId'][0]) )
    pred_results = algo.test(valset)

    rmse, mae = algo.accuracy(pred_results)
    #
    print("rmse: ", rmse, "mae: ", mae)
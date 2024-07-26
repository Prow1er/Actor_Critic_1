import numpy as np
from scipy.linalg import eig


def improved_ahp(judgment_matrix):
    # 特征向量法
    eigenvalues, eigenvectors = eig(judgment_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues.real)
    feature_vector_weights = eigenvectors[:, max_eigenvalue_index].real
    feature_vector_weights /= np.sum(feature_vector_weights)

    # 归一化判断矩阵
    normalized_matrix = judgment_matrix / np.sum(judgment_matrix, axis=0)

    # 算术平均法
    arithmetic_mean_weights = np.mean(normalized_matrix, axis=1)

    # 几何平均法
    geometric_mean_weights = np.power(np.prod(judgment_matrix, axis=1), 1 / judgment_matrix.shape[1])

    # 平均权重
    average_weights = (arithmetic_mean_weights + geometric_mean_weights + feature_vector_weights) / 3

    return average_weights


def critic_method(data):
    # 计算数据在每个特征上的平均值
    mean_data = np.mean(data, axis=0)

    # 计算数据在每个特征上的方差
    variance = np.var(data, axis=0)

    # 计算每个特征的对比度，即方差与平均值的比值
    contrast = variance / mean_data

    # 计算数据的相关矩阵
    correlation_matrix = np.corrcoef(data.T)

    # 将相关矩阵中值为1的元素（完全相关的元素）设置为0
    # 这是因为完全相关的特征在critic方法中不提供额外信息
    correlation_matrix[correlation_matrix == 1] = 0

    # 计算每个特征的冲突度，即所有相关系数的绝对值之和
    conflict = np.sum(np.abs(correlation_matrix), axis=0)

    # 计算每个特征的critic权重，即对比度与冲突度的乘积
    critic_weights = contrast * conflict

    # 归一化权重，确保它们的和为1
    critic_weights /= np.sum(critic_weights)

    return critic_weights


def comparison_matrix(scores):
    n = len(scores)
    # 构建比较矩阵
    comparison_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if scores[i] > scores[j]:
                comparison_matrix[i, j] = scores[i] / scores[j]
            elif scores[i] < scores[j]:
                comparison_matrix[i, j] = scores[j] / scores[i]
            else:
                comparison_matrix[i, j] = 1
    # print(comparison_matrix)
    return comparison_matrix


class Judge:
    def __init__(self, interval, scores_criterion_1, scores_criterion_2, scores_criterion_3):
        """
        初始化函数。
        :param interval: 异常区间大小
        :param scores_criterion_1: 准则一的评分
        :param scores_criterion_2: 准则二的评分
        :param scores_criterion_3: 准则三的评分
        """
        self.scores_criterion_3 = scores_criterion_3
        self.scores_criterion_2 = scores_criterion_2
        self.scores_criterion_1 = scores_criterion_1
        self.interval = interval

    def final_weights(self):
        weights_criterion_1 = improved_ahp(comparison_matrix(self.scores_criterion_1))
        weights_criterion_2 = improved_ahp(comparison_matrix(self.scores_criterion_2))
        weights_criterion_3 = improved_ahp(comparison_matrix(self.scores_criterion_3))

        # 将每个准则的权重相加
        ahp_weights = weights_criterion_1 + weights_criterion_2 + weights_criterion_3
        # print(np.array(self.scores_criterion_1).reshape(-1, 1))
        # critic
        critic_matrix = np.vstack((np.array(self.scores_criterion_1),
                                   np.array(self.scores_criterion_2),
                                   np.array(self.scores_criterion_3)))
        # print(critic_matrix.T)
        criric_weights = critic_method(critic_matrix.T)

        total_weights = ahp_weights * 0.8 + criric_weights * 0.2
        # 归一化总权重
        final_weights = total_weights / np.sum(total_weights)
        # 计算得分
        score_strategy = self.interval * final_weights

        return score_strategy


if __name__ == '__main__':
    judge = Judge(5, [2, 7, 6], [4, 2, 2], [4, 1, 2])
    score_strategy = judge.final_weights()
    print("score_strategy:", score_strategy)
    # score_strategy: [1.96707869 1.59175119 1.44117012]

"""
a a11 a12..
b a21
c a31
d

给出的策略来验证

 策略打分表 -> 策略计算
"""

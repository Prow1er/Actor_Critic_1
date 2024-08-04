import pymysql
import warnings
import numpy as np
import pandas as pd
from judge import Judge

idx = '利税比'
itvl = ' -20~-10'


def fetch_data_from_database(interval):
    # 数据库连接配置
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '/GAMEmode1',
        'database': 'test_data',
        'charset': 'utf8mb4'
    }

    interval_dict = {
        ' -20~-10': 10,
        ' -10~-5': 5,
        ' -5~0': 5,
        '0~5': 5,
        '5~10': 5,
        '10~20': 10
    }

    # 连接数据库
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            warnings.filterwarnings("ignore")
            sql_query = """SELECT * FROM strategies """

            # 执行查询
            cursor.execute(sql_query)
            # 获取查询结果
            df = pd.DataFrame(cursor.fetchall())
            df = df.fillna(method='ffill')
            # print(df)
            num_of_departments = int((df.shape[1] - 3) / 3)
            df_interval = df[df[1] == interval]
            # print(df_interval)
            criterion1, criterion2, criterion3 = [], [], []

            for i in range(0, num_of_departments):
                df_c1 = df_interval[df_interval[4 + i * 3] == '准则一']
                # print(df_c1)
                df_cr1 = list(df_c1[5 + 3 * i])
                df_int_cr1 = list(map(int, df_cr1))
                criterion1.append(df_int_cr1)
            # print(criterion1)

            for i in range(0, num_of_departments):
                df_c2 = df_interval[df_interval[4 + i * 3] == '准则二']
                # print(df_c2)
                df_cr2 = list(df_c2[5 + 3 * i])
                df_int_cr2 = list(map(int, df_cr2))
                criterion2.append(df_int_cr2)
            # print(criterion2)

            for i in range(0, num_of_departments):
                df_c3 = df_interval[df_interval[4 + i * 3] == '准则三']
                # print(df_c3)
                df_cr3 = list(df_c3[5 + 3 * i])
                df_int_cr3 = list(map(int, df_cr3))
                # print(df_int_cr3)
                criterion3.append(df_int_cr3)
            # print(criterion3)

            group_weights = []
            for j in range(0, num_of_departments):
                judge = Judge(interval_dict[interval], criterion1[j], criterion2[j], criterion3[j])
                department = judge.final_weights()
                group_weights.append(list(department))

            group_weights = [[x if not np.isnan(x) else 0 for x in row] for row in group_weights]
            # print(group_weights)
            # 计算每个位置上的元素之和
            sums = [sum(column) for column in zip(*group_weights)]
            non_zero_counts = [sum(1 for x in col if x != 0) for col in zip(*group_weights)]
            # 计算平均值
            average_group_weights = [s / count if count > 0 else 0 for s, count in zip(sums, non_zero_counts)]
            # print(average_group_weights)
            return average_group_weights

    finally:

        connection.close()


# 调用函数
if __name__ == '__main__':
    avg = fetch_data_from_database(itvl)
    # print(avg)

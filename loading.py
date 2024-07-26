import pandas as pd
from sqlalchemy import create_engine

# MySQL数据库配置
db_user = '@localhost'
db_password = '/GAMEmode1'
db_host = 'localhost'
db_name = 'your_database'

# 创建数据库引擎
engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')

# 读取Excel文件
excel_file_path = 'example.xlsx'
df = pd.read_excel(excel_file_path, sheet_name='Sheet1')


# 数据预处理
def process_data(df):
    # 处理缺失值
    df.fillna('', inplace=True)

    # 提取唯一异常区间
    unique_intervals = df['异常区间'].unique()

    # 创建空字典以存储每个异常区间的数据
    interval_data = {}

    for interval in unique_intervals:
        # 选择特定异常区间的数据
        interval_df = df[df['异常区间'] == interval]

        # 提取策略组
        strategy_groups = interval_df['策略组'].unique()

        # 存储每个策略组的数据
        interval_data[interval] = {}
        for group in strategy_groups:
            group_df = interval_df[interval_df['策略组'] == group]

            # 存储每个部门的数据
            departments = ['部门A', '部门B', '部门C']
            for department in departments:
                department_df = group_df[group_df[department] != '']

                # 存储策略和约束准则
                strategies = department_df[department].tolist()
                criteria = department_df['约束准则'].tolist()
                scores = department_df['评分(0-9)'].tolist()

                # 将数据组织为字典
                interval_data[interval][group] = {
                    '部门': department,
                    '策略': strategies,
                    '约束准则': criteria,
                    '评分': scores
                }

    return interval_data


# 处理数据并存储到字典
interval_data = process_data(df)


# 插入数据到MySQL
def insert_data_to_mysql(interval_data):
    for interval, groups in interval_data.items():
        for group, data in groups.items():
            department = data['部门']
            strategies = data['策略']
            criteria = data['约束准则']
            scores = data['评分']

            for i in range(len(strategies)):
                strategy = strategies[i]
                criterion = criteria[i]
                score = scores[i]

                # 插入数据
                query = f"""
                INSERT INTO your_table (异常区间, 策略组, 部门, 策略, 约束准则, 评分)
                VALUES ('{interval}', '{group}', '{department}', '{strategy}', '{criterion}', {score});
                """

                with engine.connect() as connection:
                    connection.execute(query)


# 执行插入数据函数
insert_data_to_mysql(interval_data)
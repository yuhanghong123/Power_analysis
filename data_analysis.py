from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time  # 导入时间模块
import os  # 导入os模块
import math
import numpy as np


def calculate_socal_power_potential(pred, beta, Rb, rho_g, eff):
    # 将角度转换为弧度
    beta_rad = math.radians(beta)

    # 计算每个部分
    part1 = pred[0] * Rb
    part2 = pred[1] * (1 + np.cos(beta_rad)) / 2
    part3 = pred[0] * rho_g * (1 - np.cos(beta_rad)) / 2

    # 计算总的倾斜辐射量
    Solar_power_potential = (part1 + part2 + part3) * eff

    return Solar_power_potential


def load_data_and_predict(csv_files):

    for csv_file in csv_files:
        start_time = time.time()  # 记录开始时间
        # 提取CSV文件名（不含扩展名）
        folder_name = os.path.splitext(os.path.basename(csv_file))[0]

        # 创建父文件夹 exp（如果不存在）
        parent_folder = "exp"
        os.makedirs(parent_folder, exist_ok=True)

        # 在 exp 文件夹中创建以CSV文件名为名的次级文件夹
        full_folder_path = os.path.join(parent_folder, folder_name)

        os.makedirs(full_folder_path, exist_ok=True)
        # # 提取CSV文件名作为文件夹名称
        # folder_name = os.path.splitext(os.path.basename(csv_file))[0]
        # os.makedirs(folder_name, exist_ok=True)  # 创建文件夹或确认文件夹已存在

        # 使用Pandas读取CSV文件
        df = pd.read_csv(csv_file, skiprows=18)

        # 合并年、月、日列，并创建日期时间字段
        df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(
            columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))

        # 设置日期时间字段为索引
        df.set_index('DATE', inplace=True)

        # 删除不再需要的年、月、日列
        df.drop(columns=['YEAR', 'MO', 'DY'], inplace=True)

        sns.pairplot(df)
        plt.suptitle('Scatter Plot Matrix of Different Variables', y=1.02)

        plt.savefig(os.path.join(full_folder_path, 'pairplot.png'))
        plt.close()
        # 进行相关性分析
        columns_to_correlation = df.columns.to_list()

        # 计算相关性矩阵
        correlation_matrix = df[columns_to_correlation].corr(method='pearson')

        # 可视化相关性矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True,
                    cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')

        plt.savefig(os.path.join(full_folder_path, 'correlation_matrix.png'))
        plt.close()
        # 进行过滤操作
        filtered_df = df.loc[df.index < '2023-01-01']
        pred_df = df.loc[df.index >= '2023-01-01']

        # 划分特征和标签
        X = filtered_df.drop(
            columns=['ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DIFF'])
        y = filtered_df[['ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DIFF']]

        # 定义时间序列交叉验证器
        tscv = TimeSeriesSplit(n_splits=5)

        # 初始化多输出随机森林回归模型
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42))

        # 初始化评估指标
        fold = 1
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 数据归一化
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练模型
            model.fit(X_train_scaled, y_train)

            # 预测
            y_pred = model.predict(X_test_scaled)

            # 评估模型
            mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
            mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

            print(f"Fold {fold}")
            print(f"  Label1 - MSE: {mse[0]:.4f}, MAE: {mae[0]:.4f}")
            print(f"  Label2 - MSE: {mse[1]:.4f}, MAE: {mae[1]:.4f}")

            fold += 1

        # 模型预测从2023-01-01到2024-04-01的ALLSKY_SFC_SW_DWN和ALLSKY_SFC_SW_DIFF
        pred_df_scaled = scaler.transform(pred_df.drop(
            columns=['ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DIFF']))
        potential_power = []
        predictions = model.predict(pred_df_scaled)
        for pred in predictions:
            socal_pential = calculate_socal_power_potential(
                pred, beta=30, Rb=0.8, rho_g=0.2, eff=0.18)
            potential_power.append(socal_pential)
            # print(f"Predicted solar potential: {socal_pential:.2f} kWh/m^2")
        pred_df.loc[:, 'potential_solar_power'] = potential_power
        # Assign each output column individually using .loc[row_indexer, col_indexer] = value
        pred_df.loc[:, 'pred_ALLSKY_SFC_SW_DWN'] = predictions[:, 0]
        pred_df.loc[:, 'pred_ALLSKY_SFC_SW_DIFF'] = predictions[:, 1]

        # 将预测结果保存为CSV文件
        pred_df.to_csv(os.path.join(
            full_folder_path, f'pred_{folder_name}.csv'))

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间
        print(f"  Time elapsed: {elapsed_time:.2f} seconds for {csv_file}")


# 指定多个CSV文件路径的列表
csv_files = [
    'GX_dataset_2/baise.csv',
    'GX_dataset_2/beihai.csv',
    'GX_dataset_2/chongzuo.csv',
    'GX_dataset_2/fangchenggang.csv',
    'GX_dataset_2/guigang.csv',
    'GX_dataset_2/guilin.csv',
    'GX_dataset_2/hechi.csv',
    'GX_dataset_2/hezhou.csv',
    'GX_dataset_2/laibin.csv',
    'GX_dataset_2/liuzhou.csv',
    'GX_dataset_2/nanning.csv',
    'GX_dataset_2/qinzhou.csv',
    'GX_dataset_2/wuzhou.csv',
    'GX_dataset_2/yulin.csv'
]

# 调用函数进行数据加载和预测
load_data_and_predict(csv_files)

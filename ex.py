import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel
from data_process import get_dataset
import swanlab
# 定义预测函数
def predict_next_day(model, last_sequence, scaler):
    model.eval()
    with torch.no_grad():
        last_sequence = last_sequence.to(device).float().unsqueeze(0)  # 增加 batch 维度
        next_day_prediction = model(last_sequence).cpu().numpy()

        # 创建一个适当形状的数组来进行逆变换
        dummies = np.zeros((1, last_sequence.shape[2] + 1))
        dummies[0, 0] = next_day_prediction

        # 逆变换以获得原始尺度的预测值
        next_day_prediction = scaler.inverse_transform(dummies)[:, 0]
        return next_day_prediction[0]

import torch
from model import LSTMModel

# 加载模型
model = LSTMModel()
device = torch.device('cpu')
model.to(device)

# 假设模型的最佳权重已经保存在 'best_model.pth' 文件中
model.load_state_dict(torch.load("./checkpoint/20240608_194115/best_model.pth"))
model.eval()  # 切换到评估模式

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_last_sequence(file_path, lookback, scaler):
    data = pd.read_csv(file_path)
    data = data[['date', 'close']]
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # 归一化
    data['close'] = scaler.transform(data[['close']])

    # 获取最后一个时间步长的输入序列
    last_sequence = data['close'].values[:lookback]
    last_sequence = last_sequence.reshape((1, lookback, 1))
    last_sequence = torch.tensor(last_sequence).float()

    return last_sequence


# 假设文件路径为 'E:/研一下/上证指数项目/1year.csv'
file_path = 'E:/研一下/上证指数项目/1year.csv'
lookback = 25 # 使用过去 60天的数据进行预测
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform(pd.read_csv(file_path)[['close']])




def predict_next_day(model, last_sequence, scaler):
    model.eval()
    with torch.no_grad():
        last_sequence = last_sequence.to(device).float()  # 确保是 3D 张量 (batch_size, sequence_length, input_size)
        next_day_prediction = model(last_sequence).cpu().numpy()

        # 创建一个适当形状的数组来进行逆变换
        dummies = np.zeros((1, 1))  # 修改为合适的形状
        dummies[0, 0] = next_day_prediction.item()  # 提取单个元素

        # 逆变换以获得原始尺度的预测值
        next_day_prediction = scaler.inverse_transform(dummies)
        return next_day_prediction[0, 0]
for lookback in range(1,101):
    last_sequence = get_last_sequence(file_path, lookback, scaler)
    next_day_prediction = predict_next_day(model, last_sequence, scaler)
    print(f"{lookback},Predicted next day's close price: {next_day_prediction}")
# 使用模型预测未来一天的数值


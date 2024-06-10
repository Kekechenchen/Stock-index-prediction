import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import swanlab
from copy import deepcopy as dc
import numpy as np
import os
from model import LSTMModel
from data_process import get_dataset
import datetime

# 定义保存最佳模型的函数
def save_best_model(model, config, epoch):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'./checkpoint/{timestamp}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Val Epoch: {epoch} - Best model saved at {model_path}')
    return model_path

# 定义训练函数
def train(model, train_loader, optimizer, criterion, scheduler):
        running_loss = 0
        for i, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            y_pred = model(x_batch)

            loss = criterion(y_pred, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss_epoch = running_loss / len(train_loader)
        print(f'Epoch: {epoch}, Batch: {i}, Avg. Loss: {avg_loss_epoch}')
        swanlab.log({"train/loss": running_loss}, step=epoch)
        running_loss = 0

# 定义验证函数
def validate(model, config, test_loader, criterion, epoch, best_loss=None):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        print(f'Epoch: {epoch}, Validation Loss: {avg_val_loss}')
        swanlab.log({"val/loss": avg_val_loss}, step=epoch)

    if epoch == 1:
        best_loss = avg_val_loss

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model_path = save_best_model(model, config, epoch)
        config['best_model_path'] = best_model_path

    return best_loss

# 定义可视化函数
def visualize_predictions(train_predictions, val_predictions, scaler, X_train, X_test, y_train, y_test, lookback):
    train_predictions = train_predictions.flatten()
    val_predictions = val_predictions.flatten()

    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:,0] = train_predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:,0])

    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:,0] = val_predictions
    dummies = scaler.inverse_transform(dummies)
    val_predictions = dc(dummies[:,0])

    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:,0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:,0])

    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:,0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:,0])

    plt.figure(figsize=(10, 6))
    plt.plot(new_y_train, color='red', label='Actual Train Close Price')
    plt.plot(train_predictions, color='blue', label='Predicted Train Close Price', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('(TrainSet) Stock Index Prediction with LSTM')
    plt.legend()

    plt_image = []
    plt_image.append(swanlab.Image(plt, caption="TrainSet Price Prediction"))

    plt.figure(figsize=(10, 6))
    plt.plot(new_y_test, color='red', label='Actual Test Close Price')
    plt.plot(val_predictions, color='blue', label='Predicted Test Close Price', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('(TestSet) Stock Index Prediction with LSTM')
    plt.legend()

    plt_image.append(swanlab.Image(plt, caption="TestSet Price Prediction"))

    swanlab.log({"Prediction": plt_image})

# 定义预测未来一天数值的函数
def predict_next_day(model, last_sequence, scaler):
    model.eval()
    with torch.no_grad():
        last_sequence = last_sequence.to(device).float().unsqueeze(0)  # 增加 batch 维度
        next_day_prediction = model(last_sequence).cpu().numpy()
        next_day_prediction = scaler.inverse_transform(next_day_prediction)
        return next_day_prediction[0, 0]

if __name__ == '__main__':

    swanlab.init(
        project='Google-Stock-Prediction',
        experiment_name="LSTM",
        description="基于LSTM模型对Google股票价格数据集的训练与推理",
        config={
            "learning_rate": 4e-3,
            "epochs": 50,
            "batch_size": 32,
            "lookback": 60,
            "trainset_ratio": 0.95,
            "save_path": './checkpoint/',
            "optimizer": "AdamW",
        },
    )

    config = swanlab.config
    device = torch.device('cpu')

    scaler, X_train, X_test, y_train, y_test = get_dataset('E:/研一下/上证指数项目/1year.csv', config['lookback'])
    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = LSTMModel(input_size=1, output_size=1)
    print(model)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    def lr_lambda(epoch):
        total_epochs = config.epochs
        start_lr = config.learning_rate
        end_lr = start_lr * 0.01
        update_lr = ((total_epochs - epoch) / total_epochs) * (start_lr - end_lr) + end_lr
        return update_lr * (1 / config.learning_rate)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = None
    for epoch in range(1, config.epochs + 1):
        model.train()

        swanlab.log({"train/lr": scheduler.get_last_lr()[0]}, step=epoch)

        train(model, train_loader, optimizer, criterion, scheduler)

        best_loss = validate(model, config, test_loader, criterion, epoch, best_loss)

    with torch.no_grad():
        best_model_path = config.get('best_model_path')
        if best_model_path:
            model.load_state_dict(torch.load(best_model_path))
            model.eval()
            train_predictions = model(X_train.to(device)).to('cpu').numpy()
            val_predictions = model(X_test.to(device)).to('cpu').numpy()
            visualize_predictions(train_predictions, val_predictions, scaler, X_train, X_test, y_train, y_test, config.lookback)

            # 使用模型预测未来一天的数值
            last_sequence = X_test[-1]  # 获取测试集中最后一个时间步长的输入序列
            next_day_prediction = predict_next_day(model, last_sequence, scaler)
            print(f"Predicted next day's close price: {next_day_prediction}")
        else:
            print("最佳模型路径未找到")

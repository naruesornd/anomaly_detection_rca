import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import time_series_loader,test_time_series_loader
def plot_prediction(
    model, 
    X, 
    y, 
    device, 
    seq_len=12,  # 序列长度
    batch_size=64,  # 批大小
    n_points=500,  # 要绘制的点数
    start_idx=0,  # 起始索引
    save_path=None  # 图片保存路径
):
    """
    完整的预测和可视化函数，包含NaN处理、数据加载和评估
    
    参数:
        model: 训练好的PyTorch模型
        X: 输入特征 (DataFrame, numpy array 或 Tensor)
        y: 真实标签 (Series, numpy array 或 Tensor)
        device: 计算设备
        time_series_loader: 时间序列数据加载器函数
        seq_len: 序列长度
        batch_size: 批大小
        n_points: 要绘制的点数
        start_idx: 起始索引
        save_path: 图片保存路径
    """
    # 检查并处理 NaN 值
    if isinstance(X, (pd.DataFrame, pd.Series)):
        if X.isnull().sum().sum() > 0:
            print(f"警告: 输入数据包含 {X.isnull().sum().sum()} 个 NaN 值，已自动删除")
            X = X.dropna()
            y = y[X.index]  # 保持 X 和 y 的索引一致
    elif isinstance(X, np.ndarray):
        if np.isnan(X).any():
            print("警告: 输入数据包含 NaN 值，请先处理")
            raise ValueError("输入数据包含 NaN 值")

    # 转换为Tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X.values if hasattr(X, 'values') else np.array(X), 
                        dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y.values if hasattr(y, 'values') else np.array(y),
                        dtype=torch.float32)
    
    # 检查数据量是否足够
    total_samples = X.shape[0]
    if start_idx + n_points > total_samples:
        print(f"警告: 数据量不足，只能取 {total_samples - start_idx} 个点")
        n_points = total_samples - start_idx
    
    # 获取数据子集
    X_subset = X[start_idx : start_idx + n_points]
    y_subset = y[start_idx : start_idx + n_points]
    
    # 创建数据加载器
    loader = time_series_loader(X_subset, y_subset, seq_len=seq_len, 
                              batch_size=batch_size, shuffle=False)
    
    # 模型预测
    model.eval()
    y_preds = []
    y_trues = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            y_preds.append(preds)
            y_trues.append(y_batch.numpy())
    
    # 合并结果
    y_pred = np.concatenate(y_preds, axis=0).flatten()
    y_true = np.concatenate(y_trues, axis=0).flatten()
    
    # 检查长度是否一致
    assert len(y_pred) == len(y_true), f"预测值({len(y_pred)})和真实值({len(y_true)})长度不一致"
    
    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"评估指标 (基于索引 {start_idx} 到 {start_idx + n_points} 的 {len(y_true)} 个样本):")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 创建对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label='预测值 vs 真实值')
    
    # 绘制理想对角线
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')
    
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    plt.title(f'Model Prediction Comparison (Index {start_idx}-{start_idx + n_points})')
    plt.legend()
    plt.grid(True)
    
    # 添加评估指标
    textstr = '\n'.join((
        f'MAE = {mae:.4f}',
        f'MSE = {mse:.4f}',
        f'RMSE = {rmse:.4f}',
        f'R² = {r2:.4f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                  fontsize=10, verticalalignment='top', bbox=props)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()
    
    return y_pred
from .simulate_dataflow import IndustrialDataFlowSimulator
import pandas as pd
import numpy as np
import sys
import os
from data_processor import DataProcessor
from data_processor import CycleProcessor
from feature_engineering import FeatureEngineering
import torch
from model.load_model.load_model import load_model
from data_loader.industrialstreamloader import IndustrialStreamLoader
import joblib
def simulate_plant(target_col):
    sys.path.append(os.path.abspath('../src'))

    file_path = "../data/raw/data_factory_2.csv"
    dp = DataProcessor(file_path)
    dp.change_pivot('site_date_tz','param_name','display_value')
    dp.drop_NA_with_feature(features=['PrimaryPressure','FeedTemperature'])
    dp.rename_column_to_timestamp('site_date_tz')
    dp.rename_column_to_feedflow('FeedFlowRate')
    dp.rename_column_to_permeateflow('PermeateFlowRate')
    dp.rename_column_to_feedpressure('PrimaryPressure')
    dp.rename_column_to_concentrateflow('ConcentrateFlowRate')
    
    cp = CycleProcessor(column_name='FeedFlow', df = dp.df, threshold=20)
    cp.identify_cycles()
    cp.assign_cycle_features()

    fe = FeatureEngineering(dp)
    fe.generate_cross_features(drop_features=['Recovery', 'PermeateFlow', 'PermeateConductivity', 'PermeatePressure'])
    fe.lag_engineer()

    dp.df = fe.df
    features =dp.df.columns.tolist()
    top_k_features = pd.read_csv(f"../data/temp_data/top_k_features_plant1_{target_col}.csv")
    top_k_features = top_k_features.iloc[:,1].tolist()
    target = [target_col]

    model_path = f"../model/model_weights_{target_col}.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = dp.df[top_k_features]
    model = load_model(model_path=model_path, X_train=X_train, device=device)
    data = dp.df[top_k_features + target].dropna()
    clean_X = data[top_k_features]
    clean_y = data[target]


    # 验证一致性
    assert len(clean_X) == len(clean_y), "数据长度不一致！"
    x_scaler = joblib.load(f'../data/model_data/scaler_x_{target_col}.pkl')
    y_scaler = joblib.load(f'../data/model_data/scaler_y_{target_col}.pkl')
    X = x_scaler.transform(clean_X.values)
    y = y_scaler.transform(clean_y.values.reshape(-1,1)).flatten()
    dataloader = IndustrialStreamLoader(X_scaled=X, y_scaled=y, seq_length=12, delay_ms=100)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(top_k_features)
    
    file_path = f"../data/prediction/{target_col}/predictions.csv"
    IDF = IndustrialDataFlowSimulator(model=model, data_loader=dataloader, device=device, feature_names=top_k_features,seq_length=12, output_path=file_path)
    results = IDF.run_simulation()


    

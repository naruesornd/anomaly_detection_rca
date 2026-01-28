# import sys
# import os
# from data_processor import DataProcessor
# from data_processor import CycleProcessor
# from utils.plot.plot_by_cycle import interactive_cycle_plot,plot_by_cycle
# from utils.plot.plot_by_time_period import PlotByTimePeriod
# from utils.IQR import iqr
# from feature_engineering import FeatureEngineering
# from model.coarse_feature_selection.cfs import random_forest_regressor
# import pandas as pd
# import numpy as np
# from model.fine_feature_selection.ffs import fine_feature_selection
# from model.lstm_model.enhanced_lstm import lstm_model


# def filter_features(features, target):
#     """过滤特征列表，排除空值、时间戳、特定压力/电导率列以及目标变量"""
#     exclude_terms = [
#         '', 
#         'timestamp', 
#         'ConcentratePressure', 
#         'PermeatePressure', 
#         'PermeateConductivity'
#     ]
#     return [f for f in features 
#             if f not in exclude_terms 
#             and not any(excl in f for excl in exclude_terms[1:])  # 检查部分匹配
#             and f != target]

# def model_training(target):
#     sys.path.append(os.path.abspath('../src'))

#     #data processing
#     file_path = "../data/raw/data_factory_1.xlsx"
#     dp = DataProcessor(file_path)
#     dp.change_pivot('timestamp','param_name','value')
#     dp.drop_NA_with_feature(features=['FeedFlow','FeedTemperature'])
#     dp.rename_column_to_timestamp('timestamp')
#     dp.rename_column_to_permeatepressure('Permeate Pressure')
#     #cycle processing
#     cp = CycleProcessor(signal_col='FeedFlow', df = dp.df, threshold=10)
#     cp.identify_cycles()
#     cp.assign_cycle_features()
#     # file_path = '../data/cycle_processing_data/factory1.csv'
#     # cp.export_files(file_path) 

#     #interactive_cycle_plot(cp.df, plot_by_cycle)

#     # pbt = PlotByTimePeriod(cp.df)
#     # month = ['2022-11', '2022-12','2023-01']
#     # pbt.plot_by_month(month,'FeedFlow')

#     # features = dp.list_columns()
#     # features = [f for f in features if f not in ['timestamp']]
#     # iqr(dp.df, features)

#     fe = FeatureEngineering(dp)
#     if target[0] == 'ConcentratePressure':
#         fe.generate_cross_features(drop_features=['Recovery', 'PermeateFlow', 'PermeateConductivity', 'PermeatePressure', 'ConcentratePressure'])
#     else:
#         fe.generate_cross_features(drop_features=['Recovery', 'PermeateFlow', 'PermeateConductivity', 'PermeatePressure'])
#     fe.lag_engineer()

#     dp.df = fe.df
#     features =dp.df.columns.tolist()

#     # fs = filter_features(features, target)
    
#     feature_name = target[0]
#     if feature_name == 'PermeateFlow':
#         fs = [f for f in features if f != ''and 'timestamp' not in f and 'ConcentratePressure' not in f  and 'PermeatePressure' not in f and 'PermeateConductivity' not in f and f not in ['PermeateFlow'] ]
#     if feature_name == 'PermeatePressure':
#         fs = [f for f in features if f != ''and 'timestamp' not in f and 'ConcentratePressure' not in f  and 'PermeateFlow' not in f and 'PermeateConductivity' not in f and f not in ['PermeatePressure'] ]
#     if feature_name == 'PermeateConductivity':
#         fs = [f for f in features if f != ''and 'timestamp' not in f and 'ConcentratePressure' not in f  and 'PermeateFlow' not in f and 'PermeatePressure' not in f and f not in ['PermeateConductivity'] ]
#     if feature_name == 'ConcentratePressure':
#         fs = [f for f in features if f != ''and 'timestamp' not in f and 'PermeateConductivity' not in f  and 'PermeateFlow' not in f and 'PermeatePressure' not in f and f not in ['ConcentratePressure'] ]
#     top_k_features = random_forest_regressor(dp, feature_name, fs, plant_name='plant1')
#     top_k_features = pd.read_csv(f"../data/temp_data/top_k_features_plant1_{feature_name}.csv")
#     top_k_features = top_k_features.iloc[:,1].tolist()
#     #s_features = fine_feature_selection(dp, top_k_features, target)
#     lstm_model(dp, top_k_features, target, feature_name)



####### OLD CODE ########

import sys
import os
from data_processor import DataProcessor
from data_processor import CycleProcessor
from utils.plot.plot_by_cycle import interactive_cycle_plot,plot_by_cycle
from utils.plot.plot_by_time_period import PlotByTimePeriod
from utils.IQR import iqr
from feature_engineering import FeatureEngineering
from model.coarse_feature_selection.cfs import random_forest_regressor
import pandas as pd
import numpy as np
from model.fine_feature_selection.ffs import fine_feature_selection
from model.lstm_model.enhanced_lstm import lstm_model


def filter_features(features, target):
    """过滤特征列表，排除空值、时间戳、特定压力/电导率列以及目标变量"""
    exclude_terms = [
        '', 
        'timestamp', 
        'ConcentratePressure', 
        'PermeatePressure', 
        'PermeateConductivity'
    ]
    return [f for f in features 
            if f not in exclude_terms 
            and not any(excl in f for excl in exclude_terms[1:])  # 检查部分匹配
            and f != target]

def model_training(target):
    sys.path.append(os.path.abspath('../src'))

    #data processing
    file_path = "../data/raw/data_factory_1.xlsx"
    dp = DataProcessor(file_path)
    dp.change_pivot('timestamp','param_name','value')
    dp.drop_NA_with_feature(features=['FeedFlow','FeedTemperature'])
    dp.rename_column_to_timestamp('timestamp')
    dp.rename_column_to_permeatepressure('Permeate Pressure')
    #cycle processing
    cp = CycleProcessor(column_name='FeedFlow', df = dp.df, threshold=10)
    cp.identify_cycles()
    cp.assign_cycle_features()
    # file_path = '../data/cycle_processing_data/factory1.csv'
    # cp.export_files(file_path) 

    #interactive_cycle_plot(cp.df, plot_by_cycle)

    # pbt = PlotByTimePeriod(cp.df)
    # month = ['2022-11', '2022-12','2023-01']
    # pbt.plot_by_month(month,'FeedFlow')

    # features = dp.list_columns()
    # features = [f for f in features if f not in ['timestamp']]
    # iqr(dp.df, features)

    fe = FeatureEngineering(dp)
    if target[0] == 'ConcentratePressure':
        fe.generate_cross_features(drop_features=['Recovery', 'PermeateFlow', 'PermeateConductivity', 'PermeatePressure', 'ConcentratePressure'])
    else:
        fe.generate_cross_features(drop_features=['Recovery', 'PermeateFlow', 'PermeateConductivity', 'PermeatePressure'])
    fe.lag_engineer()

    dp.df = fe.df
    features =dp.df.columns.tolist()

    # fs = filter_features(features, target)
    
    feature_name = target[0]
    if feature_name == 'PermeateFlow':
        fs = [f for f in features if f != ''and 'timestamp' not in f and 'ConcentratePressure' not in f  and 'PermeatePressure' not in f and 'PermeateConductivity' not in f and f not in ['PermeateFlow'] ]
    if feature_name == 'PermeatePressure':
        fs = [f for f in features if f != ''and 'timestamp' not in f and 'ConcentratePressure' not in f  and 'PermeateFlow' not in f and 'PermeateConductivity' not in f and f not in ['PermeatePressure'] ]
    if feature_name == 'PermeateConductivity':
        fs = [f for f in features if f != ''and 'timestamp' not in f and 'ConcentratePressure' not in f  and 'PermeateFlow' not in f and 'PermeatePressure' not in f and f not in ['PermeateConductivity'] ]
    if feature_name == 'ConcentratePressure':
        fs = [f for f in features if f != ''and 'timestamp' not in f and 'PermeateConductivity' not in f  and 'PermeateFlow' not in f and 'PermeatePressure' not in f and f not in ['ConcentratePressure'] ]
    top_k_features = random_forest_regressor(dp, feature_name, fs, plant_name='plant1')
    top_k_features = pd.read_csv(f"../data/temp_data/top_k_features_plant1_{feature_name}.csv")
    top_k_features = top_k_features.iloc[:,1].tolist()
    #s_features = fine_feature_selection(dp, top_k_features, target)
    lstm_model(dp, top_k_features, target, feature_name)




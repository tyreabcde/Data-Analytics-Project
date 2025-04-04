import pandas as pd
import numpy as np
from pathlib import Path
def load_data():
    """讀取 CSV 檔案"""
    data_dir = Path("C:/Users/USER/Desktop/Course/Data Analytics/Final Project/data_analytics_datagame/")
    train_target = pd.read_csv(data_dir / "train_target_events.csv")
    train_source = pd.read_csv(data_dir / "train_source_events.csv")
    test_source =  pd.read_csv(data_dir / "test_source_events.csv")
    return train_source, train_target, test_source

def time_transform(df):
    """轉換時間格式並新增日期、時間、星期幾欄位"""
    df['event_time'] = pd.to_datetime(df['event_time'], unit='s') \
                         .dt.tz_localize('UTC') \
                         .dt.tz_convert('Asia/Taipei')

    df['weekday'] = df['event_time'].dt.weekday + 1
    df['event_time'] = df['event_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df[['date', 'time']] = df['event_time'].str.split(' ', expand=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df.loc[df['time'].str[:2].astype(int) < 1, 'date'] -= pd.Timedelta(days=1)
    return df

def time_category(time_str):
    """根據時間分區間"""
    hour = int(time_str[:2])
    if 1 <= hour < 5:
        return '1st 1h~5h'
    elif 5 <= hour < 9:
        return '2nd 5h~9h'
    elif 9 <= hour < 17:
        return '3rd 9h~17h'
    else:
        return '4th 17h~1h'

def preprocess_data():
    """完整的數據處理流程"""
    train_source, train_target, test_source = load_data()

    train_source = time_transform(train_source)
    train_target = time_transform(train_target)
    test_source = time_transform(test_source)

    train_source['time_category'] = train_source['time'].apply(time_category)
    train_target['time_category'] = train_target['time'].apply(time_category)
    test_source['time_category'] = test_source['time'].apply(time_category)

    return train_source, train_target, test_source


def time_slot_usage_time(data, value_type='proportion'):
    #將播放時間分為四個區間
    df = data.groupby(['user_id', 'time_category', 'date'])['played_duration'].sum().sort_index(level='date').unstack().fillna(0)
    df = df.reset_index().set_index('user_id')
    #將播放時間轉換為比例
    def proportion(row):
        if row['time_category'] in ['1st 1h~5h', '2nd 5h~9h']:
            return row.iloc[1:] / 14400
        else:
            return row.iloc[1:] / 28800
    def log(x):
        return np.log(x+1)

    if value_type == 'proportion':
        df.iloc[:, 1:] = df.apply(proportion, axis=1)
    else:
        df.iloc[:, 1:] = np.log(df.iloc[:, 1:] + 1)

    df = df.reset_index().set_index(['user_id', 'time_category']).unstack(fill_value=0)

    drop_index = [
    (pd.to_datetime('2022-01-01 00:00:00'), '1st 1h~5h'),
    (pd.to_datetime('2022-01-01 00:00:00'), '2nd 5h~9h'),
    (pd.to_datetime('2022-09-17 00:00:00'), '3rd 9h~17h'),
    (pd.to_datetime('2022-09-17 00:00:00'), '4th 17h~1h')
    ]
    df = df.drop(columns=drop_index)

    #將欄位名稱改為time_slot_{i}_p或time_slot_{i}_p
    if value_type == 'proportion':
        col = [f"time_slot_{i}_p" for i in range(len(df.columns))]
    else:
        col = [f"time_slot_{i}_l" for i in range(len(df.columns))]

    df.columns = col
    return df

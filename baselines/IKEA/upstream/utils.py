import json
import pandas as pd
from typing import List

def load_json_database(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_parquet_database(filepath):
    df = pd.read_parquet(filepath)
    data_dict = df.to_dict(orient='list')  # 将 DataFrame 转换为字典
    return data_dict

def load_parquet_database_as_doc(filepath, nums:int=None, max_length:int=None, cols: List[str]=None) -> List[str]:
    df = pd.read_parquet(filepath)
    df['doc'] = ''
    if cols is None:
        cols = [col for col in df.columns if col != 'doc']
    if nums is not None:
        cols = cols[:nums-1]
    
    for col in cols:
        df['doc'] += df[col].astype(str) + ' '
    data_dict = df.to_dict(orient='list')
    
    if max_length is not None:
        return [doc[:max_length] for doc in data_dict['doc']]
    else:
        return data_dict['doc']
    
def load_model(model_name):
    '''Load joblib model.'''
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        print(f"模型 {model_name} 不存在！")
        return None


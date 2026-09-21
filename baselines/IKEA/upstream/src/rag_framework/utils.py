import tqdm
import os
import json
import torch
from torch import Tensor
from typing import List, Dict, Union, Iterator
from sentence_transformers.SentenceTransformer import SentenceTransformer
"""This file defines common helper functions"""

def chunked_matmul(A:Tensor, B:Tensor, step:int, show_progress:bool=False):
    """Matrix multiply, but A is divided into chunks to reduce memory usage.
    Args:
        A: The matrix on the left.
        B: The matrix on the right.
        step: The number of rows of a chunk.
    Return:
        The product of A and B.
    """
    n = len(A)
    results = []
    lbs = tqdm.tqdm(range(0, n, step)) if show_progress else range(0, n, step)
    for lb in lbs:
        ub = min(n, lb+step)
        results.append(torch.matmul(A[lb:ub], B))
    return torch.concatenate(results, dim=0)

def vec_distance(x:Tensor, y:Tensor):
    """Compute the distance of the vectors."""
    return torch.linalg.vector_norm(x-y, dim=-1)

def transpose_json(json_file:os.PathLike, *keys) -> Dict[str,List]:
    """Load json file and convert it to dict of lists.
    For example, the following json
    [{"id": 0, name: "Alice"}, {"id": 1, "name": "Bob"}]
    will be converted to:
    {"id": [0, 1], "name": ["Alice", "Bob"]}
    Args:
        json_file: The json file to load.
        keys: Fileds to include.
    """
    with open(json_file) as f:
        json_data = json.load(f)
    ret = {k:[] for k in keys}
    for record in json_data:
        for k in keys:
            ret[k].append(record[k])
    return ret

def transpose_jsonl(jsonl_file:os.PathLike, *keys) -> Dict[str,List]:
    """Similar to `transpose_json`, but works with jsonl files.
    Args:
        jsonl_file: The jsonl file to load.
        keys: Fileds to include.
    """
    with open(jsonl_file) as f:
        json_data = f.readlines()
    json_data = (json.loads(record) for record in json_data)
    ret = {k:[] for k in keys}
    for record in json_data:
        for k in keys:
            ret[k].append(record[k])
    return ret

def index_bools(lst:List, index:Union[Tensor,List[bool]]) -> Iterator:
    """Similar with indexing a Tensor with Tensor[bool]
    But works on lists and returns an iterator."""
    for item, keep in zip(lst, index):
        if keep:
            yield item

def index_ints(lst:List, index:Union[Tensor,List[int]]) -> Iterator:
    """Similar with indexing a Tensor with Tensor[int]
    But works on lists and returns an iterator."""
    return (lst[i] for i in index)

def dump_json(file_path: str, data: dict):
    """
    将数据保存为 JSON 格式的文件。
    
    参数:
        file_path (str): 要保存的文件路径。
        data (dict): 要保存的数据字典。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


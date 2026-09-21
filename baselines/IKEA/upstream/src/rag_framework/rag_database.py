import json
import os
import tqdm
import torch
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch import Tensor
from typing import Dict, List, Union, Optional, Tuple
"""This file defines the RagDatabase class."""

class RagDatabase:
    """The RAG database."""
    embedding_model: SentenceTransformer
    primary_key_embeddings: Tensor
    columns: Dict[str,List]

    def __init__(self, embedding_model:SentenceTransformer, primary_key_embeddings:Tensor, columns:Dict[str,List]):
        """This is not designed to be called by user. Use `from_texts` intead."""
        self.embedding_model = embedding_model
        self.primary_key_embeddings = primary_key_embeddings
        self.columns = columns
        self.column_lengths = None

    def retrieve_index_and_similarity(self, query:Union[str,Tensor], top_k:int=4) -> Tuple[Tensor, Tensor]:
        """Get the indices of k most similar records from the database."""
        # if a string is provided, convert it to embedding
        if isinstance(query, str):
            query = self.embedding_model.encode(query, convert_to_tensor=True)
        # for bge, all embeddings are normalized, so dot product measures similarity
        similarity = torch.linalg.vecdot(query, self.primary_key_embeddings)
        # get the indices of the k most similar items
        scores, indices = torch.topk(similarity, top_k)
        
        return indices, scores

    def retrieve_with_similarity(self, query:Union[str,Tensor], top_k:int=4, return_index=False) -> Tuple[Dict[str,List], Tensor, Optional[Tensor]]:
        indices, similarities = self.retrieve_index_and_similarity(query, top_k)
        # return the corresponding rows
        if return_index:
            return {k: [v[i] for i in indices] for k, v in self.columns.items()}, similarities, indices
        else:
            return {k: [v[i] for i in indices] for k, v in self.columns.items()}, similarities

    def retrieve(self, query:Union[str,Tensor], top_k:int=4) -> Dict[str,List]:
        items, _ = self.retrieve_with_similarity(query, top_k)
        return items
    
    def get_column_lengths(self):
        self.column_lengths = {key: len(value) for key, value in self.columns.items()}
        return self.column_lengths

    @classmethod
    def load(cls, load_dir:os.PathLike, embedding_model:SentenceTransformer):
        """load the database from a local directory"""
        # load the embeddings to the same device with the embedding model
        device = embedding_model.device
        # load the embeddings
        primary_key_embeddings = torch.load(os.path.join(load_dir, "primary_keys.pth"), weights_only=True).to(device)
        # load the columns
        with open(os.path.join(load_dir, "columns.json")) as json_file:
            columns = json.load(json_file)
        return cls(embedding_model, primary_key_embeddings, columns)

    def save(self, save_dir:os.PathLike):
        """save the database to a local directory."""
        # create directory
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # save the embeddings
        torch.save(self.primary_key_embeddings, os.path.join(save_dir, "primary_keys.pth"))
        # save the columns
        with open(os.path.join(save_dir, "columns.json"), "wt") as json_file:
            json.dump(self.columns, json_file)

    @classmethod
    def from_texts(cls, embedding_model:SentenceTransformer, primary_keys:List[str], columns:Optional[Dict[str,List]]=None, batch_size:int=4):
        """create a database from texts
        Args:
            embedding_model: the embedding model.
            primary_keys: A list of strings. The embedding of each string will be used as the primary key of a row.
            columns: A dict whose keys are the names of the columns, and values are lists representing the content of the columns.
            batch_size: The batch size.
        """
        n = len(primary_keys)
        if columns is None:
            # if columns is not provided, use the primary key as the only column
            columns = {"content": primary_keys}
        else:
            # if columns is provided, check their length
            for column in columns.values():
                assert len(column)==n, "each column must has the same length"
        # compute the embeddings of the primary key in batches
        primary_key_embeddings = []
        for lb in tqdm.tqdm(range(0, n, batch_size)):
            ub = min(n, lb+batch_size)
            primary_key_embeddings.append(embedding_model.encode(primary_keys[lb:ub], convert_to_tensor=True, normalize_embeddings=True))
        primary_key_embeddings = torch.concatenate(primary_key_embeddings, dim=0)
        return cls(embedding_model, primary_key_embeddings, columns)

    def append(self, primary_keys:List[str], columns:Dict[str,List], batch_size:int=4):
        """insert new entries into the database
        Args:
            primary_keys: A list of strings. The embedding of each string will be used as the primary key of a row.
            columns: A dict whose keys are the names of the columns, and values are lists representing the content of the columns.
            batch_size: The batch size.
        """
        n = len(primary_keys)
        primary_key_embeddings = [self.primary_key_embeddings]
        # compute the embeddings of the primary key in batches
        for lb in tqdm.tqdm(range(0, n, batch_size)):
            ub = min(n, lb+batch_size)
            primary_key_embeddings.append(self.embedding_model.encode(primary_keys[lb:ub], convert_to_tensor=True, normalize_embeddings=True))
        # merge the embeddings
        self.primary_key_embeddings = torch.concatenate(primary_key_embeddings)
        # append new data to the columns
        for k in self.columns.keys():
            self.columns[k] += columns[k]

class RagDatabaseWithCounts(RagDatabase):
    """RAG database. The retrieval of each column is counted."""
    retrieval_counts: Tensor

    def __init__(self, embedding_model:SentenceTransformer, primary_key_embeddings:Tensor, columns:Dict[str,List]):
        self.embedding_model = embedding_model
        self.primary_key_embeddings = primary_key_embeddings
        self.columns = columns
        self.retrieval_counts = torch.zeros(
            len(primary_key_embeddings),
            dtype=torch.int64,
            device=embedding_model.device
        )

    def retrieve_index_and_similarity(self, query:Union[str,Tensor], top_k:int=4) -> Tensor:
        indices, similarities = super().retrieve_index_and_similarity(query, top_k)
        self.retrieval_counts[indices] += 1
        return indices, similarities

    def clear_retrieval_counts(self):
        self.retrieval_counts.zero_()

    def append(self, primary_keys:List[str], columns:Dict[str,List], batch_size:int=4):
        super().append(primary_keys, columns, batch_size)
        self.retrieval_counts = torch.cat((
            self.retrieval_counts,
            torch.zeros(
                len(primary_key_embeddings),
                dtype=torch.int64,
                device=self.embedding_model.device
            )
        ), dim=0)


class DPRagDatabase(RagDatabase):
    def __init__(self, embedding_model:SentenceTransformer, primary_key_embeddings:Tensor, columns:Dict[str,List]):
        super().__init__(embedding_model, primary_key_embeddings, columns)

    def dp_retrieve_index_and_similarity(
        self, 
        query: Union[str, Tensor],
        epsilon: float,       # 隐私预算参数
        p: float = 0.1,       # 选择文档的权重比例（默认10%）
        alpha: float = 1.0,   # 权重锐度参数
        min_tau_bins: int = 100  # 离散化分箱数
    ) -> Tuple[Tensor, Tensor]:
        """差分隐私文档检索，基于top-p阈值机制"""
        
        # 1. 计算所有文档的相似度
        if isinstance(query, str):
            query = self.embedding_model.encode(query, convert_to_tensor=True)
        similarity = torch.linalg.vecdot(query, self.primary_key_embeddings)
        
        # 2. 排序相似度（降序）
        sorted_scores, sorted_indices = torch.sort(similarity, descending=True)
        
        # 3. 处理边界情况（所有分数相同）
        if torch.all(sorted_scores == sorted_scores[0]):
            return sorted_indices, sorted_scores
        
        # 4. 计算权重参数
        s_max = torch.max(sorted_scores)
        s_min = torch.min(sorted_scores)
        delta_s = s_max - s_min
        
        # 5. 计算指数权重矩阵（增强高分文档的权重）
        weights = torch.exp(alpha * (sorted_scores - s_max) / delta_s)
        total_weight = torch.sum(weights)
        
        # 6. 离散化tau候选值（平衡精度与效率）
        tau_candidates = torch.linspace(
            s_min.item(), s_max.item(), min_tau_bins
        ).to(sorted_scores.device)
        
        # 7. 计算每个tau的效用值
        utilities = []
        for tau in tau_candidates:
            mask = sorted_scores >= tau
            selected_weight = torch.sum(weights[mask])
            utility = -torch.abs(selected_weight - p * total_weight)
            utilities.append(utility)
        utilities = torch.tensor(utilities, device=sorted_scores.device)
        
        # 8. 指数机制采样tau（隐私保护核心）
        probabilities = torch.exp(epsilon * utilities / 2)
        probabilities /= torch.sum(probabilities)
        sampled_idx = torch.multinomial(probabilities, 1)
        tau = tau_candidates[sampled_idx]
        
        # 9. 根据阈值筛选文档
        mask = sorted_scores >= tau
        return sorted_indices[mask], sorted_scores[mask]
    
    def retrieve_with_similarity(self, 
                                query: Union[str, Tensor],
                                epsilon: float,       # 隐私预算参数
                                p: float = 0.1,       # 选择文档的权重比例（默认10%）
                                alpha: float = 1.0,   # 权重锐度参数
                                min_tau_bins: int = 100,  # 离散化分箱数
                                return_index=False
                                ) -> Tuple[Dict[str,List], Tensor, Optional[Tensor]]:
        
        indices, similarities = self.dp_retrieve_index_and_similarity(query, epsilon, p, alpha, min_tau_bins)
        # return the corresponding rows
        if return_index:
            return {k: [v[i] for i in indices] for k, v in self.columns.items()}, similarities, indices
        else:
            return {k: [v[i] for i in indices] for k, v in self.columns.items()}, similarities, None
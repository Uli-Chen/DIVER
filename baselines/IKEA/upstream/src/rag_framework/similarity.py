from .utils import chunked_matmul, index_bools
import torch
from torch import Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer
from typing import List, Union, Optional

def text_similarity(model:SentenceTransformer, text_0:Union[str,List[str]], text_1:Union[str,List[str]]) -> Tensor:
    """Compute the similarity of texts one by one."""
    embedding_0 = model.encode(text_0, convert_to_tensor=True, normalize_embeddings=True)
    embedding_1 = model.encode(text_1, convert_to_tensor=True, normalize_embeddings=True)
    return torch.linalg.vecdot(embedding_0, embedding_1)

def text_similarity_matrix(model:SentenceTransformer, text_0:List[str], text_1:List[str], batch_size:int=4) -> Tensor:
    """Compute the similarity of texts n by n.
    Args:
        text_0: list of input text.
        text_1: list of input text.
        batch_size: the batch size when doing dot product.
    Return:
        A matrix representing the similarities."""
    embedding_0 = model.encode(text_0, convert_to_tensor=True, normalize_embeddings=True)
    embedding_1 = model.encode(text_1, convert_to_tensor=True, normalize_embeddings=True)
    return chunked_matmul(embedding_0, embedding_1.transpose(0, 1), batch_size)

def self_similarity_matrix(model:SentenceTransformer, texts:List[str], batch_size:int=4) -> Tensor:
    """Equivalent to text_similarity_matrix(model, text, text, batch_size).
    But no repeated computation."""
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return chunked_matmul(embeddings, embeddings.transpose(0, 1), batch_size)

def find_unsimilar_texts(model:SentenceTransformer, texts:List[str], thresh:Optional[float]=None, n_preserve:Optional[int]=None, batch_size:int=4, return_idx:bool=False) -> Union[List[str]|List[int]]:
    """Find the subset of texts that are not similar to each other.
    Args:
        texts: the list of text to process.
        thresh: if specified, this function returns the subset of texts where similarity between each entry is less than `thresh`.
        n_preserve: if specified, this function returns the `n_preserve` most unsimilar texts from the input. You need to specify one and only one of `thresh` and `n_preserve`.
        batch_size: the batch size when doing dot product.
    Returns:
        a list of texts that are not similar to each other.
    """
    assert (thresh is None) ^ (n_preserve is None), "You need to specify one and only one of `thresh` and `n_preserve`."
    n = len(texts)
    # whether to keep each entry
    keep = torch.ones(n, dtype=torch.bool)
    similarities = self_similarity_matrix(model, texts, batch_size)
    if n_preserve is not None:
        n_removed = n - n_preserve
        similarities_sum = torch.sum(similarities, dim=0)
        for i in range(n_removed):
            # greedly remove the entry that is most similar to others
            entry_to_remove = int(torch.argmax(similarities_sum))
            keep[entry_to_remove] = False
            similarities_sum[entry_to_remove] = 0
            similarities_sum -= similarities[entry_to_remove]
    if thresh is not None:
        # for an undirected graph, remove fewest nodes, so that there is no edges left.
        adj_mat = (similarities-torch.eye(n, device=similarities.device)) >= thresh
        degrees = torch.sum(adj_mat, dim=0)
        while torch.any(degrees>0):
            # greedly remove the entry that is most similar to others, i.e. the node with largest degree
            entry_to_remove = int(torch.argmax(degrees))
            keep[entry_to_remove] = False
            degrees[entry_to_remove] = 0
            degrees -= adj_mat[entry_to_remove].to(dtype=degrees.dtype)
        if return_idx:
            return list(index_bools(texts, keep)), torch.nonzero(keep, as_tuple=True)[0].cpu().tolist()
    return list(index_bools(texts, keep))

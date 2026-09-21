from .llm_interface import LlmInterface, rag_chat_template
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from typing import Optional, Any, List, Dict, Tuple, Callable, Iterator, Union
from .rag_database import RagDatabase, DPRagDatabase
from .utils import index_ints
from torch import Tensor

class RagSystem:
    """The RAG system class, with optional reranker support
    Attributes:
        database: The RAG datsbase
        embedding_model: The embedding model for calculating text similarity.
        reranker: The reranker model to further select more related documents.
        format_rerank: A function that turns the database retrieval dict into a string for reranking.
        format_retrieval: A function that turns the database retrieval dict into a string for generation.
        format_template: A function that turns the retrieved text into a chat template.
    """
    database: RagDatabase
    embedding_model: SentenceTransformer
    reranker: Optional[FlagReranker]
    llm: LlmInterface
    format_rerank: Callable[[Dict[str, List]], List[str]]
    format_retrieval: Callable[[Dict[str, List]], List[str]]
    format_template: Callable[[List[str], str], List[Dict[str,str]]]

    def __init__(self,
        database: RagDatabase,
        embedding_model: SentenceTransformer,
        llm: LlmInterface,
        reranker: Optional[FlagReranker] = None,
        format_rerank: Callable = lambda d: d["content"],
        format_retrieval: Callable = lambda d: d["content"],
        format_template = rag_chat_template
    ):
        self.database = database
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.llm = llm
        self.format_rerank = format_rerank
        self.format_retrieval = format_retrieval
        self.format_template = format_template

    def fetch(self, query:str, n_retrieval:int, n_rerank:int, return_index:bool=False) -> Tuple[List[str], List[float], Optional[list]]:
        """Fetches the related documets in the database to a given query.
        If a reranker is provided, it will be used to further select more relevant documents.
        Args:
            query: The query string.
            n_retrieval: The number of documents to retrieve, if a reranker is provided, the reranker will further select n_rerank documents from them.
            n_rerank: The number of documents selected by the reranker and used for generation. If no reranker is provided, this parameter is advised to be set to n_retrieval.
            return_index: Whether to return the index of the retrieved documents.
        Returns:
            docs: A list of strings containing the retrieved documents.
            scores: A list of floats containing the scores of the retrieved documents.
        """
        # first retrieve with similarity
        retrieval, similarity, ds_idxs = self.database.retrieve_with_similarity(query, top_k=n_retrieval, return_index=return_index)
            
        # compute the score for each retrieved document
        if self.reranker is not None:
            # if reranker is provided, use it to further select more relevant documents
            reranker_inputs = self.format_rerank(retrieval)
            reranker_inputs = [(query, r) for r in reranker_inputs]
            scores = self.reranker.compute_score(reranker_inputs)
        else:
            # if no reranker is provided, simply select the top n_rerank documents in the retrieval
            scores = similarity[:n_rerank].tolist()
        
        # select the top n_rerank documents based on the scores
        indices_and_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n_rerank]
        indices = [i for i, _ in indices_and_scores]
        retrieval = {k: list(index_ints(v, indices)) for k, v in retrieval.items()}
        scores = [s for _, s in indices_and_scores]
        
        ds_idxs = ds_idxs.tolist()
        ds_idxs = index_ints(ds_idxs, indices)
        
        if return_index:
            return retrieval, scores, ds_idxs
        else:
            return retrieval, scores

    def generate(self, docs:List[str], question:str, prompt_mode:str = 'default') -> str:
        """Generate a response from the retrieved documents.
        Args:
            docs: A list of strings containing the retrieved documents.
        """
        chat_template = self.format_template(docs, question, prompt_mode)
        return self.llm.generate(chat_template)

    def stream(self, docs:List[str], quesion:str, prompt_mode:str = 'default') -> Iterator[str]:
        """Generate a streaming response from the retrieved documents.
        Args:
            docs: A list of strings containing the retrieved documents.
        """
        chat_template = self.format_template(docs, quesion, prompt_mode)
        return self.llm.stream(chat_template)

    def ask(self, query:str, n_retrieval:int=16, n_rerank:int=4, stream:bool=False, prompt_mode:str = 'default') -> Union[str, Iterator[str]]:
        """Ask the RAG system a question and get the answer
        Args:
            query: The question to be asked.
            n_retrieval: The number of documents to retrieve, if a reranker is provided, the reranker will further select n_rerank documents from them.
            n_rerank: The number of documents selected by the reranker and used for generation. If no reranker is provided, this parameter is advised to be set to n_retrieval.
            stream: Whether to return a streaming response.
        Returns:
            The response from the RAG system.
        """
        retrieval, scores = self.fetch(query, n_retrieval, n_rerank)
        docs = self.format_retrieval(retrieval)
        if stream:
            return self.stream(docs, query, prompt_mode)
        else:
            return self.generate(docs, query, prompt_mode)

class DPRagSystem(RagSystem):
    def __init__(self,
        database: DPRagDatabase,
        embedding_model: SentenceTransformer,
        llm: LlmInterface,
        reranker: Optional[FlagReranker] = None,
        format_rerank: Callable = lambda d: d["content"],
        format_retrieval: Callable = lambda d: d["content"],
        format_template = rag_chat_template
    ):
        self.database = database
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.llm = llm
        self.format_rerank = format_rerank
        self.format_retrieval = format_retrieval
        self.format_template = format_template

    def fetch(self, query:str, 
              epsilon: float = 0.5,       # 隐私预算参数
              p: float = 0.1,       # 选择文档的权重比例（默认10%）
              alpha: float = 1.0,   # 权重锐度参数
              min_tau_bins: int = 100,  # 离散化分箱数
              n_rerank:int=4,  # 重新排序的文档数量
              return_index:bool=True) -> Tuple[List[str], List[float], Optional[list]]:
        """Fetches the related documets in the database to a given query.
        If a reranker is provided, it will be used to further select more relevant documents.
        Args:
            query: The query string.
            epsilon: The privacy budget parameter.
            p: The weight ratio for selecting documents (default is 10%).
            alpha: The sharpness parameter for weights.
            min_tau_bins: The number of bins for discretization.
            return_index: Whether to return the index of the retrieved documents.
        Returns:
            docs: A list of strings containing the retrieved documents.
            scores: A list of floats containing the scores of the retrieved documents.
        """
        # first retrieve with similarity
        retrieval, similarity, ds_idxs = self.database.retrieve_with_similarity(query, epsilon, p, alpha, min_tau_bins, return_index=return_index)
            
        # compute the score for each retrieved document
        if self.reranker is not None:
            # if reranker is provided, use it to further select more relevant documents
            reranker_inputs = self.format_rerank(retrieval)
            reranker_inputs = [(query, r) for r in reranker_inputs]
            scores = self.reranker.compute_score(reranker_inputs)
        else:
            # if no reranker is provided, simply select the top n_rerank documents in the retrieval
            scores = similarity[:n_rerank].tolist() if similarity.size()[0] > n_rerank else similarity.tolist()
        
        # select the top n_rerank documents based on the scores
        indices_and_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n_rerank]
        indices = [i for i, _ in indices_and_scores]
        retrieval = {k: list(index_ints(v, indices)) for k, v in retrieval.items()}
        scores = [s for _, s in indices_and_scores]
        
        ds_idxs = ds_idxs.tolist()
        ds_idxs = index_ints(ds_idxs, indices)
        
        return retrieval, scores, ds_idxs

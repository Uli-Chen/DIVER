from sentence_transformers import SentenceTransformer
import json
import torch
from tqdm import tqdm
from src.rag_framework import RagDatabase, rag_chat_template, OpenAiWrapper, HfWrapper, transpose_jsonl, text_similarity
from FlagEmbedding import FlagReranker
import src.rag_framework as rag_framework
from typing import Optional, Any, List, Dict, Tuple, Callable, Iterator, Union

# minimal RAG implement
class RagSystem:
    def __init__(self, llm:HfWrapper, database:RagDatabase):
        self.llm = llm
        self.database = database
        self.ask_history = dict()
        self.simp_coverage = None
        self.column_lengths = None

    # returns both the answer and the retrievals, with simlaritie of the retrievals with the question
    def ask(self, question, top_k=4, template_mode='default', return_retrieval=False):
        assert template_mode in ['default', 'multi_choice', 'TF_answer', 'multi_choice_explain', 'single_choice'], 'Error: template_mode should be default, multi-choice or TF-answer.'
        retrieval, simlarity, indices = self.database.retrieve_with_similarity(question, top_k=top_k, return_index=True)
        # update the ask history
        indices = indices.tolist()
        for i in indices:
            self.ask_history[i] = self.ask_history.get(i, 0) + 1
        # concatenate and ask
        concatenated_retrieval = ["Question:{} Answer:{}".format(q, a) for q, a in zip(retrieval["question"], retrieval["answer"])]
        chat_template = rag_chat_template(concatenated_retrieval, question, template_mode)
        if not return_retrieval:
            return self.llm.generate(chat_template), [{"text":t, "simlarity":float(s)} for t, s in zip(retrieval, simlarity)]
        else:
            return self.llm.generate(chat_template), [{"text":t, "simlarity":float(s)} for t, s in zip(retrieval, simlarity)], concatenated_retrieval
    
    def get_column_lengths(self):
        self.column_lengths = {key: len(value) for key, value in self.database.columns.items()}
        return self.column_lengths
    
    def get_simp_coverage(self):
        self.get_column_lengths()
        max_length = max(self.column_lengths.values())
        simp_coverage = len(self.ask_history) / max_length
        self.simp_coverage = simp_coverage
        return simp_coverage
    
    def clear_ask_history(self):
        self.ask_history = dict()
        
class reranker_RagSystem(rag_framework.RagSystem):
    def __init__(self, database:RagDatabase, embedding_model, llm:HfWrapper, reranker, 
                format_rerank: Callable = lambda x: [a + b for a, b in zip(x["question"], x["answer"])],
                format_retrieval: Callable = lambda x: [a + b for a, b in zip(x["question"], x["answer"])],
                format_template = rag_chat_template):
        super().__init__(database, embedding_model ,llm, reranker=reranker,format_rerank=format_rerank, format_retrieval=format_retrieval)
        self.ask_history = dict()
        self.simp_coverage = None
        self.retrieval_efficiency = None
        self.column_lengths = None
        self.max_length = None
    
    def ask(self, query:str, n_retrieval:int=16, n_rerank:int=4, stream:bool=False, template_mode:str = "default",  return_retrieval=False) -> Union[Iterator[str], Tuple[str, List[Dict[str, float]], List[str]]]:
        """Ask the RAG system a question and get the answer
        Args:
            query: The question to be asked.
            n_retrieval: The number of documents to retrieve, if a reranker is provided, the reranker will further select n_rerank documents from them.
            n_rerank: The number of documents selected by the reranker and used for generation. If no reranker is provided, this parameter is advised to be set to n_retrieval.
            stream: Whether to return a streaming response.
        Returns:
            The response from the RAG system.
        """
        # assert template_mode in ['default', 'multi_choice', 'TF_answer', 'multi_choice_explain', 'single_choice', 'ask', 'defense'], 'Error: template_mode should be default, multi-choice or TF-answer, single_choice, ask, defense.'
        retrieval, scores, indices = self.fetch(query, n_retrieval, n_rerank, return_index=True)
        # update the ask history
        indices = [i for i in indices]
        for i in indices:
            self.ask_history[i] = self.ask_history.get(i, 0) + 1
        # concatenate and ask
        docs = self.format_retrieval(retrieval)
        if stream:
            return self.stream(docs, query, template_mode)
        else:
            if not return_retrieval:
                return self.generate(docs, query, template_mode), [{"similarity":float(s), "indices":indices} for s in scores]
            else:
                return self.generate(docs, query, template_mode), [{"similarity":float(s), "indices":indices} for s in scores], docs
        
        
    def get_column_lengths(self):
        self.column_lengths = {key: len(value) for key, value in self.database.columns.items()}
        return self.column_lengths
    
    def get_simp_coverage(self):
        self.get_column_lengths()
        self.max_length = max(self.column_lengths.values())
        simp_coverage = len(self.ask_history) / self.max_length
        self.simp_coverage = simp_coverage
        return simp_coverage
    
    def get_rag_efficiency(self):
        sum_value = 0
        for key, value in self.ask_history.items():
            sum_value += value
        self.retrieval_efficiency = len(self.ask_history) / (sum_value+1)
        return self.retrieval_efficiency
    
    def clear_ask_history(self):
        self.ask_history = dict()

class DPRerankerRagSystem(rag_framework.DPRagSystem):
    def __init__(self, database:RagDatabase, embedding_model, llm:HfWrapper, reranker: Optional[FlagReranker] = None, 
                format_rerank: Callable = lambda x: [a + b for a, b in zip(x["question"], x["answer"])],
                format_retrieval: Callable = lambda x: [a + b for a, b in zip(x["question"], x["answer"])],
                format_template = rag_chat_template):
        super().__init__(database, embedding_model ,llm, reranker=reranker,format_rerank=format_rerank, format_retrieval=format_retrieval)
        self.ask_history = dict()
        self.simp_coverage = None
        self.retrieval_efficiency = None
        self.column_lengths = None
        self.max_length = None
    
    def ask(self, query:str, 
            epsilon: float = 0.5,       # 隐私预算参数
            p: float = 0.1,       # 选择文档的权重比例（默认10%）
            alpha: float = 1.0,   # 权重锐度参数
            min_tau_bins: int = 100,  # 离散化分箱数
            n_rerank:int=4,  # 重新排序的文档数量
            template_mode:str = "default",  return_retrieval:bool=False, stream:bool=False) -> Union[Iterator[str], Tuple[str, List[Dict[str, float]], List[str]]]:
        """Ask the RAG system a question and get the answer
        Args:
            query: The question to be asked.
            n_retrieval: The number of documents to retrieve, if a reranker is provided, the reranker will further select n_rerank documents from them.
            n_rerank: The number of documents selected by the reranker and used for generation. If no reranker is provided, this parameter is advised to be set to n_retrieval.
            stream: Whether to return a streaming response.
        Returns:
            The response from the RAG system.
        """
        # assert template_mode in ['default', 'multi_choice', 'TF_answer', 'multi_choice_explain', 'single_choice', 'ask', 'defense'], 'Error: template_mode should be default, multi-choice or TF-answer, single_choice, ask, defense.'
        retrieval, scores, indices = self.fetch(query, epsilon, p, alpha, min_tau_bins, n_rerank, return_index=True)
        # update the ask history
        indices = [i for i in indices]
        for i in indices:
            self.ask_history[i] = self.ask_history.get(i, 0) + 1
        # concatenate and ask
        docs = self.format_retrieval(retrieval)
        if stream:
            return self.stream(docs, query, template_mode)
        else:
            if not return_retrieval:
                return self.generate(docs, query, template_mode), [{"similarity":float(s), "indices":indices} for s in scores]
            else:
                return self.generate(docs, query, template_mode), [{"similarity":float(s), "indices":indices} for s in scores], docs
        
        
    def get_column_lengths(self):
        self.column_lengths = {key: len(value) for key, value in self.database.columns.items()}
        return self.column_lengths
        # return max(self.column_lengths.values())
    
    def get_simp_coverage(self):
        self.get_column_lengths()
        self.max_length = max(self.column_lengths.values())
        simp_coverage = len(self.ask_history) / self.max_length
        self.simp_coverage = simp_coverage
        return simp_coverage
    
    def get_rag_efficiency(self):
        sum_value = 0
        for key, value in self.ask_history.items():
            sum_value += value
        self.retrieval_efficiency = len(self.ask_history) / (sum_value+1)
        return self.retrieval_efficiency
    
    def get_retrieval_number(self):
        sum_value = 0
        for key, value in self.ask_history.items():
            sum_value += value
        self.retrieval_number = sum_value
        return self.retrieval_number
    
    def clear_ask_history(self):
        self.ask_history = dict()
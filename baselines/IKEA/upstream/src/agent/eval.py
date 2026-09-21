from rouge_score import rouge_scorer
from typing import List

def compute_rouge_l(text1: str, text2: str) -> float:
    """
    计算两个文本之间的 ROUGE-L 分数。

    参数：
        text1 (str): 参考文本（reference）
        text2 (str): 生成文本（candidate）

    返回：
        float: ROUGE-L 的 F1 分数
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(text1, text2)
    return scores['rougeL'].fmeasure

def compute_set_rouge_l(text1: str, ref_texts: List[str]) -> float:
    """
    计算两个文本之间的 ROUGE-L 分数。

    参数：
        text1 (str): 参考文本（reference）
        text2 (str): 生成文本（candidate）

    返回：
        float: ROUGE-L 的 F1 分数
    """
    scores_ls = []
    ref_texts.append(" ".join(ref_texts))
    for ref_text in ref_texts:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(text1, ref_text)
        scores_ls.append(scores['rougeL'].fmeasure)
    return sum(scores_ls)/len(scores_ls) if len(scores_ls) > 0 else 0.0

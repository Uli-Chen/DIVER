import os
import time
import json
import random
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Dynamic Greedy Embedding Attack (DGEA) – pipeline‑friendly wrapper
# --------------------------------------------------------------------------------------
# This file turns the original research prototype into a *drop‑in attacker* that follows
# the same high‑level contract as MutationAttacker / RAGThiefAttacker in your pipeline.
# Minimal interface:
#   *   next_query()           → returns a crafted user prompt to send to the RAG system
#   *   process_response(txt)  → update internal state with the LLM answer (optional)
#   *   finished               → bool property telling the pipeline to stop asking
#
# Typical usage inside the extraction loop:
#     attacker = DGEAAttacker(embed_model, stats_df, vectordb, prefix, suffix)
#     while not attacker.finished:
#         prompt = attacker.next_query()          # step ➊ craft query
#         answer, retrieval, sim = rag.ask(prompt) # your existing rag_system call
#         attacker.process_response(answer)       # step ➋ learning / bookkeeping
# --------------------------------------------------------------------------------------
# NOTE 1:  the wrapper is self‑contained – it *does not* import langchain, Chroma, etc.
#          Pass in whatever objects you need (e.g. your VectorStore) at construction.
# NOTE 2:  if you need the full "auto‑harvest" loop (saving csv / checkpoints) just call
#          attacker.run(max_steps) – it is a thin convenience layer around next_query().
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Dynamic Greedy Embedding Attack (DGEA) – pipeline‑ready implementation
# Now **self‑contained**: if an embedding‑statistics CSV is missing, we automatically
# build it from a raw corpus (e.g. Enron e‑mails) using the user‑supplied embedding model.
# --------------------------------------------------------------------------------------

__all__ = ["DGEAAttacker", "build_embedding_statistics"]

from sentence_transformers import SentenceTransformer
import torch, torch.nn.functional as F

class STWrapper:
    """Make SentenceTransformer look like the attacker’s embedding_model."""
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = "cuda"):
        self.st_model = SentenceTransformer(model_name, device=device)
        # 部分版本把 tokenizer 挂在内部 modules 上；这里直接取用 transformers tokenizer
        self.tokenizer = self.st_model._first_module().tokenizer  
        self.device = device

    def _embed(self, text: str):
        vec = self.st_model.encode(
            text,
            normalize_embeddings=True,        # 直接得到单位向量
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return vec.flatten()


# -------------------------------------------------------------
# 0. utility – build CSV on the fly (the script user sent us)
# -------------------------------------------------------------

def build_embedding_statistics(
    *,
    corpus_path: Union[str, Path],   # e.g. Data/Enron/emails.csv or a .jsonl file with "message"/"text" column
    tokenizer,
    embed_fn,                       # callable (str)->Tensor of shape (1,D)
    device: str = "cuda",
    max_rows: int = 2000,
    csv_out: Union[str, Path] = "embedding_statistics.csv",
) -> pd.DataFrame:
    """Compute μ / σ² across *max_rows* documents and write to CSV if needed.

    Parameters
    ----------
    corpus_path : path to csv / jsonl containing texts in col "message" or "text".
    tokenizer    : tokenizer instance (only used to check that model is loaded).
    embed_fn     : function that returns *normalized* embedding.
    device       : torch device.
    max_rows     : limit to this many docs for speed, paper用了2000.
    csv_out      : save path.
    """
    csv_out = Path(csv_out)
    if csv_out.exists():
        return pd.read_csv(csv_out)

    print(f"[DGEA] statistics file {csv_out} not found – generating from {corpus_path} …")

    # ------- load corpus (csv or jsonl) --------------------------------------
    texts: List[str] = []
    corpus_path = Path(corpus_path)
    if corpus_path.suffix == ".csv":
        df = pd.read_csv(corpus_path, nrows=max_rows)
        col = "message" if "message" in df.columns else "text"
        texts = df[col].astype(str).tolist()
    else:  # assume jsonl
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                record = json.loads(line)
                texts.append(record.get("message") or record.get("text") or "")
    if len(texts) == 0:
        raise ValueError("Corpus is empty or missing required column.")

    # ------- embed -----------------------------------------------------------
    embed_list: List[torch.Tensor] = []
    t0 = time.time()
    for i, doc in enumerate(texts):
        emb = embed_fn(doc).to(device)
        embed_list.append(emb)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(texts)} docs → {(time.time()-t0):.1f}s")
            t0 = time.time()
    emb_matrix = torch.stack(embed_list, dim=0)  # (N,D)

    # ------- stats -----------------------------------------------------------
    mean_vec = emb_matrix.mean(dim=0).cpu().numpy()
    var_vec  = emb_matrix.var(dim=0).cpu().numpy()

    df_out = pd.DataFrame({
        "mean": mean_vec.tolist(),
        "variance": var_vec.tolist(),
    })
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_out, index=False)
    print(f"[DGEA] statistics saved to {csv_out}")
    return df_out

class DGEAAttacker:
    """Pipeline‑ready wrapper around the Dynamic Greedy Embedding Attack (DGEA).

    Parameters
    ----------
    embedding_model : object
        Any object exposing a *tokenizer* attribute and an ``_embed(text:str)->List[float]``
        method (identical to the original DGEA code). If you want to use the HF
        ``SentenceTransformer`` directly just wrap it e.g. with ``lambda t: model
        .encode(t, normalize_embeddings=True).flatten().tolist()``.
    stats_df : pd.DataFrame
        A dataframe with ``'mean'`` and ``'variance'`` columns – usually the pre‑computed
        Enron embedding statistics shipped with the paper. One row → one target vector.
    prefix, suffix : str
        Constant strings that frame every crafted prompt. The suffix is *mutated* by the
        gradient‑free GCQ attack to match the target vector.
    k : int, default 20
        Number of neighbours to retrieve when talking to the RAG system. Stored only for
        convenience – the wrapper itself is **LLM‑agnostic**.
    device : str, default "cuda"
        Torch device for the tiny optimisation steps (cosine loss only).
    save_path : str or Path, default "./dgea_ckpt"
        Folder for periodic checkpoints. If it exists and *resume* is ``True`` we pick up
        from the last vector.
    resume : bool, default False
        Continue from a previous run.
    """

    # ----------------------------------------------------------------------------------
    # PUBLIC API -----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    def __init__(
        self,
        embedding_model,
        num_targets:int,
        *,
        stats_df: Optional[pd.DataFrame] = None,
        stats_path: Optional[Union[str, Path]] = None,
        corpus_path: Optional[Union[str, Path]] = None,
        k: int = 20,
        prefix: str = "",
        suffix: str = "!" * 20,
        device: str = "cuda",
        save_path: Union[str, Path] = "./dgea_ckpt",
        resume: bool = False,
        max_rows_for_stats: int = 2000,
    ) -> None:
        self.model = embedding_model
        self.device = torch.device(device)
        self.prefix = prefix.strip()
        self.init_suffix = suffix.strip()
        self.k = k
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # ----------------- stats acquisition --------------------------------
        if stats_df is None:
            if stats_path and Path(stats_path).exists():
                stats_df = pd.read_csv(stats_path)
            elif corpus_path is not None:
                # need tokenizer & embed_fn; we assume embedding_model has .tokenizer and ._embed
                stats_df = build_embedding_statistics(
                    corpus_path=corpus_path,
                    tokenizer=self.model.tokenizer,
                    embed_fn=self.model._embed,
                    device=device,
                    max_rows=max_rows_for_stats,
                    csv_out=stats_path or (self.save_path / "embedding_statistics.csv"),
                )
            else:
                raise ValueError("Provide either stats_df, stats_path, or corpus_path to build stats.")

        # ---- vector distribution ------------------------------------------
        mean_vecs = stats_df["mean"].apply(lambda s: np.array(eval(s)))
        var_vecs  = stats_df["variance"].apply(lambda s: np.array(eval(s)))
        if len(mean_vecs) == 1:                       # 只有一行 ⇒ 重复采样
            μ = mean_vecs.iloc[0];  σ2 = var_vecs.iloc[0]
            self.target_vectors = np.stack(
                [self._sample_from_distribution(μ, σ2) for _ in range(num_targets)]
            )
        else:                                         # 每行一个 μ/σ²
            self.target_vectors = np.stack(
                [self._sample_from_distribution(μ, σ2) for μ, σ2 in zip(mean_vecs, var_vecs)]
            )

        # ---- internal state ------------------------------------------------
        self.idx = 0
        self.embedding_space: List[List[float]] = []
        self.best_suffix: Optional[str] = None

        # ---- resume -------------------------------------------------------
        self.ckpt_file = self.save_path / "dgea_checkpoint.pkl"
        if resume and self.ckpt_file.exists():
            self._load()

    # -------------------- properties -------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.idx >= len(self.target_vectors)

    # -------------------- main step --------------------------------------------------

    def next_query(self) -> str:
        """Craft the *next* adversarial prompt."""
        if self.finished:
            raise StopIteration("All target vectors processed – attacker.finished == True")

        # 1) decide the *target* embedding
        if self.idx == 0:
            target = torch.tensor(self.target_vectors[0], device=self.device)
        else:
            target = torch.tensor(
                self._find_dissimilar_vector(self.target_vectors[self.idx].shape[-1]), device=self.device
            )

        # 2) GCQ‑attack the suffix to approximate the target
        suffix, loss, best_emb = self._gcq_attack(target)
        self.best_suffix = suffix  # keep for possible inspection

        # 3) book‑keeping for next step ------------------------------------------------
        self.embedding_space.append(best_emb)
        query = f"{self.prefix} {suffix}"
        self.idx += 1
        self._save()
        return query

    def process_response(self, text: str) -> None:
        """Optional call – allow DGEA to *grow* its embedding space with new chunks.

        If you pass the LLM answer here the attacker will look for ``Content:"..."``
        snippets, embed them, and enlarge the internal embedding space. This mimics the
        feedback loop of the original script but keeps a clean API.
        """
        contents = self._extract_content(text)
        for c in contents:
            vec = self.model._embed(c)
            if all(np.linalg.norm(np.array(vec.cpu()) - np.array(v.cpu())) > 1e-6 for v in self.embedding_space):
                self.embedding_space.append(vec)
        self._save()

    # -------------------- one‑shot convenience ---------------------------------------

    def run(self, rag, max_steps: int = 100, template_mode: str = "ask", **rag_kwargs):
        """End‑to‑end loop – *only* rely on rag.ask(). Useful for quick offline tests."""
        for _ in range(max_steps):
            if self.finished:
                break
            q = self.next_query()
            ans, *_ = rag.ask(q, template_mode=template_mode, **rag_kwargs)
            self.process_response(ans)

    # ----------------------------------------------------------------------------------
    # INTERNAL HELPERS -----------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # ---------- GCQ attack (token search) ---------------------------------------------

    def _gcq_attack(
        self,
        target_embedding: torch.Tensor,
        *,
        iterations: int = 3,
        topk: int = 4,
        allow_non_ascii: bool = True,
    ) -> Tuple[str, float, List[float]]:
        """Gradient‑free Greedy Cosine Quantisation search over the *suffix* tokens."""
        tokenizer = self.model.tokenizer
        device = self.device

        # encode initial suffix → token IDs
        control_toks = tokenizer.encode(self.init_suffix, add_special_tokens=False, return_tensors="pt")[0].to(device)
        all_tokens = list(range(len(tokenizer)))
        if not allow_non_ascii:
            all_tokens = [t for t in all_tokens if tokenizer.decode([t]).isascii()]

        best_suffix = self.init_suffix
        best_loss = float("inf")
        best_emb: Optional[List[float]] = None

        cosine = nn.CosineSimilarity(dim=1, eps=1e-8)

        def cosine_loss(vec: List[float]) -> float:
            a = torch.tensor(vec, device=device).unsqueeze(0)
            b = target_embedding.unsqueeze(0)
            return 1 - cosine(a, b).item()

        # initial score – gives the loop a baseline
        best_loss = cosine_loss(self.model._embed(f"{self.prefix} {best_suffix}"))

        for _ in range(iterations):
            indices = list(range(len(control_toks)))
            random.shuffle(indices)
            for i in indices:
                current_best_ids = tokenizer.encode(best_suffix, add_special_tokens=False, return_tensors="pt")[0].to(device)
                for tok in random.sample(all_tokens, topk):
                    candidate_ids = current_best_ids.clone()
                    candidate_ids[i] = tok
                    candidate_txt = tokenizer.decode(candidate_ids)
                    full_prompt = f"{self.prefix} {candidate_txt}"
                    loss = cosine_loss(self.model._embed(full_prompt))
                    if loss < best_loss:
                        best_loss = loss
                        best_suffix = candidate_txt
                        best_emb = self.model._embed(full_prompt)
            # optional: print progress – comment out in production
            # print(f"GCQ iter done – best_loss so far: {best_loss:.4f}")

        # final safety – ensure best_emb is at least something
        if best_emb is None:
            best_emb = self.model._embed(f"{self.prefix} {best_suffix}")
        return best_suffix, best_loss, best_emb

    # ---------- target vector utilities ---------------------------------------------

    @staticmethod
    def _sample_from_distribution(mean_vec: np.ndarray, var_vec: np.ndarray) -> np.ndarray:
        """⤵ draw one sample from **N(μ, σ²)** element‑wise."""
        return np.random.normal(loc=mean_vec, scale=np.sqrt(var_vec))

    def _find_dissimilar_vector(self, dim: int) -> np.ndarray:
        """Optimise a *far* vector w.r.t. the current centroid (cosine ≃ -1)."""
        if len(self.embedding_space) == 0:
            # fallback – random in [-0.3,0.3]
            return np.random.uniform(-0.3, 0.3, size=dim)

        centroid = torch.stack(self.embedding_space).mean(dim=0).to(dtype=torch.float32, device=self.device)
        far = torch.randn(dim, device=self.device, requires_grad=True)
        far.data = 0.6 * (far - far.min()) / (far.max() - far.min()) - 0.3
        opt = torch.optim.Adam([far], lr=0.01)
        loss_fn = nn.CosineEmbeddingLoss()
        for _ in range(100): 
            opt.zero_grad()
            loss = loss_fn(far.unsqueeze(0), centroid.unsqueeze(0), torch.tensor([-1.0], device=self.device))
            loss.backward()
            opt.step()
            far.data.clamp_(-0.3, 0.3)
        return far.detach().cpu().numpy()

    # ---------- checkpointing --------------------------------------------------------

    def _save(self) -> None:
        state = {
            "idx": self.idx,
            "embedding_space": self.embedding_space,
            "best_suffix": self.best_suffix,
        }
        with open(self.ckpt_file, "wb") as f:
            pickle.dump(state, f)

    def _load(self) -> None:
        with open(self.ckpt_file, "rb") as f:
            state = pickle.load(f)
        self.idx = state["idx"]
        self.embedding_space = state["embedding_space"]
        self.best_suffix = state["best_suffix"]

    # ---------- tiny helpers ---------------------------------------------------------

    @staticmethod
    def _extract_content(text: str) -> List[str]:
        import re

        # matches  Content: "..."  or   "Content": "..."
        pattern = r'(?:\"?)Content(?:\"?)\s*:\s*\"([^\"]+)\"'
        return re.findall(pattern, text)

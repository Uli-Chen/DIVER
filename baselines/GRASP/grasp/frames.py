"""GRASP §V-D: three frames distant from observed relation-type embeddings."""
from pathlib import Path
import math


class LocalSentenceEncoder:
    """Read existing sentence-transformer weights; never download a model."""
    def __init__(self, model_path):
        self.path = Path(model_path).resolve()
        self.model = None

    def encode(self, texts):
        if self.model is None:
            if not self.path.is_dir():
                raise FileNotFoundError(f'Local sentence encoder missing: {self.path}')
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(self.path), device='cpu', local_files_only=True)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


class FrameSelector:
    def __init__(self, encoder, count=3):
        self.encoder, self.count, self.cache = encoder, count, {}

    def select(self, frames, observed_types):
        if len(frames) < self.count or len(set(frames)) != len(frames):
            raise ValueError('At least three distinct frames are required')
        types = sorted(set(observed_types))
        # The paper does not define the empty-history case; this explicit tie
        # convention uses configured order, with no invented semantic evidence.
        if not types:
            return {'frames':list(frames[:self.count]), 'distances':None,
                    'selection':'empty_history_configured_order'}
        missing = list(dict.fromkeys(x for x in [*frames, *types] if x not in self.cache))
        if missing:
            vectors = self.encoder.encode(missing)
            if len(vectors) != len(missing):raise ValueError('Encoder batch size mismatch')
            for text, vector in zip(missing, vectors):
                values = [float(v) for v in vector]
                norm = math.sqrt(sum(v*v for v in values))
                if not values or not math.isfinite(norm) or norm == 0:
                    raise ValueError('Invalid sentence embedding')
                self.cache[text] = tuple(v/norm for v in values)
        dims = {len(self.cache[x]) for x in [*frames,*types]}
        if len(dims) != 1:raise ValueError('Sentence embedding dimensions differ')
        # Mean pairwise cosine distance is an explicit implementation completion:
        # the paper does not state how distances across observed types aggregate.
        scores = [sum(1-sum(a*b for a,b in zip(self.cache[f],self.cache[t]))
                      for t in types)/len(types) for f in frames]
        order = sorted(range(len(frames)), key=lambda i:(-scores[i],i))[:self.count]
        return {'frames':[frames[i] for i in order],
                'distances':[scores[i] for i in order],
                'selection':'mean_pairwise_cosine_distance'}


def selector_from_config(config):
    from .io import PROJECT_ROOT
    path = Path(config['frame_encoder']['model_path'])
    if not path.is_absolute():path = PROJECT_ROOT/path
    return FrameSelector(LocalSentenceEncoder(path), count=3)

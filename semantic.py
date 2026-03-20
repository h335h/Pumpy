import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class SemanticFilter:
    def __init__(self, model_name: str, interest_vector_path: str, threshold: float):
        self.model_name = model_name
        self.interest_vector_path = interest_vector_path
        self.threshold = threshold
        self._model = None
        self._interest_vector = None

    def _load_model(self) -> SentenceTransformer:
        """Ленивая загрузка модели."""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_vector(self) -> np.ndarray:
        """Ленивая загрузка вектора интересов."""
        if self._interest_vector is None:
            self._interest_vector = np.load(self.interest_vector_path)
        return self._interest_vector

    def get_similarity(self, text: str) -> float:
        model = self._load_model()
        vec = self._load_vector()
        emb = model.encode([text[:512]])
        sim = cosine_similarity(emb, vec.reshape(1, -1))[0][0]
        # Выводим значение сходства в консоль
        print(f"DEBUG similarity: {sim:.4f}")
        return float(sim)

    def is_relevant(self, text: str, custom_threshold: Optional[float] = None) -> bool:
        """Проверяет, релевантен ли текст."""
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        return self.get_similarity(text) >= threshold

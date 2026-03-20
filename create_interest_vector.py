from sentence_transformers import SentenceTransformer
import numpy as np
from config import Config

def main():
    model = SentenceTransformer(Config.MODEL_NAME)
    with open(Config.POSITIVE_EXAMPLES_PATH, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    embeddings = model.encode(texts, convert_to_tensor=True)
    interest_vector = embeddings.mean(axis=0).cpu().numpy()
    np.save(Config.INTEREST_VECTOR_PATH, interest_vector)
    print(f"Interest vector saved to {Config.INTEREST_VECTOR_PATH}")

if __name__ == '__main__':
    main()

from gensim.models import Word2Vec
import ujson

FILES = []

for filename in FILES:
    with open(filename+".uts", 'r', encoding="utf-8") as f:
        json = ujson.load(f)
        words = json["texts"]
        print(f"Loaded {len(words)} words")

        print("Training Word2Vec model...")
        model = Word2Vec([words], size=100, window=5, min_count=1, workers=4)
        word_vectors = model.wv
        
        print("Saving word vectors")
        word_vectors.save(filename+".wv")

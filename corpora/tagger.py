from sner import POSClient
import ujson
from tqdm import tqdm
import string

tagger = POSClient(host='localhost', port=9198)

FILES = []

for filename in FILES:
    corpora: str = (open(filename, 'r', encoding="utf-8").read().translate(
        str.maketrans('', '', string.punctuation.replace(".", "")))
    ).lower()

    # Use SNER tagger to generate UPOS tags from corpora
    upos_texts = [
        j for sub in [
            tagger.tag(sentence) for sentence in tqdm(corpora.split('.')) if len(sentence) > 0
        ] for j in sub
    ]

    upos = [upos_text[1] for upos_text in upos_texts]
    texts = [upos_text[0] for upos_text in upos_texts]

    # Generate uts and upos file structures
    ujson.dump({"len": len(upos_texts), "upos": upos, "texts": texts}, open(filename+".uts", 'w'))
    ujson.dump({"len": len(upos_texts), "upos": upos}, open(filename+".upos", 'w'))

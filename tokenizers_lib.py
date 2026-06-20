import re
import os
import sentencepiece as spm


class WordTokenizer:
    name = "word"

    def __init__(self, text):
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        vocab = sorted(set(words))
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text):
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return [self.stoi.get(w, 0) for w in words]

    def decode(self, ids):
        return " ".join(self.itos.get(i, "<unk>") for i in ids)


class CharTokenizer:
    name = "char"

    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)


class SubwordTokenizer:
    name = "subword"

    def __init__(self, text, vocab_size=500, model_prefix="bpe_corpus"):
        corpus_path = f"{model_prefix}_input.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(text)
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        self.sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

"""
BioGPT Tokenizer — faithful reimplementation of the official HuggingFace source
=================================================================================
Reference: microsoft/biogpt  (tokenization_biogpt.py)

Pipeline
--------
  encode : text
             └─► Moses tokenize  (aggressive_dash_splits=True, escape=True)
             └─► BPE per Moses token  (</w> appended inside bpe())
             └─► vocab lookup
             └─► prepend </s> as BOS  (fairseq convention, NOT <s>)

  decode : ids
             └─► id → token string
             └─► strip BPE  (</w> → space)
             └─► Moses detokenize  (rejoin @-@, fix punctuation spacing,
                                     unescape HTML entities)

Moses dependency
----------------
  Install with:  pip install sacremoses
  If sacremoses is not available the tokenizer falls back to a built-in
  regex pre-tokenizer that replicates the most important Moses behaviours
  (aggressive dash splitting, punctuation isolation, HTML escaping).
  The fallback is good enough for inference; for training use sacremoses.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
_ESCAPE: List[Tuple[str, str]] = [
    ("&",  "&amp;"),
    ("|",  "&#124;"),
    ("<",  "&lt;"),
    (">",  "&gt;"),
    ("'",  "&apos;"),
    ('"',  "&quot;"),
    ("[",  "&#91;"),
    ("]",  "&#93;"),
]
_UNESCAPE: List[Tuple[str, str]] = [(v, k) for k, v in reversed(_ESCAPE)]


def _moses_escape(text: str) -> str:
    for src, tgt in _ESCAPE:
        text = text.replace(src, tgt)
    return text


def _moses_unescape(text: str) -> str:
    for src, tgt in _UNESCAPE:
        text = text.replace(src, tgt)
    return text


def _moses_tokenize(text: str) -> List[str]:
    while re.search(r'(\w)\-(\w)', text):
        text = re.sub(r'(\w)\-(\w)', r'\1 @-@ \2', text)
    text = re.sub(r'(?<!@)\-(?!@)', ' - ', text)

    text = re.sub(r'([.,?!;])', r' \1 ', text)

    text = re.sub(r'([\(\[\{<])', r' \1 ', text)
    text = re.sub(r'([\)\]\}>])', r' \1 ', text)

    text = re.sub(r'([%/=+])', r' \1 ', text)

    text = re.sub(r'(\D):(\D)', r'\1 : \2', text)

    tokens_raw = text.strip().split()

    return [_moses_escape(t) for t in tokens_raw if t]


def _moses_detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""

    text = " ".join(tokens)

    text = re.sub(r'\s*@-@\s*', '-', text)

    text = _moses_unescape(text)

    text = re.sub(r'\s+([\.，,;:!?\)\]\}])', r'\1', text)

    text = re.sub(r'([\(\[\{])\s+', r'\1', text)

    text = re.sub(r'(\d)\s+%', r'\1%', text)

    return text.strip()


def _get_pairs(word: Tuple[str, ...]) -> set:
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class BioGPTTokenizer:
    def __init__(self, vocab_file: str, merges_file: str):
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as f:
            merges_raw = f.read().split("\n")[:-1]
        merges = [tuple(line.split()[:2]) for line in merges_raw if line]
        self.bpe_ranks: Dict[Tuple[str, str], int] = dict(zip(merges, range(len(merges))))

        self._cache: Dict[str, str] = {}

        self._use_sacremoses = False
        try:
            import sacremoses
            self._sm = sacremoses
            self._moses_tokenizer_cache: dict = {}
            self._moses_detokenizer_cache: dict = {}
            self._use_sacremoses = True
        except ImportError:
            pass
        self.bos_token    = "</s>"
        self.eos_token    = "</s>"
        self.pad_token    = "<pad>"
        self.unk_token    = "<unk>"

        self.bos_token_id = self.encoder.get(self.bos_token, 2)
        self.eos_token_id = self.encoder.get(self.eos_token, 2)
        self.pad_token_id = self.encoder.get(self.pad_token, 1)
        self.unk_token_id = self.encoder.get(self.unk_token, 3)

        self._special_ids = {
            self.bos_token_id,
            self.pad_token_id,
            self.unk_token_id,
        }

    def _tokenize_moses(self, text: str, lang: str = "en") -> List[str]:
        if self._use_sacremoses:
            if lang not in self._moses_tokenizer_cache:
                self._moses_tokenizer_cache[lang] = self._sm.MosesTokenizer(lang=lang)
            return self._moses_tokenizer_cache[lang].tokenize(
                text, aggressive_dash_splits=True, return_str=False, escape=True
            )
        return _moses_tokenize(text)

    def _detokenize_moses(self, tokens: List[str], lang: str = "en") -> str:
        if self._use_sacremoses:
            if lang not in self._moses_detokenizer_cache:
                self._moses_detokenizer_cache[lang] = self._sm.MosesDetokenizer(lang=lang)
            return self._moses_detokenizer_cache[lang].detokenize(tokens)
        return _moses_detokenize(tokens)

    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]

        word: Tuple[str, ...] = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        result = " ".join(word)
        if result == "\n  </w>":
            result = "\n</w>"

        self._cache[token] = result
        return result
    def tokenize(self, text: str) -> List[str]:
        moses_tokens = self._tokenize_moses(text)
        bpe_tokens: List[str] = []
        for token in moses_tokens:
            if token:
                bpe_tokens.extend(self._bpe(token).split(" "))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.encoder.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.decoder.get(i, self.unk_token) for i in ids]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        tokens: List[str] = []
        for i in ids:
            if skip_special_tokens and i in self._special_ids:
                continue
            if skip_special_tokens and i == self.eos_token_id:
                continue
            tokens.append(self.decoder.get(i, self.unk_token))

        word_str = "".join(t.replace(" ", "").replace("</w>", " ") for t in tokens)
        words = word_str.split()

        return self._detokenize_moses(words)

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        pad: bool = False,
    ) -> Dict:
        all_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if pad:
            max_len = max(len(ids) for ids in all_ids)
            padded, masks = [], []
            for ids in all_ids:
                pl = max_len - len(ids)
                padded.append(ids + [self.pad_token_id] * pl)
                masks.append([1] * len(ids) + [0] * pl)
            return {"input_ids": padded, "attention_mask": masks}
        return {"input_ids": all_ids,
                "attention_mask": [[1] * len(ids) for ids in all_ids]}

    def __len__(self) -> int:
        return len(self.encoder)

    def __repr__(self) -> str:
        backend = "sacremoses" if self._use_sacremoses else "built-in fallback"
        return (f"BioGPTTokenizer(vocab_size={len(self)}, "
                f"bpe_merges={len(self.bpe_ranks)}, moses={backend})")
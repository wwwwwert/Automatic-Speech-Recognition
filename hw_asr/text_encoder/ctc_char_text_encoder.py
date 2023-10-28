import re
from collections import defaultdict
from multiprocessing import Pool
from typing import List, NamedTuple

import torch
from pyctcdecode import build_ctcdecoder


from .char_text_encoder import CharTextEncoder
from hw_asr.utils import ROOT_PATH


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.decoder = None

    def ctc_decode(self, inds: List[int]) -> str:
        text = self.decode(inds)
        splitted = text.split(self.EMPTY_TOK)

        regexpr = r'(.)\1+'
        cleared = [re.sub(regexpr, r'\1', text) for text in splitted]
        return ''.join(cleared)

    def ctc_beam_search(
            self, 
            probs: torch.Tensor,  # shape([input_dim, vocab_size])
            probs_length: torch.Tensor,
            beam_size: int = 100
        ) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        state = defaultdict()
        state[('', self.EMPTY_TOK)] = 1.0
        for frame in probs[:probs_length, :]:
            state = self.extend_and_merge(frame, state)
            state = self.truncate(state, beam_size)

        hypos = [
            Hypothesis(pref, proba.item())
            for (pref, last_char), proba in state.items()
        ]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    
    def extend_and_merge(self, frame, state):
        new_state = defaultdict(float)
        for next_char_idx, next_char_proba in enumerate(frame):
            next_char = self.ind2char[next_char_idx]
            for (pref, last_char), pref_proba in state.items():
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                    # last_char = next_char
                # new_state[(new_pref, last_char)] += pref_proba * next_char_proba

                new_state[(new_pref, next_char)] += pref_proba * next_char_proba
        
        return new_state

    def truncate(self, state, beam_size=25):
        state_list = list(state.items())
        state_list.sort(key=lambda x: x[1], reverse=True)
        return dict(state_list[:beam_size])

    def ctc_lm_beam_search(
            self, 
            batch_probs: torch.Tensor,
            probs_lengths: torch.Tensor,
            beam_width: int=25
        ):
        """
        Performs beam search with LM.
        """
        if self.decoder is None:
            vocab = self.vocab.copy()
            vocab[0] = ''
            vocab = [char.upper() for char in vocab]
            lexicon_path = ROOT_PATH / 'hw_asr' / 'text_encoder' / 'kenlm' / 'librispeech-vocab.txt'
            kenlm_path = ROOT_PATH / 'hw_asr' / 'text_encoder' / 'kenlm' / '3-gram.arpa'
            with open(lexicon_path, "r") as file:
                unigrams = [line.strip() for line in file]
            self.decoder = build_ctcdecoder(
                labels=vocab,
                kenlm_model_path=str(kenlm_path),
                unigrams=unigrams
            )
        decoded = []
        for probs, length in zip(batch_probs, probs_lengths):
            probs = probs[:length, :].detach().numpy()
            decoded.append(self.decoder.decode(probs).lower())

        return decoded
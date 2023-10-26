import logging
from typing import List

from torch import int32, zeros

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    max_time_dim = max([item['spectrogram'].shape[-1] for item in dataset_items])
    spectrogram = zeros(
        len(dataset_items), 
        dataset_items[0]['spectrogram'].shape[1],
        max_time_dim
    )

    max_text_len = max([item['text_encoded'].shape[-1] for item in dataset_items])
    text_encoded = zeros(len(dataset_items), max_text_len)
    text_encoded_length = zeros(len(dataset_items), dtype=int32)
    text = []
    audio_path = []
    spectrogram_length = zeros(len(dataset_items), dtype=int32)

    for idx, item in enumerate(dataset_items):
        item_spec = item['spectrogram']
        item_text_encoded = item['text_encoded']
        item_text = item['text']

        spectrogram[idx, :, :item_spec.shape[-1]] = item_spec
        text_encoded[idx, :item_text_encoded.shape[-1]] = item_text_encoded
        text_encoded_length[idx] = item_text_encoded.shape[-1]
        text.append(item_text)
        spectrogram_length[idx] = item_spec.shape[-1]
        audio_path.append(item['audio_path'])

    return {
        'spectrogram': spectrogram,
        'text_encoded': text_encoded,
        'text_encoded_length': text_encoded_length,
        'text': text,
        'spectrogram_length': spectrogram_length,
        'audio_path': audio_path
    }


from collections.abc import Iterator, Mapping
from typing import Literal, Self

from ctranslate2 import Encoder, StorageView
from numpy import asarray, dtype, int8, int32, int64, ndarray
from torch import Tensor, as_tensor, autocast, inference_mode
from transformers.models.bert import BertForTokenClassification, BertTokenizer

from server.features.ner.protocol import Entity, NamedEntityRecognitionProtocol
from server.utils import huggingface_download

type Labels = Literal["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class NamedEntityExtractor(NamedEntityRecognitionProtocol):
    """
    Summary
    -------
    a class for extracting named entities from text

    Methods
    -------
    extract(texts: list[str]) -> Iterator[Iterator[Entity]]
        extract named entities from a list of texts
    """

    __slots__ = ("encoder", "labels", "model", "model_classifier", "tokeniser")

    def __init__(self, *, model_path: str) -> None:
        model = BertForTokenClassification.from_pretrained(model_path)

        self.encoder = Encoder(model_path, compute_type="auto", max_queued_batches=-1)
        self.tokeniser: BertTokenizer = BertTokenizer.from_pretrained(model_path)
        self.labels: Mapping[int, Labels] = model.config.id2label  # pyright: ignore [reportAttributeAccessIssue]
        self.model_classifier = model.classifier
        self.model_classifier.eval()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        del self.encoder
        del self.model_classifier
        del self.tokeniser

    def convert_indices_to_entity_map(
        self,
        token_label_indices: Tensor,
        special_tokens_mask: ndarray[tuple[int], dtype[int64]],
        offset_mapping: ndarray[tuple[int, Literal[2]], dtype[int64]],
    ) -> Iterator[Entity]:
        """
        Summary
        -------
        convert a sequence of token label indices into a sequence of Entity objects

        Parameters
        ----------
        token_label_indices (Tensor)
            a tensor of token label indices

        special_tokens_mask (ndarray[tuple[int], dtype[int64]])
            a mask indicating which tokens are special tokens

        offset_mapping (ndarray[tuple[int, Literal[2]], dtype[int64]])
            a mapping of token start and end character offsets

        Returns
        -------
        entities (Iterator[Entity])
            an iterator of Entity objects
        """
        return (
            Entity(label=self.labels[int(token_label_index)], start=int(start), end=int(end))
            for token_label_index, is_special_token, (start, end) in zip(
                token_label_indices,
                special_tokens_mask,
                offset_mapping,
                strict=True,
            )
            if not is_special_token
        )

    def extract(self, texts: list[str]) -> Iterator[Iterator[Entity]]:
        """
        Summary
        -------
        extract named entities from a list of texts

        Parameters
        ----------
        texts (list[str])
            the texts to extract named entities from

        Returns
        -------
        entities (Iterator[Iterator[Entity]])
            an iterator of iterators of Entity objects
        """
        batch_dict = self.tokeniser(
            texts,
            padding=True,
            return_tensors="np",
            return_token_type_ids=False,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )

        input_indices: ndarray[tuple[int, int], dtype[int8]] = batch_dict["input_ids"]  # pyright: ignore [reportAssignmentType]
        attention_mask: ndarray[tuple[int, int], dtype[int8]] = batch_dict["attention_mask"]  # pyright: ignore [reportAssignmentType]
        special_tokens_mask_batch: ndarray[tuple[int, int], dtype[int64]] = batch_dict["special_tokens_mask"]  # pyright: ignore [reportAssignmentType]
        offset_mapping_batch: ndarray[tuple[int, int, Literal[2]], dtype[int64]] = batch_dict["offset_mapping"]  # pyright: ignore [reportAssignmentType]
        output = self.encoder.forward_batch(
            StorageView.from_array(input_indices.astype(int32)),
            StorageView.from_array(attention_mask.sum(1, int32)),
        )

        with inference_mode(), autocast("cpu"):
            token_label_indices_batch = self.model_classifier(as_tensor(asarray(output.last_hidden_state))).argmax(-1)

        return (
            self.convert_indices_to_entity_map(token_label_indices, special_tokens_mask, offset_mapping)
            for token_label_indices, special_tokens_mask, offset_mapping in zip(
                token_label_indices_batch,
                special_tokens_mask_batch,
                offset_mapping_batch,
                strict=True,
            )
        )


def get_named_entity_recognition_model() -> NamedEntityRecognitionProtocol:
    return NamedEntityExtractor(model_path=huggingface_download("winstxnhdw/bert-large-NER-ct2"))

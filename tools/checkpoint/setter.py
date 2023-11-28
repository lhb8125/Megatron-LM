# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

class ModelSetter(ABC):

    @classmethod
    def set_tensor(cls, dst, src):
        if src is not None:
            dst.data.copy_(src)

    @classmethod
    @abstractmethod
    def has_position_embeddings(cls, model):
        pass

    @classmethod
    @abstractmethod
    def set_embeddings(
        cls,
        model,
        word=None,
        pos=None,
    ):
        pass

    @classmethod
    @abstractmethod
    def set_final_norm(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        pass

    @classmethod
    @abstractmethod
    def set_output_word_embeddings(
        cls,
        model,
        emb=None,
    ):
        pass

    @classmethod
    @abstractmethod
    def set_output_layer(
        cls,
        model,
        weight=None,
    ):
        pass

    @classmethod
    @abstractmethod
    def set_pooler(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        pass

    @classmethod
    @abstractmethod
    def set_lm_head(
        cls,
        model,
        dense_weight=None,
        dense_bias=None,
        norm_weight=None,
        norm_bias=None,
    ):
        pass

    @classmethod
    @abstractmethod
    def set_binary_head(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        pass

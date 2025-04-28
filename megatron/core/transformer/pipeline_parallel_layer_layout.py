# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
from functools import lru_cache

import torch

from megatron.core import parallel_state
from megatron.core.transformer.enums import LayerType


class PipelineParallelLayerLayout:
    """Configuration of custom pipeline parallel layer layout."""

    def __repr__(self):
        return self.input_data

    def __init__(self, layout: str | list, pipeline_model_parallel_size: int):
        """Initialize PipelineParallelLayerLayout from a list or a str.
        Format validation will be done here.
        """

        self.input_data = layout
        if isinstance(layout, str):
            layout = eval(layout)
        else:
            layout = copy.deepcopy(layout)
        assert isinstance(layout, list), (
            f"pipeline_model_parallel_layout must be a list or a string of list,"
            f" but got {type(layout)=}."
        )
        assert all(isinstance(row, list) for row in layout), (
            f"pipeline_model_parallel_layout must be a list of lists, but got"
            f" {[type(row) for row in layout]=}"
        )

        if not isinstance(layout[0][0], list):  # 1D PP layout
            # Check PP size and get VPP size
            assert len(layout) % pipeline_model_parallel_size == 0, (
                f"pipeline_model_parallel_layout must be divisible"
                f" by pipeline_model_parallel_size ({len(layout)=},"
                f" {pipeline_model_parallel_size=})"
            )
            virtual_pipeline_model_parallel_size = len(layout) // pipeline_model_parallel_size

            # Transfer 1D layout to 2D layout
            layout = [
                [
                    layout[vpp_rank * pipeline_model_parallel_size + pp_rank]
                    for vpp_rank in range(virtual_pipeline_model_parallel_size)
                ]
                for pp_rank in range(pipeline_model_parallel_size)
            ]
        else:  # 2D PP layout
            # Make sure it is a 2D layout
            assert all(
                all(isinstance(x, list) for x in row) for row in layout
            ), "2D pipeline_model_parallel_layout must be a list of lists of lists"

            # Check PP size and get VPP size
            assert len(layout) == pipeline_model_parallel_size, (
                f"pipeline_model_parallel_layout must be a list of length equal"
                f" to pipeline_model_parallel_size ({len(layout)=}, "
                f" {pipeline_model_parallel_size=})"
            )
            virtual_pipeline_model_parallel_size = len(layout[0])

            # All ranks must have the same vpp size
            vpp_sizes = [len(x) for x in layout]
            assert all(
                x == virtual_pipeline_model_parallel_size for x in vpp_sizes
            ), f'all ranks must have the same number of virtual pipeline stages, {vpp_sizes=}'

        # Transfer all strings in pipeline_model_parallel_layout to LayerType
        for pp_rank in range(pipeline_model_parallel_size):
            for vpp_rank in range(virtual_pipeline_model_parallel_size):
                transferred_layout = []
                for layer_type in layout[pp_rank][vpp_rank]:
                    assert isinstance(layer_type, LayerType) or isinstance(layer_type, str), (
                        f"elementes in pipeline_model_parallel_layout must be LayerType or str,"
                        f" but got {type(layer_type)}."
                    )
                    if isinstance(layer_type, str):
                        layer_type = layer_type.strip().lower()
                        assert (
                            layer_type in LayerType.__members__
                        ), f"{layer_type} is not a valid LayerType"
                        layer_type = LayerType[layer_type]
                    transferred_layout.append(layer_type)
                layout[pp_rank][vpp_rank] = transferred_layout

        """Flatten the pipeline layout in layer id order."""
        flatten_layout = []
        for vpp_rank in range(virtual_pipeline_model_parallel_size):
            for row in layout:
                flatten_layout.extend(row[vpp_rank])

        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.layout = layout
        self.flatten_layout = flatten_layout

    def validate_layer_layout(self, num_layers: int):
        """Check whether the layout is valid."""

        # Check whether the input layer id is valid
        assert all(
            isinstance(x, LayerType) for x in self.flatten_layout
        ), "All layers must be a valid LayerType."

        # Embedding layer and loss layer must be specified
        assert (
            self.flatten_layout[0] == LayerType.embedding
        ), f"The first layer must be embedding, but got {self.flatten_layout[0]}"
        assert (
            self.flatten_layout[-1] == LayerType.loss
        ), f"The last layer must be loss, but got {self.flatten_layout[-1]}"

        # Layer number verification
        assert (
            self.flatten_layout.count(LayerType.embedding) == 1
        ), "Embedding must be specified exactly once"
        assert self.flatten_layout.count(LayerType.loss) == 1, "Loss must be specified exactly once"
        assert (
            self.flatten_layout.count(LayerType.decoder) == num_layers
        ), "Number of decoder layers must match num_layers"

        # TODO: remove them in the future once they are supported
        if self.flatten_layout.count(LayerType.encoder) > 0:
            raise NotImplementedError("Encoder layer is not supported for flexible pipeline layout")
        if self.flatten_layout.count(LayerType.mtp) > 0:
            raise NotImplementedError("MTP layer is not supported for flexible pipeline layout")

    def get_num_layers_to_build(self, layer_type: LayerType = LayerType.decoder):
        """Get the number of layers to build in the pipeline stage"""
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank() or 0

        # Count layer numbers in this stage.
        num_layers_to_build = self.layout[pp_rank][vpp_rank].count(layer_type)
        return num_layers_to_build

    def get_layer_offset(self, layer_type: LayerType = LayerType.decoder):
        """Get the layer offset in the pipeline stage"""
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank() or 0

        # Calculate the offset by summing up the number of
        # layers in all the previous pipeline stages.
        offset = 0
        for _vp_rank in range(vpp_rank + 1):
            for _pp_rank in range(
                self.pipeline_model_parallel_size if _vp_rank < vpp_rank else pp_rank
            ):
                offset += self.layout[_pp_rank][_vp_rank].count(layer_type)
        return offset

    def get_layer_id_list(self, layer_type: LayerType = LayerType.decoder):
        """Get the list of layer_id for each layer in the pipeline stage."""
        offset = self.get_layer_offset(layer_type=layer_type)
        num_layers_to_build = self.get_num_layers_to_build(layer_type=layer_type)
        return list(range(offset, offset + num_layers_to_build))

    def pretty_repr(self):
        """Pretty representation of the custom layout, showing the layers held by each stage.
        Example:
                            VPP rank 0                 VPP rank 1
        PP rank 0           embedding,decoder*2        decoder*2
        PP rank 1-13        decoder*2                  decoder*2
        PP rank 14          decoder*2                  mtp
        PP rank 15          decoder*2                  loss
        """

        matrix = []
        if self.virtual_pipeline_model_parallel_size > 1:
            header = [""] + [
                f"VPP rank {vpp_rank}"
                for vpp_rank in range(self.virtual_pipeline_model_parallel_size)
            ]
            matrix.append(header)

        prev_row_repr, prev_row_start_pp_rank = None, None
        for pp_rank in range(self.pipeline_model_parallel_size + 1):
            row_repr = []
            if pp_rank < self.pipeline_model_parallel_size:
                for vpp_rank in range(self.virtual_pipeline_model_parallel_size):
                    stage = self.layout[pp_rank][vpp_rank]
                    stage_repr = []
                    prev_layer, prev_layer_cnt = None, 0
                    for layer_type in stage + [None]:
                        if layer_type == prev_layer:
                            prev_layer_cnt += 1
                        else:
                            if prev_layer_cnt > 1:
                                stage_repr.append(f"{prev_layer.name}*{prev_layer_cnt}")
                            elif prev_layer_cnt == 1:
                                stage_repr.append(f"{prev_layer.name}")
                            prev_layer, prev_layer_cnt = layer_type, 1
                    if len(stage_repr) == 0:
                        stage_repr.append(f"(empty stage)")
                    row_repr.append(",".join(stage_repr))

            if row_repr != prev_row_repr:
                if prev_row_start_pp_rank == pp_rank - 1:
                    matrix.append([f"PP rank {pp_rank - 1}"] + prev_row_repr)
                elif prev_row_repr is not None:
                    matrix.append(
                        [f"PP rank {prev_row_start_pp_rank}-{pp_rank - 1}"] + prev_row_repr
                    )
                prev_row_repr, prev_row_start_pp_rank = row_repr, pp_rank

        # Indent the matrix to make it more readable
        lens = [max(map(len, col)) for col in zip(*matrix)]
        indents = 8 if self.virtual_pipeline_model_parallel_size <= 4 else 4
        fmt = (" " * indents).join('{{:{}}}'.format(x) for x in lens)
        return "\n".join([fmt.format(*row) for row in matrix])

    @staticmethod
    @lru_cache()
    def from_str(layout, pipeline_model_parallel_size):
        """Parse the pipeline model parallel layout from a string."""
        parsed_layout = PipelineParallelLayerLayout(layout, pipeline_model_parallel_size)
        # Pretty print the layout distribution.
        if torch.distributed.get_rank() == 0:
            print(
                f"Parse pipeline model parallel layout {layout} to:\n" + parsed_layout.pretty_repr()
            )
        return parsed_layout

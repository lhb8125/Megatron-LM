"""CPU-only control collectives, separate from native NCCL model communication.

Different EP groups can finish at different times. A world NCCL metric reduce
must not be interleaved with still-running EP traffic on other ranks. Gloo
statistics rendezvous does not enqueue GPU work or change model gradients.
"""


class HostControl:
    def __init__(self):
        import torch.distributed as dist

        self.group = dist.new_group(backend="gloo")

    def reduce(self, tensor, *, maximum=False):
        import torch.distributed as dist

        value = tensor.detach().to(device="cpu", copy=True)
        dist.all_reduce(
            value, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM, group=self.group
        )
        return value

    def gather(self, value):
        import torch.distributed as dist

        result = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(result, value, group=self.group)
        return result

    def broadcast(self, values):
        import torch.distributed as dist

        dist.broadcast_object_list(values, src=0, group=self.group)

    def barrier(self):
        import torch.distributed as dist

        dist.monitored_barrier(group=self.group, wait_all_ranks=True)

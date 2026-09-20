import torch
import torch.distributed as dist

from mor_mlite.pretraining.control import HostControl


def _host_control_worker(rank, rendezvous):
    from datetime import timedelta

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        control = HostControl()
        value = torch.tensor([rank + 1.0], dtype=torch.float64, requires_grad=True)
        assert control.reduce(value).item() == 3
        assert control.reduce(value, maximum=True).item() == 2
        assert value.item() == rank + 1 and value.requires_grad
        assert control.gather({"rank": rank}) == [{"rank": 0}, {"rank": 1}]
        payload = ["complete" if rank == 0 else None]
        control.broadcast(payload)
        assert payload == ["complete"]
        control.barrier()
    finally:
        dist.destroy_process_group()


def test_host_control_real_two_process_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _host_control_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_host_statistics_are_detached_copied_and_use_explicit_gloo_group(monkeypatch):
    group = object()
    monkeypatch.setattr(
        dist, "new_group", lambda **kw: group if kw == {"backend": "gloo"} else None
    )
    control = HostControl()
    assert control.group is group
    original = torch.tensor([1.0, 2.0], requires_grad=True)

    def reduce(value, *, op, group):
        assert group is control.group
        assert value.device.type == "cpu" and not value.requires_grad
        assert value.data_ptr() != original.data_ptr()
        assert op == dist.ReduceOp.SUM
        value.mul_(4)

    monkeypatch.setattr(dist, "all_reduce", reduce)
    assert torch.equal(control.reduce(original), torch.tensor([4.0, 8.0]))
    assert torch.equal(original, torch.tensor([1.0, 2.0]))


def test_host_control_objects_barrier_and_maximum_never_use_default_group(monkeypatch):
    group = object()
    control = HostControl.__new__(HostControl)
    control.group = group
    monkeypatch.setattr(dist, "get_world_size", lambda actual: 2 if actual is group else 0)
    calls = []
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda output, item, **kw: (
            calls.append(kw),
            output.__setitem__(slice(None), [item, item]),
        ),
    )
    monkeypatch.setattr(dist, "broadcast_object_list", lambda values, **kw: calls.append(kw))
    monkeypatch.setattr(dist, "monitored_barrier", lambda **kw: calls.append(kw))
    monkeypatch.setattr(dist, "all_reduce", lambda value, **kw: calls.append(kw))
    assert control.gather({"input": "hash"}) == [{"input": "hash"}] * 2
    control.broadcast([True])
    control.barrier()
    control.reduce(torch.tensor(1.0), maximum=True)
    assert all(call["group"] is group for call in calls)
    assert calls[2]["wait_all_ranks"] is True
    assert calls[3]["op"] == dist.ReduceOp.MAX

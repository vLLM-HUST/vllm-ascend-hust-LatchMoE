from types import SimpleNamespace

import pytest
import torch

from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
from vllm_moe_offload_ascend.moe_offload.cpu_first_loader import (
    CPU_FIRST_MARKER,
    CPU_FIRST_PROCESSED_MARKER,
    _can_process_cpu_first_weight_pair,
    ensure_moe_layer_id,
    maybe_create_unquantized_cpu_first_weights,
    maybe_process_unquantized_cpu_first_weights,
    maybe_register_unquantized_fixed_slot_weights,
)
from vllm_moe_offload_ascend.moe_offload.host_store import ExpertWeightBundle, HostExpertStore
from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime
from vllm_moe_offload_ascend.moe_offload.slot_bank import ExpertSlotBank, SlotState
from vllm_moe_offload_ascend.moe_offload.transfer_engine import (
    TransferEngine,
    _contiguous_load_runs,
    _try_batch_view,
)


class FakeMethod:
    def __init__(self):
        self.moe = SimpleNamespace(is_act_and_mul=True, has_bias=False)

    def _maybe_pad_weight(self, weight):
        return weight


class FakeRuntime:
    def __init__(self, config):
        self.config = config

    def should_use_fixed_slot_plan_for_layer(self, layer_id):
        return int(layer_id) == 7


class TinyLayer(torch.nn.Module):
    layer_id = 7


def test_ensure_moe_layer_id_recovers_routed_experts_layer_name():
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.12.mlp.experts"

    assert ensure_moe_layer_id(layer) == 12
    assert layer.layer_id == 12


def test_ensure_moe_layer_id_uses_fused_moe_id_before_name_is_available():
    layer = torch.nn.Module()
    layer.moe_layer_id = 7

    assert ensure_moe_layer_id(layer) == 7
    assert layer.layer_id == 7

def test_cpu_first_processing_accepts_vllm_temporary_npu_weights():
    npu_weight = SimpleNamespace(device=SimpleNamespace(type="npu"))
    cpu_weight = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    meta_weight = SimpleNamespace(device=SimpleNamespace(type="meta"))

    assert _can_process_cpu_first_weight_pair(npu_weight, npu_weight)
    assert _can_process_cpu_first_weight_pair(cpu_weight, cpu_weight)
    assert not _can_process_cpu_first_weight_pair(npu_weight, cpu_weight)
    assert not _can_process_cpu_first_weight_pair(meta_weight, meta_weight)


def test_cpu_first_create_weights_allocates_offloaded_experts_on_cpu():
    method = FakeMethod()
    layer = TinyLayer()
    config = MoeOffloadConfig(
        enabled=True,
        num_slots=2,
        cpu_first_load=True,
        pin_host_memory=False,
    )

    called = maybe_create_unquantized_cpu_first_weights(
        method,
        layer,
        runtime=FakeRuntime(config),
        num_experts=4,
        hidden_size=3,
        intermediate_size_per_partition=2,
        params_dtype=torch.float32,
        extra_weight_attrs={"weight_loader": lambda *args, **kwargs: None},
    )

    assert called is True
    assert getattr(layer, CPU_FIRST_MARKER) is True
    assert layer.w13_weight.device.type == "cpu"
    assert layer.w2_weight.device.type == "cpu"
    assert tuple(layer.w13_weight.shape) == (4, 4, 3)
    assert tuple(layer.w2_weight.shape) == (4, 3, 2)
    assert hasattr(layer.w13_weight, "weight_loader")
    assert hasattr(layer.w2_weight, "weight_loader")


def test_cpu_first_create_weights_skips_resident_or_disabled_layer():
    method = FakeMethod()
    layer = TinyLayer()
    disabled = MoeOffloadConfig(enabled=True, num_slots=2, cpu_first_load=False)

    called = maybe_create_unquantized_cpu_first_weights(
        method,
        layer,
        runtime=FakeRuntime(disabled),
        num_experts=4,
        hidden_size=3,
        intermediate_size_per_partition=2,
        params_dtype=torch.float32,
        extra_weight_attrs={},
    )

    assert called is False
    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")


def test_cpu_first_process_formats_weights_and_registers_without_host_clone():
    method = FakeMethod()
    layer = TinyLayer()
    config = MoeOffloadConfig(
        enabled=True,
        num_slots=2,
        cpu_first_load=True,
        pin_host_memory=False,
    )
    runtime = MoeOffloadRuntime(config)

    assert maybe_create_unquantized_cpu_first_weights(
        method,
        layer,
        runtime=runtime,
        num_experts=4,
        hidden_size=3,
        intermediate_size_per_partition=2,
        params_dtype=torch.float32,
        extra_weight_attrs={},
    )
    layer.w13_weight.data.copy_(
        torch.arange(layer.w13_weight.numel(), dtype=torch.float32).reshape(layer.w13_weight.shape)
    )
    layer.w2_weight.data.copy_(
        torch.arange(layer.w2_weight.numel(), dtype=torch.float32).reshape(layer.w2_weight.shape)
    )

    processed = maybe_process_unquantized_cpu_first_weights(
        method,
        layer,
        runtime=runtime,
    )

    assert processed is True
    assert getattr(layer, CPU_FIRST_PROCESSED_MARKER) is True
    assert tuple(layer.w13_weight.shape) == (4, 3, 4)
    assert tuple(layer.w2_weight.shape) == (4, 2, 3)
    assert runtime.is_layer_registered(7)
    bundle = runtime._host_store.get(7, 1)
    assert bundle.w13.data_ptr() == layer.w13_weight[1].data_ptr()
    assert bundle.w2.data_ptr() == layer.w2_weight[1].data_ptr()


def test_normal_loading_registers_fixed_slots_and_retains_original_weights():
    layer = TinyLayer()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
        requires_grad=False,
    )
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            cpu_first_load=False,
            release_original_expert_weights=False,
            pin_host_memory=False,
        )
    )

    assert maybe_register_unquantized_fixed_slot_weights(
        layer,
        runtime=runtime,
        slot_device=torch.device("cpu"),
    )
    assert runtime.is_layer_registered(7)
    bundle = runtime._host_store.get(7, 1)
    assert bundle.w13.data_ptr() != layer.w13_weight[1].data_ptr()
    assert bundle.w2.data_ptr() != layer.w2_weight[1].data_ptr()
    assert torch.equal(bundle.w13, layer.w13_weight[1])
    assert torch.equal(bundle.w2, layer.w2_weight[1])
    assert layer.w13_weight.shape == (4, 2, 3)
    assert layer.w2_weight.shape == (4, 3, 2)


def test_normal_loading_fixed_slot_registration_is_idempotent():
    layer = TinyLayer()
    layer.w13_weight = torch.nn.Parameter(torch.ones((4, 2, 3)), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.ones((4, 3, 2)), requires_grad=False)
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(enabled=True, num_slots=2, pin_host_memory=False)
    )

    assert maybe_register_unquantized_fixed_slot_weights(
        layer,
        runtime=runtime,
        slot_device=torch.device("cpu"),
    )
    first_bank = runtime._slot_banks[7]
    assert maybe_register_unquantized_fixed_slot_weights(
        layer,
        runtime=runtime,
        slot_device=torch.device("cpu"),
    )
    assert runtime._slot_banks[7] is first_bank


def test_host_store_clone_tensors_false_adopts_cpu_parameter_views():
    layer = TinyLayer()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
        requires_grad=False,
    )
    store = HostExpertStore()

    store.register_layer(layer, clone_tensors=False)

    bundle = store.get(7, 2)
    assert bundle.w13.data_ptr() == layer.w13_weight[2].data_ptr()
    assert bundle.w2.data_ptr() == layer.w2_weight[2].data_ptr()


def test_host_store_bundles_are_views_of_contiguous_layer_buffers():
    layer = TinyLayer()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
        requires_grad=False,
    )
    store = HostExpertStore()

    store.register_layer(layer, clone_tensors=True)

    layer_buffer = store.get_layer_buffer(7)
    bundle = store.get(7, 3)
    assert layer_buffer.w13.is_contiguous()
    assert layer_buffer.w2.is_contiguous()
    assert bundle.w13.data_ptr() == layer_buffer.w13[3].data_ptr()
    assert bundle.w2.data_ptr() == layer_buffer.w2[3].data_ptr()


def test_host_store_get_layer_buffer_fails_closed_for_unknown_layer():
    store = HostExpertStore()

    with pytest.raises(KeyError, match="layer 99 is not registered"):
        store.get_layer_buffer(99)


def test_transfer_engine_builds_batch_view_for_contiguous_host_experts():
    layer = TinyLayer()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
        requires_grad=False,
    )
    store = HostExpertStore()
    store.register_layer(layer, clone_tensors=True)

    contiguous = tuple(store.get(7, expert_id).w13 for expert_id in (0, 1, 2))
    non_contiguous = tuple(store.get(7, expert_id).w13 for expert_id in (0, 2))

    batch = _try_batch_view(contiguous)
    assert batch is not None
    assert tuple(batch.shape) == (3, 2, 3)
    assert torch.equal(batch[2], store.get(7, 2).w13)
    assert _try_batch_view(non_contiguous) is None


def test_transfer_engine_splits_run_around_non_contiguous_source():
    layer = TinyLayer()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(5 * 3 * 2, dtype=torch.float32).reshape(5, 3, 2),
        requires_grad=False,
    )
    store = HostExpertStore()
    store.register_layer(layer, clone_tensors=True)
    bank = ExpertSlotBank(
        5,
        tuple(layer.w13_weight.shape[1:]),
        tuple(layer.w2_weight.shape[1:]),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    detached_middle = ExpertWeightBundle(
        layer_id=7,
        expert_id=2,
        w13=store.get(7, 2).w13.clone(),
        w2=store.get(7, 2).w2.clone(),
    )
    bundles = (
        store.get(7, 0),
        store.get(7, 1),
        detached_middle,
        store.get(7, 3),
        store.get(7, 4),
    )
    loads = tuple(
        (bundle, bank.slots[slot_id])
        for slot_id, bundle in enumerate(bundles)
    )

    runs = _contiguous_load_runs(loads)

    assert [[bundle.expert_id for bundle, _ in run] for run in runs] == [
        [0, 1],
        [2],
        [3, 4],
    ]


def test_transfer_engine_load_many_sync_copies_contiguous_run():
    layer = TinyLayer()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
        requires_grad=False,
    )
    store = HostExpertStore()
    store.register_layer(layer, clone_tensors=True)
    bank = ExpertSlotBank(
        4,
        tuple(layer.w13_weight.shape[1:]),
        tuple(layer.w2_weight.shape[1:]),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    loads = [(store.get(7, expert_id), bank.slots[expert_id]) for expert_id in (0, 1, 2)]

    TransferEngine().load_many_sync(loads)

    assert torch.equal(bank.w13_slots[:3], layer.w13_weight[:3])
    assert torch.equal(bank.w2_slots[:3], layer.w2_weight[:3])
    assert all(bank.slots[slot_id].state == SlotState.READY for slot_id in (0, 1, 2))

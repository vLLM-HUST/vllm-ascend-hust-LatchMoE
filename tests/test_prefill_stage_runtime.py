import pytest
import torch

import vllm_moe_offload_ascend.moe_offload.host_store as host_store
from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey
from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime
from vllm_moe_offload_ascend.moe_offload.slot_bank import ExpertSlotBank, SlotState
from vllm_moe_offload_ascend.moe_offload.transfer_engine import TransferReadyEvent


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_id = 7
        self.w13_weight = torch.nn.Parameter(
            torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        )
        self.w2_weight = torch.nn.Parameter(
            torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        )


def test_slot_bank_lookup_expert_id_tracks_resident_index():
    bank = ExpertSlotBank(
        2,
        (1, 1),
        (1, 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    first = bank.allocate_for(ExpertKey(7, 3), step_id=1)
    bank.mark_ready(first.slot_id)
    second = bank.assign_slot(1, ExpertKey(7, 5), step_id=2)
    bank.mark_ready(second.slot_id)

    assert bank.lookup_expert_id(3) is first
    assert bank.lookup_expert_id(5) is second
    assert bank.lookup(ExpertKey(7, 3)) is first

    bank.clear_slot(first.slot_id)
    assert bank.lookup_expert_id(3) is None
    assert bank.lookup(ExpertKey(7, 3)) is None

    bank.assign_transient_slot(second.slot_id, ExpertKey(7, 9), step_id=3)
    assert bank.lookup_expert_id(5) is None
    assert bank.lookup_expert_id(9) is None


def test_suspend_fixed_slot_execution_is_nested_and_exception_safe():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))

    assert runtime.should_use_fixed_slot_plan_for_layer(7) is True
    with runtime.suspend_fixed_slot_execution():
        assert runtime.should_use_fixed_slot_plan_for_layer(7) is False
        with runtime.suspend_fixed_slot_execution():
            assert runtime.should_use_fixed_slot_plan_for_layer(7) is False
        assert runtime.should_use_fixed_slot_plan_for_layer(7) is False
    assert runtime.should_use_fixed_slot_plan_for_layer(7) is True

    with pytest.raises(RuntimeError, match="diagnostic failure"):
        with runtime.suspend_fixed_slot_execution():
            assert runtime.should_use_fixed_slot_plan_for_layer(7) is False
            raise RuntimeError("diagnostic failure")
    assert runtime.should_use_fixed_slot_plan_for_layer(7) is True


def test_transfer_ready_event_binds_slot_owner_and_generation():
    bank = ExpertSlotBank(
        1,
        (1, 1),
        (1, 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    slot = bank.assign_slot(0, ExpertKey(7, 0), step_id=1)
    ready = TransferReadyEvent(object(), ((slot, slot.lease()),))

    ready.mark_ready()
    ready.mark_ready()
    assert slot.state == SlotState.READY

    slot.state = SlotState.LOADING
    stale_ready = TransferReadyEvent(object(), ((slot, slot.lease()),))
    bank.clear_slot(0)
    replacement = bank.assign_slot(0, ExpertKey(7, 1), step_id=2)

    with pytest.raises(RuntimeError, match="stale H2D completion"):
        stale_ready.mark_ready()
    assert replacement.state == SlotState.LOADING
    assert replacement.expert_key == ExpertKey(7, 1)


def test_transfer_ready_event_publishes_only_after_consumer_wait():
    bank = ExpertSlotBank(
        1,
        (1, 1),
        (1, 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    slot = bank.assign_slot(0, ExpertKey(7, 0), step_id=1)
    event = object()
    ready = TransferReadyEvent(event, ((slot, slot.lease()),))
    calls = []

    class ConsumerStream:
        def wait_event(self, observed_event):
            assert observed_event is event
            assert slot.state == SlotState.LOADING
            assert ready.completed is False
            calls.append("wait")

    ready.install_consumer_dependency(ConsumerStream())

    assert calls == ["wait"]
    assert ready.consumer_wait_installed is True
    assert ready.completed is True
    assert slot.state == SlotState.READY


def test_transfer_ready_event_rejects_stale_lease_before_consumer_wait():
    bank = ExpertSlotBank(
        1,
        (1, 1),
        (1, 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    slot = bank.assign_slot(0, ExpertKey(7, 0), step_id=1)
    ready = TransferReadyEvent(object(), ((slot, slot.lease()),))
    bank.clear_slot(0)
    bank.assign_slot(0, ExpertKey(7, 1), step_id=2)

    class ConsumerStream:
        def wait_event(self, _event):
            raise AssertionError("stale transfer must fail before stream handoff")

    with pytest.raises(RuntimeError, match="stale H2D completion"):
        ready.install_consumer_dependency(ConsumerStream())


def test_memory_ledger_is_cached_and_invalidated_on_structural_changes():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            release_original_expert_weights=True,
        )
    )
    empty = runtime.memory_ledger()
    assert runtime.memory_ledger() is empty

    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    after_register = runtime.memory_ledger()
    assert after_register is runtime.memory_ledger()
    assert after_register.registered_layers == 1
    assert after_register.original_expert_weight_bytes > 0
    assert after_register.prefill_stage_bank_bytes == 0

    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
    )
    assert runtime.memory_ledger() is after_register

    plan = runtime.release_original_expert_weights_if_ready(layer)

    assert plan.ready
    after_release = runtime.memory_ledger()
    assert after_release is runtime.memory_ledger()
    assert after_release is not after_register
    assert after_release.original_expert_weight_bytes == 0


def test_memory_ledger_separates_resident_shared_weights_from_shared_gate():
    class SharedExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float32))
            self.expert_gate = torch.nn.Linear(3, 1, bias=False)

    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    shared = SharedExpert()
    runtime.register_resident_shared_weights(
        layer_id=3,
        shared_experts=shared,
        shared_gate=shared.expert_gate,
    )

    ledger = runtime.memory_ledger()
    assert ledger.resident_shared_weight_bytes == shared.weight.numel() * shared.weight.element_size()
    assert ledger.shared_gate_weight_bytes == (
        shared.expert_gate.weight.numel() * shared.expert_gate.weight.element_size()
    )
    assert ledger.host_experts == 0
    assert ledger.slot_bank_bytes == 0

    runtime.register_resident_shared_weights(
        layer_id=3,
        shared_experts=shared,
        shared_gate=shared.expert_gate,
    )
    assert runtime.memory_ledger().total_managed_bytes == ledger.total_managed_bytes


def test_memory_ledger_unwraps_vllm_shared_experts_adapter():
    class SharedExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float32))

    class SharedExpertsAdapter:
        def __init__(self, layer) -> None:
            self._layer = layer

    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    shared = SharedExpert()
    runtime.register_resident_shared_weights(
        layer_id=4,
        shared_experts=SharedExpertsAdapter(shared),
        shared_gate=None,
    )

    assert runtime.memory_ledger().resident_shared_weight_bytes == (
        shared.weight.numel() * shared.weight.element_size()
    )


def test_memory_ledger_counts_prefill_stage_banks_and_mapping_buffers():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    before = runtime.memory_ledger()

    runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
    )
    after = runtime.memory_ledger()

    stage_bank = runtime._prefill_stage_banks[7][0]
    stage_mapping = runtime._prefill_stage_log2phy_buffers[7][0]
    assert after is not before
    assert after.prefill_stage_bank_count == 1
    assert after.prefill_stage_bank_bytes == stage_bank.total_bytes
    assert after.prefill_stage_mapping_bytes == (
        stage_mapping.numel() * stage_mapping.element_size()
    )
    assert after.total_managed_bytes == (
        after.original_expert_weight_bytes
        + after.host_store_bytes
        + after.slot_bank_bytes
        + after.prefill_stage_bank_bytes
        + after.prefill_stage_mapping_bytes
    )


def test_full_layer_prefill_plan_uses_identity_mapping_and_all_experts():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    prepared, payload = runtime.prepare_full_layer_prefill_plan(
        layer_id=7,
        num_logical_experts=4,
        device=torch.device("cpu"),
    )

    assert prepared.physical_expert_count == 4
    assert prepared.log2phy.tolist() == [0, 1, 2, 3]
    assert prepared.mapping.active_experts == (0, 1, 2, 3)
    assert prepared.mapping.active_slot_ids == (0, 1, 2, 3)
    assert prepared.mapping.slot_to_expert == (0, 1, 2, 3)
    assert torch.equal(prepared.w1, layer.w13_weight)
    assert torch.equal(prepared.w2, layer.w2_weight)
    assert payload["num_logical_experts"] == 4
    assert payload["physical_expert_count"] == 4
    assert payload["pool_reused"] is False


def test_full_layer_prefill_plan_reuses_layout_pool_across_layers():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    first_layer = TinyLayer()
    second_layer = TinyLayer()
    second_layer.layer_id = 8
    with torch.no_grad():
        second_layer.w13_weight.add_(1000)
        second_layer.w2_weight.add_(2000)
    runtime.register_layer_for_fixed_slots(
        first_layer,
        slot_device=torch.device("cpu"),
    )
    runtime.register_layer_for_fixed_slots(
        second_layer,
        slot_device=torch.device("cpu"),
    )

    first, first_payload = runtime.prepare_full_layer_prefill_plan(
        layer_id=7,
        num_logical_experts=4,
        device=torch.device("cpu"),
    )
    bank_w1_ptr = first.w1.data_ptr()
    mapping_ptr = first.log2phy.data_ptr()
    second, second_payload = runtime.prepare_full_layer_prefill_plan(
        layer_id=8,
        num_logical_experts=4,
        device=torch.device("cpu"),
    )

    assert first_payload["pool_reused"] is False
    assert second_payload["pool_reused"] is True
    assert second.w1.data_ptr() == bank_w1_ptr
    assert second.log2phy.data_ptr() == mapping_ptr
    assert torch.equal(second.w1, second_layer.w13_weight)
    assert torch.equal(second.w2, second_layer.w2_weight)
    assert len(runtime._full_layer_prefill_pool) == 1


def test_memory_ledger_counts_shared_full_layer_prefill_storage_once():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    before = runtime.memory_ledger()

    prepared, _ = runtime.prepare_full_layer_prefill_plan(
        layer_id=7,
        num_logical_experts=4,
        device=torch.device("cpu"),
    )
    after = runtime.memory_ledger()

    assert after is not before
    assert after.full_layer_prefill_bank_count == 1
    assert after.full_layer_prefill_bank_bytes == (
        prepared.w1.numel() * prepared.w1.element_size()
        + prepared.w2.numel() * prepared.w2.element_size()
    )
    assert after.full_layer_prefill_mapping_bytes == (
        prepared.log2phy.numel() * prepared.log2phy.element_size()
    )
    assert after.total_npu_slot_bytes == (
        after.slot_bank_bytes
        + after.prefill_stage_bank_bytes
        + after.prefill_stage_mapping_bytes
        + after.full_layer_prefill_bank_bytes
        + after.full_layer_prefill_mapping_bytes
    )


def test_full_layer_prefill_plan_rejects_host_expert_count_mismatch():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="host expert count.*4, expected 5"):
        runtime.prepare_full_layer_prefill_plan(
            layer_id=7,
            num_logical_experts=5,
            device=torch.device("cpu"),
        )


def test_prefill_stage_plan_uses_dedicated_buffer_and_log2phy_mapping():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    prepared, ready_event, payload = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
    )

    assert ready_event is None
    assert payload["buffer_index"] == 0
    assert payload["miss_experts"] == [3, 1]
    assert prepared.log2phy.tolist() == [-1, 1, -1, 0]
    assert torch.equal(prepared.w1[0], layer.w13_weight[3])
    assert torch.equal(prepared.w1[1], layer.w13_weight[1])
    assert torch.equal(prepared.w2[0], layer.w2_weight[3])
    assert torch.equal(prepared.w2[1], layer.w2_weight[1])


def test_prefill_stage_plan_canonicalizes_all_hit_wave_into_temp_slots():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    main = runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
    )

    prepared, ready_event, payload = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(1, 3),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
        build_log2phy=False,
    )

    assert ready_event is None
    assert payload["hit_experts"] == [1, 3]
    assert payload["miss_experts"] == []
    assert payload["h2d_bytes"] == 0
    assert payload["d2d_bytes"] > 0
    assert prepared.w1.data_ptr() != main.w1.data_ptr()
    assert torch.equal(prepared.w1[0], layer.w13_weight[1])
    assert torch.equal(prepared.w1[1], layer.w13_weight[3])
    assert torch.equal(prepared.w2[0], layer.w2_weight[1])
    assert torch.equal(prepared.w2[1], layer.w2_weight[3])


def test_prefill_stage_plan_double_buffers_do_not_share_storage():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    first, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
    )
    first_log2phy = first.log2phy.tolist()
    second, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(2, 3),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=1,
        async_load=False,
    )

    assert first.w1.data_ptr() != second.w1.data_ptr()
    assert first.w2.data_ptr() != second.w2.data_ptr()
    assert first_log2phy == [0, 1, -1, -1]
    assert second.log2phy.tolist() == [-1, -1, 0, 1]


def test_prefill_stage_pool_is_shared_across_equal_layout_layers():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    first_layer = TinyLayer()
    second_layer = TinyLayer()
    second_layer.layer_id = 9
    runtime.register_layer_for_fixed_slots(
        first_layer,
        slot_device=torch.device("cpu"),
    )
    runtime.register_layer_for_fixed_slots(
        second_layer,
        slot_device=torch.device("cpu"),
    )

    first, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
    )
    second, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=9,
        active_experts=(2, 3),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
    )

    assert first.w1.data_ptr() == second.w1.data_ptr()
    assert first.w2.data_ptr() == second.w2.data_ptr()
    assert runtime._prefill_stage_banks[7] is runtime._prefill_stage_banks[9]
    assert runtime.memory_ledger().prefill_stage_bank_count == 1
    assert runtime.memory_ledger().prefill_stage_bank_bytes == (
        runtime._prefill_stage_banks[7][0].total_bytes
    )


def test_shared_prefill_stage_pool_remembers_cross_layer_release_event():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    first_layer = TinyLayer()
    second_layer = TinyLayer()
    second_layer.layer_id = 9
    runtime.register_layer_for_fixed_slots(first_layer, slot_device=torch.device("cpu"))
    runtime.register_layer_for_fixed_slots(second_layer, slot_device=torch.device("cpu"))
    event = object()

    runtime.remember_prefill_stage_buffer_release(
        layer_id=7,
        buffer_index=1,
        event=event,
    )

    assert runtime.prefill_stage_buffer_release_event(
        layer_id=9,
        buffer_index=1,
    ) is event


def test_prefill_stage_plan_reuses_fixed_log2phy_per_buffer():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    first, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
    )
    first_ptr = first.log2phy.data_ptr()
    second, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(2,),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
    )
    third, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(3,),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=1,
        async_load=False,
    )

    assert second.log2phy.data_ptr() == first_ptr
    assert second.log2phy.tolist() == [-1, -1, 0, -1]
    assert third.log2phy.data_ptr() != first_ptr
    assert third.log2phy.tolist() == [-1, -1, -1, 0]


def test_prefill_stage_plan_rejects_buffer_outside_configured_capacity():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            prefill_buffer_count=1,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="prefill_buffer_count=1"):
        runtime.prepare_prefill_stage_plan(
            layer_id=7,
            active_experts=(0,),
            num_logical_experts=4,
            device=torch.device("cpu"),
            buffer_index=1,
            async_load=False,
        )


def test_prefill_stage_plan_can_skip_log2phy_for_wave_plan_remap():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    primed, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
    )
    assert primed.log2phy.tolist() == [0, 1, -1, -1]

    prepared, _, payload = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(3, 2),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
        build_log2phy=False,
    )

    assert payload["log2phy_built"] is False
    assert prepared.log2phy.data_ptr() == primed.log2phy.data_ptr()
    assert prepared.log2phy.tolist() == [0, 1, -1, -1]
    assert torch.equal(prepared.w1[0], layer.w13_weight[3])
    assert torch.equal(prepared.w1[1], layer.w13_weight[2])
    assert torch.equal(prepared.w2[0], layer.w2_weight[3])
    assert torch.equal(prepared.w2[1], layer.w2_weight[2])


def test_register_layer_pins_host_store_when_enabled(monkeypatch):
    def fake_pin(tensor):
        return tensor, True, None

    monkeypatch.setattr(host_store, "_maybe_pin_tensor", fake_pin)
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            pin_host_memory=True,
        )
    )
    layer = TinyLayer()

    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    event = runtime.profiling_summary()["events"][-1]
    report = event["payload"]["host_store"]
    assert report["pin_memory_requested"] is True
    assert report["pin_memory_enabled"] is True
    assert report["pinned_tensors"] == 2
    assert report["pin_failures"] == []


def test_register_layer_reports_pin_memory_failure_without_blocking(monkeypatch):
    def fake_pin(tensor):
        return tensor, False, "RuntimeError:no pin support"

    monkeypatch.setattr(host_store, "_maybe_pin_tensor", fake_pin)
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            pin_host_memory=True,
        )
    )
    layer = TinyLayer()

    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    event = runtime.profiling_summary()["events"][-1]
    report = event["payload"]["host_store"]
    assert report["pin_memory_requested"] is True
    assert report["pin_memory_enabled"] is False
    assert report["pinned_tensors"] == 0
    assert len(report["pin_failures"]) == 2


def test_prepare_ready_slot_plan_reuses_main_slot_bank_without_transfer(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    loaded = runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        step_id=5,
    )

    def fail_load_sync(*args, **kwargs):
        raise AssertionError("ready-slot plan must not trigger H2D load")

    def fail_load_async(*args, **kwargs):
        raise AssertionError("ready-slot plan must not trigger async load")

    monkeypatch.setattr(runtime._transfer_engine, "load_sync", fail_load_sync)
    monkeypatch.setattr(runtime._transfer_engine, "load_async", fail_load_async)

    prepared = runtime.prepare_ready_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        step_id=9,
    )

    assert prepared.w1.data_ptr() == runtime._slot_banks[7].w13_slots.data_ptr()
    assert prepared.w2.data_ptr() == runtime._slot_banks[7].w2_slots.data_ptr()
    assert prepared.w1.data_ptr() == loaded.w1.data_ptr()
    assert prepared.log2phy.tolist() == [-1, 1, -1, 0]
    assert runtime._slot_banks[7].lookup(ExpertKey(7, 3)).last_used_step == 9
    assert runtime._slot_banks[7].lookup(ExpertKey(7, 1)).last_used_step == 9
    assert runtime._prefill_stage_banks == {}


def test_prepare_ready_slot_plan_can_skip_log2phy_for_wave_plan_remap(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        step_id=5,
    )

    monkeypatch.setattr(
        runtime._transfer_engine,
        "load_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ready-slot plan must not trigger H2D load")
        ),
    )

    prepared = runtime.prepare_ready_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        step_id=9,
        build_log2phy=False,
    )

    assert prepared.w1.data_ptr() == runtime._slot_banks[7].w13_slots.data_ptr()
    assert prepared.w2.data_ptr() == runtime._slot_banks[7].w2_slots.data_ptr()
    assert prepared.log2phy.numel() == 0
    assert prepared.mapping.active_slot_ids == (0, 1)
    assert runtime.ready_slot_ids_for_experts(
        layer_id=7,
        expert_ids=(3, 1, 2),
    ) == {3: 0, 1: 1}
    readiness, ready_slot_ids = runtime.slot_readiness_and_ready_slot_ids_for_experts(
        layer_id=7,
        expert_ids=(3, 1, 2),
    )
    assert readiness == {3: True, 1: True, 2: False}
    assert ready_slot_ids == {3: 0, 1: 1}
    assert runtime._prefill_stage_banks == {}


def test_prepare_ready_slot_plan_fails_closed_on_miss(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3,),
        num_logical_experts=4,
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        runtime._transfer_engine,
        "load_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ready-slot miss must not trigger sync load")
        ),
    )

    with pytest.raises(RuntimeError, match="non-ready experts"):
        runtime.prepare_ready_slot_plan(
            layer_id=7,
            active_experts=(3, 1),
            num_logical_experts=4,
            device=torch.device("cpu"),
        )


def test_prepare_fixed_slot_plan_records_decode_hit_miss_profile(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            gmm_profile_path="/tmp/test-decode-stage-profile.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3,),
        num_logical_experts=4,
        device=torch.device("cpu"),
    )
    load_calls = []
    validate_layout_values = []
    original_load_many_sync = runtime._transfer_engine.load_many_sync

    def counted_load_many_sync(loads, **kwargs):
        load_calls.extend(
            (int(bundle.expert_id), int(slot.slot_id)) for bundle, slot in loads
        )
        validate_layout_values.append(bool(kwargs.get("validate_layout", True)))
        original_load_many_sync(loads, **kwargs)

    monkeypatch.setattr(runtime._transfer_engine, "load_many_sync", counted_load_many_sync)

    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        record_stage_profile=True,
    )

    assert load_calls == [(1, 1)]
    assert validate_layout_values == [False]
    event = runtime.profiling_summary()["events"][-1]
    assert event["name"] == "decode_fixed_slot_stage"
    assert event["layer_id"] == 7
    assert event["payload"]["active_experts"] == [3, 1]
    assert event["payload"]["hit_experts"] == [3]
    assert event["payload"]["miss_experts"] == [1]
    assert event["payload"]["n_hits"] == 1
    assert event["payload"]["n_misses"] == 1
    assert event["payload"]["h2d_bytes"] > 0


def test_decode_stage_profile_can_omit_expert_lists(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS", "0")
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
            gmm_profile_path="/tmp/test-decode-summary-profile.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(3,),
        num_logical_experts=4,
    )

    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
    )

    event = [
        item
        for item in runtime.profiling_summary()["events"]
        if item["name"] == "decode_fixed_slot_stage"
    ][-1]
    payload = event["payload"]
    assert payload["n_active"] == 2
    assert payload["n_hits"] == 1
    assert payload["n_misses"] == 1
    assert payload["expert_lists"] == "omitted"
    assert "active_experts" not in payload
    assert "hit_experts" not in payload
    assert "miss_experts" not in payload
    assert payload["h2d_bytes"] > 0
    assert payload["mapping_mode"] == "persistent_log2phy"


def test_decode_stage_profile_can_be_sampled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE", "2")
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
            gmm_profile_path="/tmp/test-decode-sampled-profile.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0,),
        num_logical_experts=4,
    )
    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(1,),
        num_logical_experts=4,
    )
    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0,),
        num_logical_experts=4,
    )

    events = [
        item
        for item in runtime.profiling_summary()["events"]
        if item["name"] == "decode_fixed_slot_stage"
    ]
    assert [event["payload"]["step_id"] for event in events] == [0, 2]
    assert all(event["payload"]["profile_sample_rate"] == 2 for event in events)


def test_prepare_fixed_slot_plan_profile_uses_cached_expert_bytes(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            gmm_profile_path="/tmp/test-decode-profile-cached-bytes.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    monkeypatch.setattr(
        runtime,
        "estimate_expert_weight_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("profile hot path must use cached expert bytes")
        ),
    )

    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        record_stage_profile=True,
    )

    event = [
        item
        for item in runtime.profiling_summary()["events"]
        if item["name"] == "decode_fixed_slot_stage"
    ][-1]
    assert event["payload"]["n_misses"] == 2
    assert event["payload"]["h2d_bytes"] == (
        runtime._expert_weight_bytes_by_layer[7] * 2
    )


def test_cached_layer_expert_weight_bytes_returns_registered_expert_size():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    assert runtime.cached_layer_expert_weight_bytes(layer_id=7) == (
        runtime._expert_weight_bytes_by_layer[7]
    )
    assert runtime.cached_layer_expert_weight_bytes(layer_id=99) == 0


def test_slot_allocation_waits_for_loading_slots_before_eviction(monkeypatch):
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=1))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    bank = runtime._slot_banks[7]
    key0 = ExpertKey(7, 0)
    slot = bank.allocate_for(key0, step_id=1)
    slot.state = SlotState.LOADING
    sync_calls = []

    def synchronize():
        sync_calls.append("sync")
        slot.state = SlotState.READY

    monkeypatch.setattr(runtime._transfer_engine, "synchronize", synchronize)

    allocated = runtime._allocate_slot_with_loading_fallback(
        bank,
        ExpertKey(7, 1),
        step_id=2,
    )

    assert sync_calls == ["sync"]
    assert allocated.slot_id == 0
    assert allocated.expert_key == ExpertKey(7, 1)
    assert allocated.state == SlotState.LOADING


def test_slot_allocation_waits_for_computing_slots_before_eviction(monkeypatch):
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    prepared = runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        step_id=1,
    )
    bank = runtime._slot_banks[7]

    class FakeEvent:
        def __init__(self):
            self.synchronize_calls = 0

        def synchronize(self):
            self.synchronize_calls += 1

    event = FakeEvent()
    monkeypatch.setattr(
        runtime,
        "_record_current_stream_event_or_none",
        lambda: event,
    )

    handle = runtime.begin_slot_compute(prepared)
    assert handle is not None
    assert [slot.state for slot in bank.slots] == [
        SlotState.COMPUTING,
        SlotState.COMPUTING,
    ]

    runtime.end_slot_compute(handle)
    assert [slot.state for slot in bank.slots] == [
        SlotState.COMPUTING,
        SlotState.COMPUTING,
    ]

    allocated = runtime._allocate_slot_with_loading_fallback(
        bank,
        ExpertKey(7, 2),
        step_id=2,
    )

    assert event.synchronize_calls == 1
    assert allocated.slot_id == 0
    assert allocated.expert_key == ExpertKey(7, 2)
    assert allocated.state == SlotState.LOADING
    assert bank.lookup(ExpertKey(7, 0)) is None
    assert bank.lookup(ExpertKey(7, 1)).state == SlotState.READY


def test_computing_slot_cannot_be_reassigned_before_its_completion_event():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=1))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    prepared = runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(0,),
        num_logical_experts=4,
        device=torch.device("cpu"),
    )
    bank = runtime._slot_banks[7]
    handle = runtime.begin_slot_compute(prepared)
    assert handle is not None

    with pytest.raises(RuntimeError, match="cannot be reassigned"):
        bank.assign_slot(0, ExpertKey(7, 1), step_id=2)
    with pytest.raises(RuntimeError, match="cannot be reassigned"):
        bank.clear_slot(0)

    runtime.end_slot_compute(handle)


def test_prepare_fixed_slot_plan_rolls_back_failed_sync_load(monkeypatch):
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    monkeypatch.setattr(
        runtime._transfer_engine,
        "load_many_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("copy failed")
        ),
    )

    with pytest.raises(RuntimeError, match="copy failed"):
        runtime.prepare_fixed_slot_plan(
            layer_id=7,
            active_experts=(0, 1),
            num_logical_experts=4,
            device=torch.device("cpu"),
        )

    bank = runtime._slot_banks[7]
    assert all(slot.state == SlotState.EMPTY for slot in bank.slots)
    assert bank.lookup(ExpertKey(7, 0)) is None
    assert bank.lookup(ExpertKey(7, 1)) is None


def test_stage_fixed_slot_plan_rolls_back_failed_async_load(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
            async_load=True,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    monkeypatch.setattr(
        runtime._transfer_engine,
        "load_many_async",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("enqueue failed")
        ),
    )

    with pytest.raises(RuntimeError, match="enqueue failed"):
        runtime.stage_fixed_slot_plan(
            layer_id=7,
            active_experts=(0, 1),
            num_logical_experts=4,
        )

    bank = runtime._slot_banks[7]
    assert all(slot.state == SlotState.EMPTY for slot in bank.slots)
    assert bank.lookup(ExpertKey(7, 0)) is None
    assert bank.lookup(ExpertKey(7, 1)) is None


def test_prefill_stage_plan_known_miss_skips_main_slot_lookup():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3,),
        num_logical_experts=4,
        device=torch.device("cpu"),
    )

    prepared, _, payload = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(3,),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
        known_miss=True,
    )

    assert payload["known_miss"] is True
    assert payload["hit_experts"] == []
    assert payload["miss_experts"] == [3]
    assert torch.equal(prepared.w1[0], layer.w13_weight[3])
    assert torch.equal(prepared.w2[0], layer.w2_weight[3])
    assert prepared.w1.data_ptr() != runtime._slot_banks[7].w13_slots.data_ptr()


def test_prefill_stage_plan_known_miss_uses_transient_stage_slots():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    prepared, _, payload = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(2,),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
        known_miss=True,
    )
    stage_bank = runtime._prefill_stage_banks[7][0]

    assert payload["known_miss"] is True
    assert prepared.mapping.slot_to_expert[0] == 2
    assert stage_bank.lookup(ExpertKey(7, 2)) is None
    assert stage_bank.slots[0].expert_key == ExpertKey(7, 2)
    assert torch.equal(prepared.w1[0], layer.w13_weight[2])


def test_stage_bank_assign_and_clear_updates_lookup():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    bank = runtime._get_prefill_stage_bank(
        layer_id=7,
        buffer_index=0,
        template_bank=runtime._slot_banks[7],
    )

    key = ExpertKey(7, 2)
    slot = bank.assign_slot(0, key, step_id=3)

    assert bank.lookup(key) is slot
    bank.clear_slot(0)
    assert bank.lookup(key) is None


def test_full_residency_staging_noops_when_slots_less_than_logical_experts():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    assert runtime.stage_full_residency_slot_plan(layer_id=7) is False
    assert runtime.log2phy_buffer(7).tolist() == [-1, -1, -1, -1]


def test_full_residency_staging_raises_during_capture_if_log2phy_uninitialized(monkeypatch):
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=4,
            graph_compatible_offload=True,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    monkeypatch.setattr(runtime_mod, "_is_current_graph_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="log2phy buffer is still all -1"):
        runtime.stage_full_residency_slot_plan(layer_id=7)


def test_stage_fixed_slot_plan_writes_persistent_log2phy_directly():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
            gmm_profile_path="/tmp/test-decode-persistent-log2phy.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    buf = runtime.log2phy_buffer(7)
    assert buf is not None
    ptr = buf.data_ptr()

    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
    )

    assert prepared.log2phy.data_ptr() == ptr
    assert runtime.log2phy_buffer(7).data_ptr() == ptr
    assert runtime.log2phy_buffer(7).tolist() == [-1, 1, -1, 0]
    events = runtime.profiling_summary()["events"]
    stage_events = [
        event
        for event in events
        if event["name"] == "decode_fixed_slot_stage"
    ]
    assert stage_events[-1]["payload"]["mapping_mode"] == "persistent_log2phy"
    assert not any(event["name"] == "decode_log2phy_commit" for event in events)


def test_capture_safe_slot_weights_locks_fixed_address_fingerprint():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    prepared = runtime.capture_safe_slot_weights(layer_id=7)

    assert prepared is not None
    assert prepared.w1.data_ptr() == runtime._slot_banks[7].w13_slots.data_ptr()
    assert prepared.log2phy.data_ptr() == runtime.log2phy_buffer(7).data_ptr()
    with pytest.raises(RuntimeError, match="addresses are already captured"):
        runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))


def test_post_capture_staging_records_stable_address_evidence():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
        )
    )
    runtime.register_layer_for_fixed_slots(TinyLayer(), slot_device=torch.device("cpu"))
    locked = runtime.capture_safe_slot_weights(layer_id=7)
    assert locked is not None

    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
    )

    events = runtime.profiling_summary()["events"]
    lock = next(event for event in events if event["name"] == "graph_slot_address_lock")
    validation = next(
        event for event in events if event["name"] == "graph_slot_address_validate"
    )
    assert validation["payload"]["matches_capture_fingerprint"] is True
    for key in ("w13_data_ptr", "w2_data_ptr", "log2phy_data_ptr"):
        assert lock["payload"][key] == validation["payload"][key]


def test_begin_slot_compute_refuses_h2d_slot_until_ready_event_completes():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=1))
    runtime.register_layer_for_fixed_slots(TinyLayer(), slot_device=torch.device("cpu"))
    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0,),
        num_logical_experts=4,
    )
    slot = runtime._slot_banks[7].slots[prepared.mapping.active_slot_ids[0]]
    slot.state = SlotState.LOADING
    ready_event = TransferReadyEvent(object(), ((slot, slot.lease()),))

    with pytest.raises(RuntimeError, match="not ready for compute"):
        runtime.begin_slot_compute(prepared)

    ready_event.mark_ready()
    handle = runtime.begin_slot_compute(prepared)
    assert handle is not None
    runtime.end_slot_compute(handle)


def test_decode_waits_before_publishing_async_slot_mapping(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
            async_load=True,
        )
    )
    runtime.register_layer_for_fixed_slots(
        TinyLayer(),
        slot_device=torch.device("cpu"),
    )
    pending_loads = []
    observed = []

    def fake_load_many_async(
        loads,
        *,
        wait_event=None,
        record_event=True,
        validate_layout=True,
    ):
        assert wait_event is None
        assert record_event is True
        assert validate_layout is False
        pending_loads.extend(loads)
        for bundle, slot in loads:
            slot.w13.copy_(bundle.w13)
            slot.w2.copy_(bundle.w2)
            slot.state = SlotState.LOADING
        return "decode-ready"

    def wait_before_publish(ready_event):
        assert ready_event == "decode-ready"
        observed.append(runtime.log2phy_buffer(7).tolist())
        for _, slot in pending_loads:
            slot.state = SlotState.READY

    monkeypatch.setattr(
        runtime._transfer_engine,
        "load_many_async",
        fake_load_many_async,
    )
    monkeypatch.setattr(runtime, "_wait_transfer_event", wait_before_publish)

    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(1, 2),
        num_logical_experts=4,
    )

    assert observed == [[-1, -1, -1, -1]]
    assert prepared.log2phy.tolist() == [-1, 0, 1, -1]


def test_staging_fails_closed_if_captured_slot_address_changes():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(enabled=True, num_slots=1, graph_compatible_offload=True)
    )
    runtime.register_layer_for_fixed_slots(TinyLayer(), slot_device=torch.device("cpu"))
    assert runtime.capture_safe_slot_weights(layer_id=7) is not None

    bank = runtime._slot_banks[7]
    bank.w13_slots = bank.w13_slots.clone()
    with pytest.raises(RuntimeError, match="address fingerprint changed"):
        runtime.stage_fixed_slot_plan(
            layer_id=7,
            active_experts=(0,),
            num_logical_experts=4,
        )


def test_stage_fixed_slot_plan_batches_decode_misses_async(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=3,
            graph_compatible_offload=True,
            async_load=True,
            gmm_profile_path="/tmp/test-decode-async-stage.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3,),
        num_logical_experts=4,
        device=torch.device("cpu"),
        step_id=1,
    )
    calls = []
    waits = []

    def fail_load_sync(*args, **kwargs):
        raise AssertionError("async decode staging must not use per-miss load_sync")

    def fake_load_many_async(
        loads,
        *,
        wait_event=None,
        record_event=True,
        validate_layout=True,
    ):
        calls.append(
            (
                [
                    (int(bundle.expert_id), int(slot.slot_id))
                    for bundle, slot in loads
                ],
                wait_event,
                bool(record_event),
                bool(validate_layout),
            )
        )
        for bundle, slot in loads:
            slot.w13.copy_(bundle.w13)
            slot.w2.copy_(bundle.w2)
            slot.state = SlotState.READY
        return "decode-ready"

    monkeypatch.setattr(runtime._transfer_engine, "load_sync", fail_load_sync)
    monkeypatch.setattr(runtime._transfer_engine, "load_many_async", fake_load_many_async)
    monkeypatch.setattr(
        runtime,
        "_wait_transfer_event",
        lambda ready_event: waits.append(ready_event),
    )

    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1, 2),
        num_logical_experts=4,
    )

    assert calls == [([(1, 1), (2, 2)], None, True, False)]
    assert waits == ["decode-ready"]
    assert prepared.log2phy.tolist() == [-1, 1, 2, 0]
    assert runtime.log2phy_buffer(7).tolist() == [-1, 1, 2, 0]
    event = [
        item
        for item in runtime.profiling_summary()["events"]
        if item["name"] == "decode_fixed_slot_stage"
    ][-1]
    assert event["payload"]["hit_experts"] == [3]
    assert event["payload"]["miss_experts"] == [1, 2]
    assert event["payload"]["stage_mode"] == "async_decode_load_many"
    assert event["payload"]["mapping_mode"] == "persistent_log2phy"
    assert "load_enqueue_ms" in event["payload"]
    assert "ready_wait_ms" in event["payload"]


def test_stage_fixed_slot_plan_profile_uses_cached_expert_bytes(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
            async_load=True,
            gmm_profile_path="/tmp/test-stage-profile-cached-bytes.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    monkeypatch.setattr(
        runtime,
        "estimate_expert_weight_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stage profile hot path must use cached expert bytes")
        ),
    )

    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
    )

    event = [
        item
        for item in runtime.profiling_summary()["events"]
        if item["name"] == "decode_fixed_slot_stage"
    ][-1]
    assert event["payload"]["n_misses"] == 2
    assert event["payload"]["h2d_bytes"] == (
        runtime._expert_weight_bytes_by_layer[7] * 2
    )


def test_stage_fixed_slot_plan_async_hit_only_skips_loader(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
            async_load=True,
            gmm_profile_path="/tmp/test-decode-async-hit-only.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        step_id=1,
    )
    monkeypatch.setattr(
        runtime._transfer_engine,
        "load_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hit-only decode stage must not load")
        ),
    )
    monkeypatch.setattr(
        runtime._transfer_engine,
        "load_many_async",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hit-only decode stage must not enqueue async load")
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_wait_transfer_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hit-only decode stage must not wait")
        ),
    )

    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
    )

    assert prepared.log2phy.tolist() == [-1, 1, -1, 0]
    event = [
        item
        for item in runtime.profiling_summary()["events"]
        if item["name"] == "decode_fixed_slot_stage"
    ][-1]
    assert event["payload"]["n_hits"] == 2
    assert event["payload"]["n_misses"] == 0
    assert event["payload"]["stage_mode"] == "main_slot_hit"
    assert event["payload"]["log2phy_update_count"] == 2


def test_stage_fixed_slot_plan_skips_redundant_hit_only_log2phy_update(monkeypatch):
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=2,
            graph_compatible_offload=True,
            async_load=True,
            gmm_profile_path="/tmp/test-decode-log2phy-hit-skip.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
    )

    def forbidden_update(*args, **kwargs):
        raise AssertionError("unchanged hit-only decode must not rewrite log2phy")

    monkeypatch.setattr(runtime_mod, "_update_log2phy_entries", forbidden_update)
    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(3, 1),
        num_logical_experts=4,
    )

    assert prepared.log2phy.tolist() == [-1, 1, -1, 0]
    event = [
        item
        for item in runtime.profiling_summary()["events"]
        if item["name"] == "decode_fixed_slot_stage"
    ][-1]
    assert event["payload"]["n_hits"] == 2
    assert event["payload"]["n_misses"] == 0
    assert event["payload"]["stage_mode"] == "main_slot_hit"
    assert event["payload"]["log2phy_update_count"] == 0


def test_stage_fixed_slot_plan_updates_only_changed_log2phy_entries(monkeypatch):
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=3,
            graph_compatible_offload=True,
            async_load=True,
            gmm_profile_path="/tmp/test-decode-log2phy-changed-only.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1, 2),
        num_logical_experts=4,
    )
    original_update = runtime_mod._update_log2phy_entries
    calls = []

    def recording_update(log2phy, *, expert_ids, slot_ids):
        calls.append((tuple(expert_ids), tuple(slot_ids)))
        return original_update(log2phy, expert_ids=expert_ids, slot_ids=slot_ids)

    monkeypatch.setattr(runtime_mod, "_update_log2phy_entries", recording_update)
    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(1, 3),
        num_logical_experts=4,
    )

    assert calls == [((3,), (0,))]
    assert prepared.log2phy.tolist() == [0, 1, 2, 0]
    assert prepared.mapping.active_experts == (1, 3)
    assert prepared.mapping.active_slot_ids == (1, 0)
    event = [
        item
        for item in runtime.profiling_summary()["events"]
        if item["name"] == "decode_fixed_slot_stage"
    ][-1]
    assert event["payload"]["log2phy_update_count"] == 1


def test_update_log2phy_entries_single_update_skips_tensor_materialization(monkeypatch):
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    log2phy = torch.full((4,), -1, dtype=torch.long)

    def forbidden_as_tensor(*args, **kwargs):
        raise AssertionError("single-entry log2phy update should not materialize tensors")

    monkeypatch.setattr(runtime_mod.torch, "as_tensor", forbidden_as_tensor)
    runtime_mod._update_log2phy_entries(
        log2phy,
        expert_ids=(2,),
        slot_ids=(1,),
    )

    assert log2phy.tolist() == [-1, -1, 1, -1]


def test_stage_fixed_slot_plan_updates_only_active_log2phy_entries(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=3,
            graph_compatible_offload=True,
            async_load=True,
            gmm_profile_path="/tmp/test-decode-log2phy-active-only.jsonl",
        )
    )
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1, 2),
        num_logical_experts=4,
    )
    before = runtime.log2phy_buffer(7).clone()

    monkeypatch.setattr(
        runtime._transfer_engine,
        "load_many_async",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hit-only decode stage must not enqueue async load")
        ),
    )
    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(1, 2),
        num_logical_experts=4,
    )

    assert before.tolist() == [0, 1, 2, -1]
    assert prepared.log2phy[1].item() == 1
    assert prepared.log2phy[2].item() == 2
    # Inactive entries are intentionally not cleared on the decode hot path; the
    # captured MLP only indexes current topk_ids, so this avoids a per-step fill_.
    assert prepared.log2phy[0].item() == 0
    assert prepared.mapping.active_experts == (1, 2)
    assert prepared.mapping.active_slot_ids == (1, 2)


def test_prefill_stage_plan_threads_wait_event_to_async_loader(monkeypatch):
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    calls = []
    def fake_load_many_async(
        loads,
        *,
        wait_event=None,
        record_event=True,
        validate_layout=True,
    ):
        calls.append(
            (
                [
                    (int(bundle.expert_id), int(slot.slot_id))
                    for bundle, slot in loads
                ],
                wait_event,
                bool(record_event),
                bool(validate_layout),
            )
        )
        for bundle, slot in loads:
            slot.w13.copy_(bundle.w13)
            slot.w2.copy_(bundle.w2)
            slot.state = SlotState.READY
        return "ready-wave"

    monkeypatch.setattr(runtime._transfer_engine, "load_many_async", fake_load_many_async)
    wait_event = object()
    prepared, ready_event, payload = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=True,
        wait_event=wait_event,
    )

    assert calls == [([(0, 0), (1, 1)], wait_event, True, False)]
    assert payload["buffer_index"] == 0
    assert ready_event == "ready-wave"
    assert prepared.log2phy.tolist() == [0, 1, -1, -1]


def test_prefill_route_stats_cache_consumes_matching_topk_once():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    topk_ids = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)

    runtime.cache_prefill_route_stats(
        layer_id=7,
        topk_ids=topk_ids,
        token_counts_by_expert={1: 1, 2: 2, 3: 1},
    )

    assert runtime.consume_prefill_route_stats(
        layer_id=7,
        topk_ids=topk_ids,
    ) == {1: 1, 2: 2, 3: 1}
    assert runtime.consume_prefill_route_stats(
        layer_id=7,
        topk_ids=topk_ids,
    ) is None


def test_prefill_route_stats_cache_rejects_different_storage_same_layout():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    topk_ids = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)
    same_layout = topk_ids.clone()

    runtime.cache_prefill_route_stats(
        layer_id=7,
        topk_ids=topk_ids,
        token_counts_by_expert={1: 1, 2: 2, 3: 1},
    )

    assert same_layout.data_ptr() != topk_ids.data_ptr()
    assert runtime.consume_prefill_route_stats(
        layer_id=7,
        topk_ids=same_layout,
    ) is None


def test_prefill_route_stats_cache_rejects_mutated_tensor_version():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    topk_ids = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)

    runtime.cache_prefill_route_stats(
        layer_id=7,
        topk_ids=topk_ids,
        token_counts_by_expert={1: 1, 2: 2, 3: 1},
    )
    topk_ids.copy_(torch.tensor([[0, 1], [1, 0]], dtype=torch.int32))

    assert runtime.consume_prefill_route_stats(
        layer_id=7,
        topk_ids=topk_ids,
    ) is None


def test_prefill_route_stats_cache_rejects_stale_topk_shape():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    topk_ids = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)
    reshaped = topk_ids.reshape(4, 1)

    runtime.cache_prefill_route_stats(
        layer_id=7,
        topk_ids=topk_ids,
        token_counts_by_expert={1: 1, 2: 2, 3: 1},
    )

    assert runtime.consume_prefill_route_stats(
        layer_id=7,
        topk_ids=reshaped,
    ) is None


# ---------------------------------------------------------------------------
# M3: log2phy_valid flag
# ---------------------------------------------------------------------------

def test_expert_slot_mapping_log2phy_invalid_raises_on_remap():
    """M3: remap_topk_ids must raise when the mapping was built with
    build_log2phy=False (stale persistent buffer — would silently mis-route)."""
    from vllm_moe_offload_ascend.moe_offload.slot_mapping import ExpertSlotMapping

    stale = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)
    mapping = ExpertSlotMapping(
        layer_id=7,
        active_experts=(1, 2),
        logical_to_physical=stale,
        slot_to_expert=(0, 1, None, None),
        active_slot_ids=(0, 1),
        log2phy_valid=False,
    )
    topk_ids = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="log2phy_valid=False"):
        mapping.remap_topk_ids(topk_ids)


def test_prepare_ready_slot_plan_build_log2phy_false_returns_invalid_mapping():
    """M3: prepare_ready_slot_plan(build_log2phy=False) must mark the returned
    mapping as invalid so callers cannot accidentally consume the stale buffer."""
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    # Warm the bank so experts 0 and 1 are READY.
    runtime.prepare_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        step_id=1,
    )
    prepared = runtime.prepare_ready_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        build_log2phy=False,
    )
    assert not prepared.mapping.log2phy_valid
    with pytest.raises(RuntimeError, match="log2phy_valid=False"):
        prepared.mapping.remap_topk_ids(torch.tensor([[0, 1]], dtype=torch.long))


def test_prepare_prefill_stage_plan_build_log2phy_false_returns_invalid_mapping():
    """M3: same guarantee for the stage-bank path."""
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    prepared, _, _ = runtime.prepare_prefill_stage_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=4,
        device=torch.device("cpu"),
        buffer_index=0,
        async_load=False,
        build_log2phy=False,
    )
    assert not prepared.mapping.log2phy_valid
    with pytest.raises(RuntimeError, match="log2phy_valid=False"):
        prepared.mapping.remap_topk_ids(torch.tensor([[0, 1]], dtype=torch.long))


# ---------------------------------------------------------------------------
# L2: SEW_LOG2PHY_VALIDATE env gate
# ---------------------------------------------------------------------------

def test_remap_topk_ids_cpu_raises_on_missing_expert():
    """L2: on CPU the check is always on."""
    from vllm_moe_offload_ascend.moe_offload.slot_mapping import ExpertSlotMapping

    mapping = ExpertSlotMapping(
        layer_id=7,
        active_experts=(0, 1),
        logical_to_physical=torch.tensor([-1, 0, 1, -1], dtype=torch.int32),
        slot_to_expert=(0, 1, None, None),
        active_slot_ids=(0, 1),
    )
    # Expert 2 is not staged (maps to -1).
    with pytest.raises(RuntimeError, match="experts without ready slots"):
        mapping.remap_topk_ids(torch.tensor([[2, 0]], dtype=torch.long))


def test_remap_topk_ids_env_validate_triggers_on_negative(monkeypatch):
    """L2: SEW_LOG2PHY_VALIDATE makes the check run on non-CPU devices too."""
    from vllm_moe_offload_ascend.moe_offload.slot_mapping import ExpertSlotMapping

    monkeypatch.setenv("SEW_LOG2PHY_VALIDATE", "1")

    # Fake a non-CPU tensor device by monkeypatching .device.type.
    import types
    t = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)
    logical = torch.tensor([[2, 0]], dtype=torch.long)

    # We can't create a real NPU tensor in this env, but the env-gate branch
    # is exercised by faking device.type to something other than "cpu".
    class FakeDevice:
        type = "npu"

    class FakeTensor:
        def __init__(self, real):
            self._real = real
            self.device = FakeDevice()
        def __getitem__(self, key):
            result = self._real[key]
            result.device = FakeDevice()
            return result
        def __lt__(self, other):
            return self._real < other
        def any(self):
            return self._real.any()

    mapping = ExpertSlotMapping(
        layer_id=7,
        active_experts=(0, 1),
        logical_to_physical=t,
        slot_to_expert=(0, 1, None, None),
        active_slot_ids=(0, 1),
    )
    # The standard CPU path will still catch it — confirm the message is raised.
    with pytest.raises(RuntimeError, match="experts without ready slots"):
        mapping.remap_topk_ids(logical)

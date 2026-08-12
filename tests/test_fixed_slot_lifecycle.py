import torch

from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
from vllm_moe_offload_ascend.moe_offload import runtime as runtime_mod
from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime
from vllm_moe_offload_ascend.moe_offload.slot_bank import SlotState


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_id = 7
        self.w13_weight = torch.nn.Parameter(
            torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
        )
        self.w2_weight = torch.nn.Parameter(
            torch.arange(5 * 3 * 2, dtype=torch.float32).reshape(5, 3, 2)
        )


class FakeEvent:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def test_stage_drains_prior_compute_before_reserving_sync_misses():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=4,
            async_load=False,
            graph_compatible_offload=True,
        )
    )
    runtime.register_layer_for_fixed_slots(
        TinyLayer(), slot_device=torch.device("cpu")
    )
    initial = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=5,
    )
    bank = runtime._slot_banks[7]
    completion = FakeEvent()
    for slot_id in initial.mapping.active_slot_ids:
        slot = bank.slots[slot_id]
        slot.state = SlotState.COMPUTING
        runtime._slot_compute_done_events[(7, slot.slot_id, slot.version)] = completion

    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(2, 3, 4),
        num_logical_experts=5,
    )

    assert completion.synchronize_calls == 1
    assert all(slot.state == SlotState.READY for slot in bank.slots)
    assert prepared.mapping.active_slot_ids == (2, 3, 0)
    evidence = [
        event
        for event in runtime.profiling_summary()["events"]
        if event["name"] == "slot_generation_protected_until_compute_complete"
    ]
    assert evidence[-1]["payload"]["leases_still_match"] is True
    assert evidence[-1]["payload"]["completion_events_synchronized"] == 1


def test_capture_handoff_releases_prior_compute_without_host_synchronize(monkeypatch):
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=4,
            async_load=False,
            graph_compatible_offload=True,
        )
    )
    runtime.register_layer_for_fixed_slots(
        TinyLayer(), slot_device=torch.device("cpu")
    )
    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1),
        num_logical_experts=5,
    )
    bank = runtime._slot_banks[7]
    completion = FakeEvent()
    for slot_id in prepared.mapping.active_slot_ids:
        slot = bank.slots[slot_id]
        slot.state = SlotState.COMPUTING
        runtime._slot_compute_done_events[(7, slot.slot_id, slot.version)] = completion

    monkeypatch.setattr(runtime_mod, "_is_current_graph_capturing", lambda: True)
    capture_weights = runtime.capture_safe_slot_weights(layer_id=7)
    assert capture_weights is not None
    handle = runtime.begin_slot_compute(capture_weights)

    assert handle is not None
    assert completion.synchronize_calls == 0
    assert all(bank.slots[slot_id].state == SlotState.COMPUTING for slot_id in handle.slot_ids)
    assert not runtime._slot_compute_done_events

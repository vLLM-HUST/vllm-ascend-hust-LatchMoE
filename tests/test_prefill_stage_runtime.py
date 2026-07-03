import pytest
import torch

import vllm_moe_offload_ascend.moe_offload.host_store as host_store
from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey
from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime
from vllm_moe_offload_ascend.moe_offload.slot_bank import SlotState


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
    original_load_many_sync = runtime._transfer_engine.load_many_sync

    def counted_load_many_sync(loads, **kwargs):
        load_calls.extend(
            (int(bundle.expert_id), int(slot.slot_id)) for bundle, slot in loads
        )
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
    event = runtime.profiling_summary()["events"][-1]
    assert event["name"] == "decode_fixed_slot_stage"
    assert event["layer_id"] == 7
    assert event["payload"]["active_experts"] == [3, 1]
    assert event["payload"]["hit_experts"] == [3]
    assert event["payload"]["miss_experts"] == [1]
    assert event["payload"]["n_hits"] == 1
    assert event["payload"]["n_misses"] == 1
    assert event["payload"]["h2d_bytes"] > 0


def test_slot_allocation_waits_for_loading_slots_before_eviction(monkeypatch):
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=1))
    layer = TinyLayer()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    bank = runtime._slot_banks[7]
    key0 = ExpertKey(7, 0)
    slot = bank.allocate_for(key0, step_id=1)
    slot.state = SlotState.LOADING
    sync_calls = []

    monkeypatch.setattr(
        runtime._transfer_engine,
        "synchronize",
        lambda: sync_calls.append("sync"),
    )

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

    assert calls == [([(1, 1), (2, 2)], None, True, True)]
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


def test_prefill_route_stats_cache_does_not_depend_on_data_ptr():
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
    ) == {1: 1, 2: 2, 3: 1}


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

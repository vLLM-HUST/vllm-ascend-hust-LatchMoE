import torch
import pytest

from vllm_moe_offload_ascend.moe_offload.phase_split import (
    B2DirectScatterPayload,
    build_b2_pair_microbatch,
    build_b2_pair_microbatch_from_index,
    build_b2_pair_microbatch_from_plan,
    build_b2_routed_pair_index,
    build_b2_wave_microbatch_plan,
    build_b2_wave_microbatch_plans,
    count_routed_tokens_by_expert,
    direct_scatter_add_b2_permuted_output,
    direct_scatter_add_b2_permuted_outputs,
    materialize_b2_pair_microbatches_from_plans,
    plan_b2_prefill_async_schedule,
    plan_balanced_b2_waves,
    scatter_add_b2_pair_output,
    scatter_add_b2_pair_outputs,
    simulate_b2_prefill_issue_log,
)


def test_count_routed_tokens_by_expert_counts_topk_pairs():
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.int32,
    )

    assert count_routed_tokens_by_expert(topk_ids) == {
        1: 2,
        3: 3,
        4: 2,
        7: 1,
    }


def test_plan_balanced_b2_waves_is_canonical_across_readiness_states():
    token_counts = {1: 10, 2: 9, 3: 8, 4: 1, 5: 1}
    plans = tuple(
        plan_balanced_b2_waves(
            token_counts,
            num_slots=2,
            slot_readiness=readiness,
        )
        for readiness in (
            {},
            {3: True},
            {expert_id: True for expert_id in token_counts},
        )
    )

    assert plans[0].waves == plans[1].waves == plans[2].waves
    assert plans[0].waves == ((1,), (2, 5), (3, 4))
    assert all(len(wave) <= 2 for wave in plans[0].waves)
    loads = [plans[0].wave_tokens(wave) for wave in plans[0].waves]
    assert max(loads) - min(loads) <= 8


def test_plan_balanced_b2_waves_folds_partial_hit_tail_into_miss_waves():
    readiness = {1: True}
    plan = plan_balanced_b2_waves(
        {1: 10, 2: 9, 3: 8, 4: 7},
        num_slots=4,
        slot_readiness=readiness,
    )

    hit_waves = []
    miss_waves = []
    mixed_waves = []
    for wave in plan.waves:
        hits = [expert_id for expert_id in wave if readiness.get(expert_id, False)]
        misses = [expert_id for expert_id in wave if not readiness.get(expert_id, False)]
        if hits and misses:
            mixed_waves.append(wave)
        elif hits:
            hit_waves.append(wave)
        else:
            miss_waves.append(wave)

    assert not hit_waves
    assert not miss_waves
    assert mixed_waves
    assert all(len(wave) <= 4 for wave in plan.waves)
    assert len(plan.waves) == 1


def test_plan_balanced_b2_waves_legacy_fold_flag_cannot_change_membership():
    readiness = {1: True}
    plans = tuple(
        plan_balanced_b2_waves(
            {1: 10, 2: 9, 3: 8, 4: 7},
            num_slots=4,
            slot_readiness=readiness,
            fold_partial_hits_into_miss=fold,
        )
        for fold in (False, True)
    )

    assert plans[0].waves == plans[1].waves == ((1, 2, 3, 4),)


def test_plan_balanced_b2_waves_sorts_wave_members_for_contiguous_stage_copy():
    plan = plan_balanced_b2_waves(
        {5: 100, 1: 90, 2: 80, 4: 70},
        num_slots=4,
        slot_readiness={},
    )

    assert plan.waves == ((1, 2, 4, 5),)
    assert plan.wave_tokens(plan.waves[0]) == 340


def test_plan_balanced_b2_waves_sorts_mixed_wave_canonically():
    plan = plan_balanced_b2_waves(
        {1: 10, 2: 100, 5: 90},
        num_slots=3,
        slot_readiness={1: True},
    )

    assert plan.waves == ((1, 2, 5),)


def test_mixed_wave_microbatch_uses_positional_slots_not_main_bank_ids():
    """A folded mixed wave (hits + misses) is staged into a temporary stage bank
    at positional slots 0..k-1.  build_b2_wave_microbatch_plans must therefore NOT
    apply the hit experts' main-bank slot ids to that wave, or two distinct experts
    collide on the same physical slot and read each other's weights (silent
    corruption).  Pure-hit waves keep using main-bank ids (zero-copy fast path).
    """
    active = {1: 10, 2: 9, 3: 8, 4: 7}
    readiness = {1: False, 2: False, 3: False, 4: True}
    plan = plan_balanced_b2_waves(active, num_slots=4, slot_readiness=readiness)
    # The folding produces a single mixed wave.
    assert len(plan.waves) == 1
    wave = plan.waves[0]
    assert any(readiness[e] for e in wave) and any(not readiness[e] for e in wave)

    topk_ids = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 1]], dtype=torch.int64)
    topk_weights = torch.ones(4, 2, dtype=torch.float32)
    pair_index = build_b2_routed_pair_index(topk_ids, topk_weights)

    # Expert 4 is resident in MAIN-BANK slot 0, but its position in the sorted
    # staged wave is not 0.
    ready_slot_ids = {4: 0}
    plans = build_b2_wave_microbatch_plans(
        pair_index, plan.waves, physical_slot_by_expert=ready_slot_ids
    )
    p = plans[0]

    # Every physical slot id must equal the expert's POSITIONAL index in the wave
    # (stage-bank layout), never the main-bank slot id.
    pos = {int(e): idx for idx, e in enumerate(wave)}
    logical = p.logical_expert_ids.tolist()
    physical = p.physical_slot_ids.tolist()
    expected = [pos[int(e)] for e in logical]
    assert physical == expected

    # And no two distinct experts may share a physical slot.
    slot_to_experts = {}
    for logical_id, slot in zip(logical, physical):
        slot_to_experts.setdefault(slot, set()).add(logical_id)
    assert all(len(experts) == 1 for experts in slot_to_experts.values())


def test_pure_hit_wave_microbatch_still_uses_main_bank_slot_ids():
    """Regression guard for the L4 fix: pure-hit waves must keep the zero-copy
    main-bank mapping (they execute against the main slot bank, not a stage bank).
    """
    active = {1: 5, 2: 5}
    readiness = {1: True, 2: True}
    plan = plan_balanced_b2_waves(active, num_slots=4, slot_readiness=readiness)
    wave = plan.waves[0]
    assert all(readiness[e] for e in wave)

    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int64)
    topk_weights = torch.ones(2, 2, dtype=torch.float32)
    pair_index = build_b2_routed_pair_index(topk_ids, topk_weights)

    # Resident at non-positional main-bank slots.
    ready_slot_ids = {1: 5, 2: 7}
    plans = build_b2_wave_microbatch_plans(
        pair_index, plan.waves, physical_slot_by_expert=ready_slot_ids
    )
    p = plans[0]
    mapping = dict(zip(p.logical_expert_ids.tolist(), p.physical_slot_ids.tolist()))
    assert mapping[1] == 5
    assert mapping[2] == 7


def test_plan_b2_prefill_async_schedule_primes_miss_waves_without_changing_compute_order():
    waves = ((1, 2), (3,), (4, 5), (6,))
    readiness = {
        1: True,
        2: True,
        3: True,
        4: False,
        5: False,
        6: False,
    }

    schedule = plan_b2_prefill_async_schedule(waves, slot_readiness=readiness)

    assert schedule.compute_order == (0, 1, 2, 3)
    assert schedule.hit_wave_indices == (0, 1)
    assert schedule.staged_wave_indices == (2, 3)
    assert schedule.staged_issue_order == (2, 3)
    assert schedule.prefetch_depth == 1
    assert schedule.buffer_count == 2
    assert schedule.initial_stage_count == 1
    assert schedule.to_jsonable()["staged_issue_order"] == [2, 3]


def test_plan_b2_prefill_async_schedule_clamps_prime_count_by_depth_and_buffers():
    waves = ((1,), (2,), (3,), (4,), (5,))
    readiness = {1: True, 2: False, 3: False, 4: False, 5: False}

    schedule = plan_b2_prefill_async_schedule(
        waves,
        slot_readiness=readiness,
        prefetch_depth=2,
        buffer_count=3,
    )

    assert schedule.staged_issue_order == (1, 2, 3, 4)
    assert schedule.prefetch_depth == 2
    assert schedule.buffer_count == 3
    assert schedule.initial_stage_count == 2

    log = simulate_b2_prefill_issue_log(schedule)
    assert log[0] == ("issue_stage", 1)
    assert log[1] == ("issue_stage", 2)
    assert log.index(("issue_stage", 2)) < log.index(("compute", 1))
    assert log.index(("issue_stage", 3)) < log.index(("compute", 2))


def test_plan_b2_prefill_async_schedule_disables_initial_prime_when_depth_zero():
    waves = ((1,), (2,), (3,))
    readiness = {1: True, 2: False, 3: False}

    schedule = plan_b2_prefill_async_schedule(
        waves,
        slot_readiness=readiness,
        prefetch_depth=0,
        buffer_count=3,
    )

    assert schedule.initial_stage_count == 0
    log = simulate_b2_prefill_issue_log(schedule)
    assert log[:3] == (
        ("issue_hit", 0),
        ("compute", 0),
        ("issue_stage", 1),
    )


def test_plan_b2_prefill_async_schedule_uses_transfer_estimates_when_enabled():
    waves = ((1,), (2,), (3,), (4,))
    readiness = {1: True, 2: True, 3: False, 4: False}

    schedule = plan_b2_prefill_async_schedule(
        waves,
        slot_readiness=readiness,
        h2d_bytes_by_wave={2: 100, 3: 1000},
        compute_cost_by_wave={0: 1.0, 1: 10.0, 2: 5.0, 3: 4.0},
        transfer_aware=True,
    )

    assert schedule.transfer_aware is True
    assert schedule.compute_order == (1, 0, 3, 2)
    assert schedule.staged_issue_order == (3, 2)
    assert schedule.to_jsonable()["estimated_h2d_bytes_by_wave"] == [0, 0, 100, 1000]

    log = simulate_b2_prefill_issue_log(schedule, prefetch_depth=1, buffer_count=2)
    assert log[0] == ("issue_stage", 3)
    assert log.index(("issue_stage", 3)) < log.index(("compute", 1))
    assert log.index(("issue_stage", 2)) < log.index(("compute", 3))


def test_plan_b2_prefill_async_schedule_keeps_order_when_prefetch_disabled():
    waves = ((1,), (2,), (3,))
    readiness = {1: True, 2: False, 3: False}

    schedule = plan_b2_prefill_async_schedule(
        waves,
        slot_readiness=readiness,
        prefetch_depth=0,
        h2d_bytes_by_wave={2: 1000, 1: 1},
        compute_cost_by_wave={0: 100.0, 1: 1.0, 2: 1.0},
        transfer_aware=True,
    )

    assert schedule.transfer_aware is False
    assert schedule.compute_order == (0, 1, 2)
    assert schedule.staged_issue_order == (1, 2)


def test_simulate_b2_prefill_issue_log_primes_stage_before_hit_compute():
    waves = ((1,), (2,), (3,), (4,))
    readiness = {1: True, 2: True, 3: False, 4: False}
    schedule = plan_b2_prefill_async_schedule(waves, slot_readiness=readiness)

    log = simulate_b2_prefill_issue_log(schedule, prefetch_depth=1, buffer_count=2)

    assert log[:5] == (
        ("issue_stage", 2),
        ("issue_hit", 0),
        ("compute", 0),
        ("issue_hit", 1),
        ("compute", 1),
    )
    assert log.index(("issue_stage", 3)) < log.index(("compute", 2))
    assert log[-2:] == (
        ("compute", 2),
        ("compute", 3),
    )


def test_build_b2_pair_microbatch_keeps_only_wave_pairs_and_remaps_slots():
    hidden_states = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.tensor(
        [
            [0.1, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float32,
    )
    log2phy = torch.full((8,), -1, dtype=torch.long)
    log2phy[3] = 0
    log2phy[4] = 2

    mb = build_b2_pair_microbatch(
        hidden_states,
        topk_ids,
        topk_weights,
        log2phy,
        wave_experts=(3, 4),
    )

    assert mb.num_pairs == 5
    assert mb.restore_token_indices.tolist() == [0, 1, 1, 2, 3]
    assert mb.logical_expert_ids.tolist() == [3, 3, 4, 4, 3]
    assert mb.topk_ids.squeeze(1).tolist() == [0, 0, 2, 2, 0]
    assert torch.allclose(
        mb.topk_weights.squeeze(1),
        torch.tensor([0.3, 0.4, 0.5, 0.7, 0.9]),
    )
    assert torch.equal(mb.hidden_states, hidden_states[mb.restore_token_indices])


def test_build_b2_pair_microbatch_from_index_matches_scan_builder():
    hidden_states = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.tensor(
        [
            [0.1, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float32,
    )
    log2phy = torch.full((8,), -1, dtype=torch.long)
    log2phy[3] = 0
    log2phy[4] = 2
    index = build_b2_routed_pair_index(topk_ids, topk_weights)

    scan = build_b2_pair_microbatch(
        hidden_states,
        topk_ids,
        topk_weights,
        log2phy,
        wave_experts=(3, 4),
    )
    fast = build_b2_pair_microbatch_from_index(
        hidden_states,
        index,
        log2phy,
        wave_experts=(3, 4),
    )

    assert fast.num_pairs == scan.num_pairs
    assert torch.equal(fast.hidden_states, scan.hidden_states)
    assert torch.equal(fast.topk_ids, scan.topk_ids)
    assert torch.equal(fast.topk_weights, scan.topk_weights)
    assert torch.equal(fast.restore_token_indices, scan.restore_token_indices)
    assert torch.equal(fast.logical_expert_ids, scan.logical_expert_ids)


def test_build_b2_routed_pair_index_accepts_cached_pair_offsets():
    hidden_states = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    log2phy = torch.full((8,), -1, dtype=torch.long)
    log2phy[3] = 0
    log2phy[4] = 2

    index = build_b2_routed_pair_index(
        topk_ids,
        topk_weights,
        pair_offsets_by_expert={
            3: (1, 2, 7),
            4: (3, 5),
        },
    )
    mb = build_b2_pair_microbatch_from_index(
        hidden_states,
        index,
        log2phy,
        wave_experts=(3, 4),
    )

    assert mb.restore_token_indices.tolist() == [0, 1, 1, 2, 3]
    assert mb.logical_expert_ids.tolist() == [3, 3, 4, 4, 3]
    assert mb.topk_ids.squeeze(1).tolist() == [0, 0, 2, 2, 0]


def test_build_b2_routed_pair_index_can_skip_cached_offset_validation():
    topk_ids = torch.tensor([[1, 2]], dtype=torch.long)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)

    with pytest.raises(ValueError, match="outside topk_ids flat size"):
        build_b2_routed_pair_index(
            topk_ids,
            topk_weights,
            pair_offsets_by_expert={1: (0, 99)},
        )

    index = build_b2_routed_pair_index(
        topk_ids,
        topk_weights,
        pair_offsets_by_expert={1: (0, 99)},
        validate_pair_offsets=False,
    )

    assert index.pair_offsets_by_expert[1] == (0, 99)


def test_build_b2_pair_microbatch_from_wave_plan_matches_index_builder():
    hidden_states = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.tensor(
        [
            [0.1, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float32,
    )
    log2phy = torch.full((8,), -1, dtype=torch.long)
    log2phy[3] = 0
    log2phy[4] = 2
    index = build_b2_routed_pair_index(topk_ids, topk_weights)
    plan = build_b2_wave_microbatch_plan(index, wave_experts=(3, 4))

    from_index = build_b2_pair_microbatch_from_index(
        hidden_states,
        index,
        log2phy,
        wave_experts=(3, 4),
    )
    from_plan = build_b2_pair_microbatch_from_plan(
        hidden_states,
        topk_weights,
        log2phy,
        plan,
    )

    assert plan.num_pairs == 5
    assert torch.equal(plan.token_indices, torch.tensor([0, 1, 1, 2, 3]))
    assert torch.equal(plan.physical_slot_ids, torch.tensor([0, 0, 1, 1, 0]))
    assert torch.equal(from_plan.hidden_states, from_index.hidden_states)
    assert torch.equal(from_plan.topk_ids, from_index.topk_ids)
    assert torch.equal(from_plan.topk_weights, from_index.topk_weights)
    assert torch.equal(from_plan.restore_token_indices, from_index.restore_token_indices)
    assert torch.equal(from_plan.logical_expert_ids, from_index.logical_expert_ids)


def test_build_b2_pair_microbatch_from_wave_plan_can_skip_log2phy():
    hidden_states = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.tensor(
        [
            [0.1, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float32,
    )
    log2phy = torch.full((8,), -1, dtype=torch.long)
    log2phy[3] = 0
    log2phy[4] = 1
    index = build_b2_routed_pair_index(topk_ids, topk_weights)
    plan = build_b2_wave_microbatch_plan(index, wave_experts=(3, 4))

    from_log2phy = build_b2_pair_microbatch_from_plan(
        hidden_states,
        topk_weights,
        log2phy,
        plan,
    )
    from_slots = build_b2_pair_microbatch_from_plan(
        hidden_states,
        topk_weights,
        None,
        plan,
    )

    assert torch.equal(from_slots.topk_ids, from_log2phy.topk_ids)
    assert torch.equal(from_slots.topk_ids.squeeze(1), torch.tensor([0, 0, 1, 1, 0]))
    assert torch.equal(from_slots.hidden_states, from_log2phy.hidden_states)
    assert torch.equal(from_slots.topk_weights, from_log2phy.topk_weights)
    assert torch.equal(from_slots.restore_token_indices, from_log2phy.restore_token_indices)
    assert torch.equal(from_slots.logical_expert_ids, from_log2phy.logical_expert_ids)


def test_build_b2_wave_microbatch_plan_can_use_main_slot_ids_for_hit_wave():
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    index = build_b2_routed_pair_index(topk_ids, topk_weights)

    plan = build_b2_wave_microbatch_plan(
        index,
        wave_experts=(3, 4),
        physical_slot_by_expert={3: 5, 4: 2},
    )

    assert plan.num_pairs == 5
    assert torch.equal(plan.physical_slot_ids, torch.tensor([5, 5, 2, 2, 5]))


def test_build_b2_wave_microbatch_plan_handles_empty_wave():
    hidden_states = torch.randn(2, 4)
    topk_ids = torch.tensor([[1, 2], [3, 1]], dtype=torch.long)
    topk_weights = torch.ones(2, 2)
    log2phy = torch.full((4,), -1, dtype=torch.long)
    index = build_b2_routed_pair_index(topk_ids, topk_weights)
    plan = build_b2_wave_microbatch_plan(index, wave_experts=(0,))

    mb = build_b2_pair_microbatch_from_plan(
        hidden_states,
        topk_weights,
        log2phy,
        plan,
    )

    assert plan.num_pairs == 0
    assert mb.num_pairs == 0
    assert mb.hidden_states.shape == (0, 4)


def test_build_b2_wave_microbatch_plans_match_single_wave_builder():
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    index = build_b2_routed_pair_index(topk_ids, topk_weights)
    waves = ((3,), (0,), (4, 7), (1,))

    batched = build_b2_wave_microbatch_plans(index, waves)

    assert len(batched) == len(waves)
    assert all(plan.layer_token_indices is batched[0].layer_token_indices for plan in batched)
    assert all(plan.layer_topk_positions is batched[0].layer_topk_positions for plan in batched)
    assert all(plan.layer_logical_expert_ids is batched[0].layer_logical_expert_ids for plan in batched)
    assert all(plan.layer_physical_slot_ids is batched[0].layer_physical_slot_ids for plan in batched)
    assert [plan.layer_pair_start for plan in batched] == [0, 3, 3, 6]
    assert [plan.layer_pair_end for plan in batched] == [3, 3, 6, 8]
    for wave, batch_plan in zip(waves, batched, strict=True):
        single_plan = build_b2_wave_microbatch_plan(index, wave)
        assert batch_plan.wave_experts == single_plan.wave_experts
        assert torch.equal(batch_plan.pair_offsets, single_plan.pair_offsets)
        assert torch.equal(batch_plan.token_indices, single_plan.token_indices)
        assert torch.equal(batch_plan.topk_positions, single_plan.topk_positions)
        assert torch.equal(batch_plan.logical_expert_ids, single_plan.logical_expert_ids)
        assert torch.equal(batch_plan.physical_slot_ids, single_plan.physical_slot_ids)


def test_build_b2_wave_microbatch_plans_can_skip_global_pair_order():
    topk_ids = torch.tensor(
        [
            [1, 2],
            [2, 1],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    index = build_b2_routed_pair_index(topk_ids, topk_weights)

    (plan,) = build_b2_wave_microbatch_plans(
        index,
        ((2, 1),),
        preserve_pair_order=False,
    )

    assert plan.pair_offsets.tolist() == [1, 2, 0, 3]
    assert plan.logical_expert_ids.tolist() == [2, 2, 1, 1]
    assert plan.physical_slot_ids.tolist() == [0, 0, 1, 1]


def test_build_b2_wave_microbatch_plans_reuses_flat_pair_offsets():
    topk_ids = torch.tensor(
        [
            [1, 2],
            [3, 1],
            [2, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    index = build_b2_routed_pair_index(topk_ids, topk_weights)

    plans = build_b2_wave_microbatch_plans(
        index,
        ((2,), (3, 1)),
        preserve_pair_order=False,
    )

    assert plans[0].pair_offsets.tolist() == [1, 4]
    assert plans[1].pair_offsets.tolist() == [2, 5, 0, 3]
    assert plans[0].layer_flat_pair_offsets is plans[1].layer_flat_pair_offsets
    assert torch.equal(
        plans[0].layer_flat_pair_offsets,
        torch.tensor([1, 4, 2, 5, 0, 3]),
    )


def test_build_b2_wave_microbatch_plans_can_skip_input_normalization():
    topk_ids = torch.tensor(
        [
            [1, 2],
            [3, 1],
            [2, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    index = build_b2_routed_pair_index(topk_ids, topk_weights)

    default = build_b2_wave_microbatch_plans(
        index,
        (("2",), ("3", "1")),
        physical_slot_by_expert={"1": "5"},
        preserve_pair_order=False,
    )
    fast = build_b2_wave_microbatch_plans(
        index,
        ((2,), (3, 1)),
        physical_slot_by_expert={1: 5},
        preserve_pair_order=False,
        normalize_inputs=False,
    )

    assert [plan.wave_experts for plan in default] == [plan.wave_experts for plan in fast]
    assert [plan.pair_offsets.tolist() for plan in default] == [
        plan.pair_offsets.tolist() for plan in fast
    ]
    assert [plan.physical_slot_ids.tolist() for plan in default] == [
        plan.physical_slot_ids.tolist() for plan in fast
    ]


def test_build_b2_wave_microbatch_plans_can_omit_topk_positions():
    hidden_states = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    topk_ids = torch.tensor(
        [
            [1, 2],
            [3, 1],
            [2, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=torch.float32,
    )
    index = build_b2_routed_pair_index(topk_ids, topk_weights)

    plans = build_b2_wave_microbatch_plans(
        index,
        ((2,), (3, 1)),
        preserve_pair_order=False,
        include_topk_positions=False,
    )
    microbatches = materialize_b2_pair_microbatches_from_plans(
        hidden_states,
        topk_weights,
        plans,
    )

    assert all(plan.topk_positions.numel() == 0 for plan in plans)
    assert all(plan.layer_topk_positions.numel() == 0 for plan in plans)
    assert [plan.pair_offsets.tolist() for plan in plans] == [[1, 4], [2, 5, 0, 3]]
    assert microbatches[0].topk_weights.squeeze(1).tolist() == pytest.approx([0.2, 0.5])
    assert microbatches[1].topk_weights.squeeze(1).tolist() == pytest.approx(
        [0.3, 0.6, 0.1, 0.4]
    )


def test_build_b2_wave_microbatch_plans_can_omit_logical_ids_with_slot_ids():
    hidden_states = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    index = build_b2_routed_pair_index(topk_ids, topk_weights)

    (plan,) = build_b2_wave_microbatch_plans(
        index,
        ((2, 1),),
        preserve_pair_order=False,
        include_logical_expert_ids=False,
    )
    assert plan.logical_expert_ids.numel() == 0

    mb = build_b2_pair_microbatch_from_plan(
        hidden_states,
        topk_weights,
        None,
        plan,
    )
    assert mb.topk_ids.squeeze(1).tolist() == [0, 0, 1, 1]
    assert mb.logical_expert_ids.numel() == 0

    with pytest.raises(ValueError, match="omitted logical expert ids"):
        build_b2_pair_microbatch_from_plan(
            hidden_states,
            topk_weights,
            torch.arange(3, dtype=torch.long),
            plan,
        )


def test_materialize_b2_pair_microbatches_from_plans_matches_per_wave_builder():
    hidden_states = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    topk_ids = torch.tensor(
        [
            [1, 3],
            [3, 4],
            [1, 4],
            [7, 3],
        ],
        dtype=torch.long,
    )
    topk_weights = torch.tensor(
        [
            [0.1, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float32,
    )
    index = build_b2_routed_pair_index(topk_ids, topk_weights)
    plans = build_b2_wave_microbatch_plans(
        index,
        ((3,), (4, 7), (1,)),
        physical_slot_by_expert={3: 5},
    )

    materialized = materialize_b2_pair_microbatches_from_plans(
        hidden_states,
        topk_weights,
        plans,
    )

    assert len(materialized) == len(plans)
    for plan, mb in zip(plans, materialized, strict=True):
        expected = build_b2_pair_microbatch_from_plan(
            hidden_states,
            topk_weights,
            None,
            plan,
        )
        assert mb.num_pairs == expected.num_pairs
        assert torch.equal(mb.hidden_states, expected.hidden_states)
        assert torch.equal(mb.topk_ids, expected.topk_ids)
        assert torch.equal(mb.topk_weights, expected.topk_weights)
        assert torch.equal(mb.restore_token_indices, expected.restore_token_indices)
        assert torch.equal(mb.logical_expert_ids, expected.logical_expert_ids)


def test_build_b2_wave_microbatch_plans_all_empty_waves():
    topk_ids = torch.tensor([[1, 2], [3, 1]], dtype=torch.long)
    topk_weights = torch.ones(2, 2)
    index = build_b2_routed_pair_index(topk_ids, topk_weights)

    plans = build_b2_wave_microbatch_plans(index, ((0,), (4, 5)))

    assert len(plans) == 2
    assert all(plan.num_pairs == 0 for plan in plans)
    assert plans[0].wave_experts == (0,)
    assert plans[1].wave_experts == (4, 5)


def test_scatter_add_b2_pair_output_accumulates_duplicate_tokens():
    full = torch.zeros(4, 2)
    pair_output = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )
    restore = torch.tensor([0, 1, 1, 3], dtype=torch.long)

    out = scatter_add_b2_pair_output(full, pair_output, restore)

    assert out is full
    assert torch.equal(
        full,
        torch.tensor(
            [
                [1.0, 1.0],
                [5.0, 5.0],
                [0.0, 0.0],
                [4.0, 4.0],
            ]
        ),
    )


def test_scatter_add_b2_pair_outputs_matches_per_wave_scatter():
    pair_outputs = (
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]),
        torch.empty(0, 2),
    )
    restore_indices = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1, 3, 0], dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    sequential = torch.zeros(4, 2)
    batched = torch.zeros(4, 2)

    for output, indices in zip(pair_outputs, restore_indices, strict=True):
        scatter_add_b2_pair_output(sequential, output, indices)
    scatter_add_b2_pair_outputs(batched, pair_outputs, restore_indices)

    assert torch.equal(batched, sequential)
    assert torch.equal(
        batched,
        torch.tensor(
            [
                [6.0, 6.0],
                [5.0, 5.0],
                [0.0, 0.0],
                [4.0, 4.0],
            ]
        ),
    )


def test_direct_scatter_add_b2_permuted_output_matches_combine_then_scatter():
    full_old = torch.zeros(4, 2)
    full_direct = torch.zeros(4, 2)
    permuted = torch.tensor(
        [
            [4.0, 40.0],
            [1.0, 10.0],
            [5.0, 50.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )
    # AllGather token_combine uses abs(expanded_row_idx) as an unpermute index:
    # microbatch row i gathers permuted row abs(expanded_row_idx)[i].
    expanded = torch.tensor([3, 0, 4, 1, 2], dtype=torch.int32)
    weights = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
    restore = torch.tensor([0, 1, 1, 3, 0], dtype=torch.long)

    expanded_abs = expanded.abs().to(torch.long)
    wave_local = permuted.index_select(0, expanded_abs) * weights
    scatter_add_b2_pair_output(full_old, wave_local, restore)

    direct_scatter_add_b2_permuted_output(
        full_direct,
        permuted_tokens=permuted,
        expanded_row_idx=expanded,
        topk_weights=weights,
        restore_token_indices=restore,
    )

    assert torch.allclose(full_direct, full_old)


def test_direct_scatter_add_b2_permuted_outputs_batches_waves():
    payloads = (
        B2DirectScatterPayload(
            permuted_tokens=torch.tensor([[3.0, 30.0], [1.0, 10.0], [5.0, 50.0]]),
            expanded_row_idx=torch.tensor([2, 0, 1], dtype=torch.int32),
            topk_weights=torch.tensor([[0.5], [0.7], [0.9]]),
            restore_token_indices=torch.tensor([0, 1, 0], dtype=torch.long),
        ),
        B2DirectScatterPayload(
            permuted_tokens=torch.tensor([[4.0, 40.0], [2.0, 20.0]]),
            expanded_row_idx=torch.tensor([1, 0], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25], [0.75]]),
            restore_token_indices=torch.tensor([2, 1], dtype=torch.long),
        ),
    )
    sequential = torch.zeros(3, 2)
    batched = torch.zeros(3, 2)

    for payload in payloads:
        direct_scatter_add_b2_permuted_output(
            sequential,
            permuted_tokens=payload.permuted_tokens,
            expanded_row_idx=payload.expanded_row_idx,
            topk_weights=payload.topk_weights,
            restore_token_indices=payload.restore_token_indices,
        )
    direct_scatter_add_b2_permuted_outputs(batched, payloads)

    assert torch.allclose(batched, sequential)
    assert torch.allclose(
        batched,
        torch.tensor(
            [
                [3.4, 34.0],
                [5.1, 51.0],
                [0.5, 5.0],
            ]
        ),
    )


def test_direct_scatter_add_b2_permuted_outputs_offsets_wave_local_indices():
    payloads = (
        B2DirectScatterPayload(
            permuted_tokens=torch.tensor([[1.0], [2.0]]),
            expanded_row_idx=torch.tensor([1, 0], dtype=torch.int32),
            topk_weights=torch.tensor([[10.0], [20.0]]),
            restore_token_indices=torch.tensor([0, 1], dtype=torch.long),
        ),
        B2DirectScatterPayload(
            permuted_tokens=torch.tensor([[100.0], [200.0]]),
            expanded_row_idx=torch.tensor([1, 0], dtype=torch.int32),
            topk_weights=torch.tensor([[0.1], [0.2]]),
            restore_token_indices=torch.tensor([0, 1], dtype=torch.long),
        ),
    )

    full = torch.zeros(2, 1)
    direct_scatter_add_b2_permuted_outputs(full, payloads)

    assert torch.allclose(full, torch.tensor([[40.0], [40.0]]))


def test_direct_scatter_uses_canonical_wave_order_and_fp32_accumulator():
    payloads = tuple(
        B2DirectScatterPayload(
            permuted_tokens=torch.tensor([[value]], dtype=torch.float32),
            expanded_row_idx=torch.tensor([0], dtype=torch.int32),
            topk_weights=torch.tensor([[1.0]], dtype=torch.float32),
            restore_token_indices=torch.tensor([0], dtype=torch.long),
            wave_index=wave_index,
        )
        for wave_index, value in ((2, 1.0), (1, -1.0e8), (0, 1.0e8))
    )
    output = torch.zeros(1, 1, dtype=torch.float32)

    direct_scatter_add_b2_permuted_outputs(output, payloads)

    assert output.dtype == torch.float32
    assert output.item() == 1.0


def test_direct_scatter_promotes_bfloat16_payload_to_accumulator_dtype():
    output = torch.zeros(1, 1, dtype=torch.float32)
    direct_scatter_add_b2_permuted_output(
        output,
        permuted_tokens=torch.tensor([[2.0]], dtype=torch.bfloat16),
        expanded_row_idx=torch.tensor([0], dtype=torch.int32),
        topk_weights=torch.tensor([[0.5]], dtype=torch.bfloat16),
        restore_token_indices=torch.tensor([0], dtype=torch.long),
    )

    assert output.dtype == torch.float32
    assert output.item() == 1.0


def test_build_b2_pair_microbatch_can_skip_hot_path_unstaged_validation():
    hidden_states = torch.randn(2, 4)
    topk_ids = torch.tensor([[1, 2], [3, 1]], dtype=torch.long)
    topk_weights = torch.ones(2, 2)
    log2phy = torch.full((4,), -1, dtype=torch.long)
    log2phy[2] = 0

    mb = build_b2_pair_microbatch(
        hidden_states,
        topk_ids,
        topk_weights,
        log2phy,
        wave_experts=(1, 2),
    )

    assert mb.num_pairs == 3
    assert (mb.topk_ids < 0).any()


def test_build_b2_pair_microbatch_rejects_unstaged_wave_expert_when_validating(monkeypatch):
    monkeypatch.setenv("SEW_B2_VALIDATE", "1")
    hidden_states = torch.randn(2, 4)
    topk_ids = torch.tensor([[1, 2], [3, 1]], dtype=torch.long)
    topk_weights = torch.ones(2, 2)
    log2phy = torch.full((4,), -1, dtype=torch.long)
    log2phy[2] = 0

    with pytest.raises(RuntimeError, match="unstaged experts"):
        build_b2_pair_microbatch(
            hidden_states,
            topk_ids,
            topk_weights,
            log2phy,
            wave_experts=(1, 2),
        )


# ---------------------------------------------------------------------------
# L1: slice_positions — sparse non-identity active expert set
# ---------------------------------------------------------------------------

def test_plan_hit_miss_phases_sets_slice_positions_for_sparse_active_set():
    """L1: expert_indices holds logical ids; slice_positions must hold compact
    positions so _compute_wave indexes group_list/weights correctly even when
    active experts are not the dense identity (0,1,2,...)."""
    from vllm_moe_offload_ascend.moe_offload.phase_split import plan_hit_miss_phases

    # Active experts: sparse logical ids 3, 7, 12 at compact positions 0,1,2.
    active_expert_ids = (3, 7, 12)
    expert_slices = [(0, 5), (5, 9), (9, 11)]
    readiness = {3: True, 7: False, 12: False}

    plan = plan_hit_miss_phases(
        expert_slices=expert_slices,
        active_expert_ids=active_expert_ids,
        slot_readiness=readiness,
        max_phases=2,
    )

    hit_phase = next(p for p in plan.phases if p.is_hit)
    miss_phase = next(p for p in plan.phases if not p.is_hit)

    # Logical ids must remain logical.
    assert hit_phase.expert_indices == (3,)
    assert set(miss_phase.expert_indices) == {7, 12}

    # slice_positions must be compact positions (0-based into active set).
    assert hit_phase.slice_positions == (0,)        # expert 3 is at position 0
    assert set(miss_phase.slice_positions) == {1, 2}  # experts 7,12 at positions 1,2

    # resolved_slice_positions must equal slice_positions when set.
    assert hit_phase.resolved_slice_positions() == hit_phase.slice_positions
    assert miss_phase.resolved_slice_positions() == miss_phase.slice_positions


def test_plan_capacity_bounded_phases_sets_slice_positions():
    """L1: capacity-bounded planner must also set slice_positions correctly."""
    from vllm_moe_offload_ascend.moe_offload.phase_split import plan_capacity_bounded_phases

    active_expert_ids = (5, 10, 15, 20)
    expert_slices = [(0, 3), (3, 7), (7, 9), (9, 13)]

    plan = plan_capacity_bounded_phases(
        expert_slices=expert_slices,
        active_expert_ids=active_expert_ids,
        num_slots=2,
    )

    assert len(plan.phases) == 2
    wave0, wave1 = plan.phases[0], plan.phases[1]

    # Wave 0 covers positions 0,1; wave 1 covers positions 2,3.
    assert wave0.slice_positions == (0, 1)
    assert wave1.slice_positions == (2, 3)

    # Logical ids must be the actual expert ids.
    assert wave0.expert_indices == (5, 10)
    assert wave1.expert_indices == (15, 20)


def test_device_wave_microbatch_planner_matches_exact_pair_coverage():
    from vllm_moe_offload_ascend.moe_offload.phase_split import (
        build_b2_device_wave_microbatch_plans,
        materialize_b2_pair_microbatches_from_plans,
    )

    topk_ids = torch.tensor([[3, 1], [2, 3], [0, 2]], dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]],
        dtype=torch.float32,
    )
    waves = ((0, 3), (1, 2))
    plans = build_b2_device_wave_microbatch_plans(
        topk_ids,
        waves,
        num_logical_experts=4,
        active_experts=(0, 1, 2, 3),
        wave_pair_counts=(3, 3),
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    microbatches = materialize_b2_pair_microbatches_from_plans(
        hidden,
        topk_weights,
        plans,
    )

    all_offsets = torch.cat(tuple(plan.pair_offsets for plan in plans))
    assert all_offsets.sort().values.tolist() == list(range(6))
    assert plans[0].physical_slot_ids.tolist() == [1, 1, 0]
    assert plans[1].physical_slot_ids.tolist() == [0, 1, 1]
    assert sum(batch.num_pairs for batch in microbatches) == 6
    restored = torch.cat(
        tuple(batch.restore_token_indices for batch in microbatches)
    )
    assert restored.sort().values.tolist() == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize(
    ("active_experts", "expected_wave_count"),
    (
        (tuple(range(128)), 4),
        (tuple(range(127)), 4),
        (tuple(expert for expert in range(128) if expert != 126), 4),
        ((0, 127), 1),
    ),
)
def test_device_wave_planner_covers_expert_127_boundaries(
    active_experts,
    expected_wave_count,
):
    from vllm_moe_offload_ascend.moe_offload.phase_split import (
        build_b2_device_wave_microbatch_plans,
    )

    waves = tuple(
        tuple(active_experts[start : start + 32])
        for start in range(0, len(active_experts), 32)
    )
    topk_ids = torch.tensor(active_experts, dtype=torch.long).reshape(-1, 1)
    plans = build_b2_device_wave_microbatch_plans(
        topk_ids,
        waves,
        num_logical_experts=128,
        active_experts=active_experts,
        wave_pair_counts=tuple(len(wave) for wave in waves),
    )

    assert len(plans) == expected_wave_count
    offsets = torch.cat(tuple(plan.pair_offsets for plan in plans))
    assert offsets.sort().values.tolist() == list(range(len(active_experts)))


def test_device_wave_planner_rejects_wave_union_mismatch_before_gather():
    from vllm_moe_offload_ascend.moe_offload.phase_split import (
        build_b2_device_wave_microbatch_plans,
    )

    with pytest.raises(ValueError, match="waves do not match"):
        build_b2_device_wave_microbatch_plans(
            torch.tensor([[0], [127]], dtype=torch.long),
            ((0,),),
            num_logical_experts=128,
            active_experts=(0, 127),
            wave_pair_counts=(2,),
        )


def test_device_wave_planner_rejects_invalid_topk_without_index_error():
    from vllm_moe_offload_ascend.moe_offload.phase_split import (
        build_b2_device_wave_microbatch_plans,
    )

    with pytest.raises(ValueError, match="invalid or uncovered top-k IDs"):
        build_b2_device_wave_microbatch_plans(
            torch.tensor([[0], [128]], dtype=torch.long),
            ((0,),),
            num_logical_experts=128,
            active_experts=(0,),
            wave_pair_counts=None,
        )


def test_device_wave_planner_rejects_stale_per_wave_pair_counts():
    from vllm_moe_offload_ascend.moe_offload.phase_split import (
        build_b2_device_wave_microbatch_plans,
    )

    with pytest.raises(ValueError, match="do not match current top-k contents"):
        build_b2_device_wave_microbatch_plans(
            torch.tensor([[0], [0], [1]], dtype=torch.long),
            ((0,), (1,)),
            num_logical_experts=2,
            active_experts=(0, 1),
            wave_pair_counts=(1, 2),
        )


def test_device_scatter_descriptor_preserves_exact_scatter_result():
    from vllm_moe_offload_ascend.moe_offload.phase_split import (
        B2DirectScatterPayload,
        build_b2_device_scatter_descriptor,
        direct_scatter_add_b2_permuted_outputs,
    )

    permuted = torch.tensor([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
    expanded = torch.tensor([2, 0, 1], dtype=torch.int32)
    weights = torch.tensor([[0.5], [0.25], [1.0]])
    restore = torch.tensor([0, 1, 0], dtype=torch.long)
    descriptor = build_b2_device_scatter_descriptor(
        expanded_row_idx=expanded,
        topk_weights=weights,
        restore_token_indices=restore,
        output_dtype=permuted.dtype,
        output_device=permuted.device,
    )
    output = torch.zeros(2, 2)

    direct_scatter_add_b2_permuted_outputs(
        output,
        (
            B2DirectScatterPayload(
                permuted_tokens=permuted,
                expanded_row_idx=expanded,
                topk_weights=weights,
                restore_token_indices=restore,
                device_descriptor=descriptor,
            ),
        ),
    )

    expected = torch.stack(
        (permuted[2] * 0.5 + permuted[1], permuted[0] * 0.25)
    )
    assert torch.allclose(output, expected)

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_moe_offload_ascend.patches import patch_fused_moe


class _FakeCausalConvOp:
    def __init__(self, schema: str):
        self.default = SimpleNamespace(_schema=schema)
        self.calls = []

    def __call__(self, output, x, weight, *args, **kwargs):
        self.calls.append((output, x, weight, args, kwargs))
        return output


def _install_fake_op(monkeypatch, schema: str) -> _FakeCausalConvOp:
    fake_op = _FakeCausalConvOp(schema)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        patch_fused_moe._GDN_CAUSAL_CONV1D_OP_NAME,
        fake_op,
        raising=False,
    )
    return fake_op


def test_gdn_custom_op_vendor_bootstrap_uses_extension_location(monkeypatch, tmp_path, capsys):
    extension = tmp_path / "installed" / "vllm_ascend" / "vllm_ascend_C.so"
    extension.parent.mkdir(parents=True)
    extension.touch()
    vendor = extension.parent / patch_fused_moe._GDN_CUSTOM_OP_VENDOR_RELATIVE_PATH
    library = vendor / patch_fused_moe._GDN_CUSTOM_OP_VENDOR_LIBRARY
    library.parent.mkdir(parents=True)
    library.touch()

    monkeypatch.delenv(patch_fused_moe._GDN_CUSTOM_OP_VENDOR_ENV, raising=False)
    monkeypatch.setattr(
        patch_fused_moe.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(extension)),
    )

    assert patch_fused_moe._bootstrap_gdn_custom_op_vendor_env()
    assert (
        os.environ[patch_fused_moe._GDN_CUSTOM_OP_VENDOR_ENV]
        == str(vendor)
    )
    assert "gdn_custom_opp=extension_vendor" in capsys.readouterr().out


def test_gdn_custom_op_vendor_bootstrap_is_idempotent(monkeypatch, tmp_path):
    extension = tmp_path / "installed" / "vllm_ascend" / "vllm_ascend_C.so"
    extension.parent.mkdir(parents=True)
    extension.touch()
    vendor = extension.parent / patch_fused_moe._GDN_CUSTOM_OP_VENDOR_RELATIVE_PATH
    library = vendor / patch_fused_moe._GDN_CUSTOM_OP_VENDOR_LIBRARY
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setenv(patch_fused_moe._GDN_CUSTOM_OP_VENDOR_ENV, str(vendor))
    monkeypatch.setattr(
        patch_fused_moe.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(extension)),
    )

    assert not patch_fused_moe._bootstrap_gdn_custom_op_vendor_env()


def test_gdn_tensor_abi_adapter_converts_legacy_kwargs(monkeypatch):
    fake_op = _install_fake_op(
        monkeypatch,
        "_C_ascend::npu_causal_conv1d_custom("
        "Tensor output, Tensor x, Tensor weight, Tensor conv_state, "
        "Tensor? bias_opt, Tensor? query_start_loc_opt, "
        "Tensor? cache_indices_opt, Tensor? initial_state_mode_opt, "
        "Tensor? num_accepted_tokens_opt, int activation_mode, "
        "int pad_slot_id, int run_mode) -> Tensor output",
    )

    assert patch_fused_moe._patch_gdn_causal_conv1d_tensor_abi()
    output = torch.empty((1, 2))
    x = torch.empty((1, 2))
    weight = torch.empty((2, 2))
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state=torch.empty((1, 2)),
        bias_opt=None,
        query_start_loc_opt=(0, 1),
        cache_indices_opt=[7],
        initial_state_mode_opt=(),
        num_accepted_tokens_opt=[],
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=0,
    )

    assert len(fake_op.calls) == 1
    kwargs = fake_op.calls[0][4]
    assert torch.equal(kwargs["query_start_loc_opt"], torch.tensor([0, 1], dtype=torch.int64))
    assert torch.equal(kwargs["cache_indices_opt"], torch.tensor([7], dtype=torch.int64))
    assert kwargs["query_start_loc_opt"].device == x.device
    assert kwargs["initial_state_mode_opt"] is None
    assert kwargs["num_accepted_tokens_opt"] is None


def test_gdn_tensor_abi_adapter_converts_legacy_positional_arguments(monkeypatch):
    fake_op = _install_fake_op(
        monkeypatch,
        "Tensor output, Tensor x, Tensor weight, Tensor conv_state, Tensor? bias_opt, "
        "Tensor? query_start_loc_opt, Tensor? cache_indices_opt, "
        "Tensor? initial_state_mode_opt, Tensor? num_accepted_tokens_opt, "
        "int activation_mode, int pad_slot_id, int run_mode",
    )

    assert patch_fused_moe._patch_gdn_causal_conv1d_tensor_abi()
    output = torch.empty((1, 2))
    x = torch.empty((1, 2))
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        torch.empty((2, 2)),
        torch.empty((1, 2)),
        None,
        (0, 1),
        (9,),
        (),
        (1,),
        1,
        -1,
        1,
    )

    positional = fake_op.calls[0][3]
    assert torch.equal(positional[2], torch.tensor([0, 1], dtype=torch.int64))
    assert torch.equal(positional[3], torch.tensor([9], dtype=torch.int64))
    assert positional[4] is None
    assert torch.equal(positional[5], torch.tensor([1], dtype=torch.int64))


def test_gdn_tensor_abi_adapter_is_noop_for_legacy_int_array_schema(monkeypatch):
    fake_op = _install_fake_op(
        monkeypatch,
        "Tensor output, Tensor x, Tensor weight, Tensor conv_state, Tensor? bias_opt, "
        "int[] query_start_loc_opt, int[] cache_indices_opt, "
        "int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, "
        "int activation_mode, int pad_slot_id, int run_mode",
    )

    assert not patch_fused_moe._patch_gdn_causal_conv1d_tensor_abi()
    assert torch.ops._C_ascend.npu_causal_conv1d_custom is fake_op


def test_gdn_tensor_abi_adapter_preserves_matching_tensor_identity(monkeypatch):
    fake_op = _install_fake_op(
        monkeypatch,
        "Tensor output, Tensor x, Tensor weight, Tensor conv_state, Tensor? bias_opt, "
        "Tensor? query_start_loc_opt, Tensor? cache_indices_opt, "
        "Tensor? initial_state_mode_opt, Tensor? num_accepted_tokens_opt, "
        "int activation_mode, int pad_slot_id, int run_mode",
    )

    assert patch_fused_moe._patch_gdn_causal_conv1d_tensor_abi()
    output = torch.empty((1, 2))
    x = torch.empty((1, 2))
    query_start_loc = torch.tensor([0, 1], dtype=torch.int64)
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        torch.empty((2, 2)),
        conv_state=torch.empty((1, 2)),
        bias_opt=None,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=torch.tensor([3], dtype=torch.int64),
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=None,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )

    assert fake_op.calls[0][4]["query_start_loc_opt"] is query_start_loc


def test_gdn_pytorch_fallback_preserves_prefill_state(monkeypatch):
    fake_namespace = SimpleNamespace()
    fake_torch = SimpleNamespace(ops=SimpleNamespace(_C_ascend=fake_namespace))
    monkeypatch.setenv(patch_fused_moe._GDN_PYTORCH_FALLBACK_ENV, "1")

    assert patch_fused_moe._patch_gdn_causal_conv1d_pytorch_fallback(fake_torch)
    fallback = getattr(
        fake_namespace,
        patch_fused_moe._GDN_CAUSAL_CONV1D_OP_NAME,
    )

    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    weight = torch.ones(3, 4)
    conv_state = torch.zeros(2, 2, 4)
    output = torch.empty_like(x)
    fallback(
        output,
        x,
        weight,
        conv_state=conv_state,
        query_start_loc_opt=(0, 2),
        cache_indices_opt=(1,),
        initial_state_mode_opt=(0,),
        activation_mode=0,
        run_mode=0,
    )

    assert torch.equal(output, torch.tensor([[0, 1, 2, 3], [4, 6, 8, 10]], dtype=torch.float32))
    assert torch.equal(conv_state[1], x)


def test_gdn_pytorch_fallback_rejects_speculative_state(monkeypatch):
    monkeypatch.setenv(patch_fused_moe._GDN_PYTORCH_FALLBACK_ENV, "1")
    with pytest.raises(RuntimeError, match="speculative decoding"):
        patch_fused_moe._gdn_causal_conv1d_pytorch_fallback(
            torch.empty(1, 4),
            torch.empty(1, 4),
            torch.empty(3, 4),
            conv_state=torch.empty(1, 2, 4),
            num_accepted_tokens_opt=(1,),
        )


def test_gdn_fused_gating_fallback_matches_reference_shapes():
    A_log = torch.tensor([-1.0, 0.0])
    a = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    b = torch.tensor([[0.0, 1.0], [-1.0, 2.0]])
    dt_bias = torch.tensor([0.5, -0.5])

    g, beta_output = patch_fused_moe._gdn_fused_gating_pytorch_fallback(
        A_log, a, b, dt_bias
    )

    assert g.shape == (1, 2, 2)
    assert beta_output.shape == (1, 2, 2)
    assert torch.all((beta_output >= 0) & (beta_output <= 1))

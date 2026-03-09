import argparse

import torch
import torch_npu  # noqa: F401
import vllm  # noqa: F401

import vllm_ascend.platform  # noqa: F401
from vllm_ascend.ops.triton.mamba.causal_conv1d import (
    PAD_SLOT_ID,
    _pick_causal_conv1d_update_fast_path_launch_params,
    _pick_causal_conv1d_update_sla_launch_params,
    _select_causal_conv1d_update_fast_path,
    _select_causal_conv1d_update_sla_fast_path,
    causal_conv1d_update_npu,
)


HARD_CASES = [
    "decode_contig",
    "decode_state_stride",
    "mtp1_contig_all1",
    "mtp1_contig_alt12",
    "mtp1_state_stride_all1",
    "mtp1_state_stride_alt12",
]

SOFT_CASES = [
    "decode_contig_4096",
    "mtp1_contig_all1_4096",
    "mtp3_contig_alt1234_2048",
]


def _make_cache_line_stride_conv_state(total_entries, dim, state_len, dtype,
                                       device):
    backing = torch.randn(total_entries * 2,
                          dim,
                          state_len,
                          device=device,
                          dtype=dtype)
    return backing[::2]


def _make_stride_state_indices(total_entries, device):
    return torch.arange(total_entries,
                        device=device,
                        dtype=torch.int32).repeat_interleave(2)[::2]


def _build_case(case_name: str, batch: int, device: str) -> dict:
    dtype = torch.bfloat16
    dim = 2048
    if case_name.endswith("_4096"):
        dim = 4096

    weight = torch.randn(dim, 4, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    query_start_loc = None
    num_accepted_tokens = None
    max_query_len = -1

    if case_name == "decode_contig":
        x = torch.randn(batch, dim, device=device, dtype=dtype)
        conv_state = torch.randn(batch, dim, 3, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
    elif case_name == "decode_state_stride":
        x = torch.randn(batch, dim, device=device, dtype=dtype)
        conv_state = _make_cache_line_stride_conv_state(batch, dim, 3, dtype,
                                                        device)
        conv_state_indices = _make_stride_state_indices(batch, device)
    elif case_name == "mtp1_contig_all1":
        x = torch.randn(batch * 2, dim, device=device, dtype=dtype)
        conv_state = torch.randn(batch, dim, 4, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
        query_start_loc = torch.arange(0, (batch + 1) * 2, 2, device=device,
                                       dtype=torch.int32)
        num_accepted_tokens = torch.ones(batch,
                                         device=device,
                                         dtype=torch.int32)
        max_query_len = 2
    elif case_name == "mtp1_contig_alt12":
        x = torch.randn(batch * 2, dim, device=device, dtype=dtype)
        conv_state = torch.randn(batch, dim, 4, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
        query_start_loc = torch.arange(0, (batch + 1) * 2, 2, device=device,
                                       dtype=torch.int32)
        num_accepted_tokens = torch.tensor([1, 2] * (batch // 2),
                                           device=device,
                                           dtype=torch.int32)
        max_query_len = 2
    elif case_name == "mtp1_state_stride_all1":
        x = torch.randn(batch * 2, dim, device=device, dtype=dtype)
        conv_state = _make_cache_line_stride_conv_state(batch, dim, 4, dtype,
                                                        device)
        conv_state_indices = _make_stride_state_indices(batch, device)
        query_start_loc = torch.arange(0, (batch + 1) * 2, 2, device=device,
                                       dtype=torch.int32)
        num_accepted_tokens = torch.ones(batch,
                                         device=device,
                                         dtype=torch.int32)
        max_query_len = 2
    elif case_name == "mtp1_state_stride_alt12":
        x = torch.randn(batch * 2, dim, device=device, dtype=dtype)
        conv_state = _make_cache_line_stride_conv_state(batch, dim, 4, dtype,
                                                        device)
        conv_state_indices = _make_stride_state_indices(batch, device)
        query_start_loc = torch.arange(0, (batch + 1) * 2, 2, device=device,
                                       dtype=torch.int32)
        num_accepted_tokens = torch.tensor([1, 2] * (batch // 2),
                                           device=device,
                                           dtype=torch.int32)
        max_query_len = 2
    elif case_name == "decode_contig_4096":
        x = torch.randn(batch, dim, device=device, dtype=dtype)
        conv_state = torch.randn(batch, dim, 3, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
    elif case_name == "mtp1_contig_all1_4096":
        x = torch.randn(batch * 2, dim, device=device, dtype=dtype)
        conv_state = torch.randn(batch, dim, 4, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
        query_start_loc = torch.arange(0, (batch + 1) * 2, 2, device=device,
                                       dtype=torch.int32)
        num_accepted_tokens = torch.ones(batch,
                                         device=device,
                                         dtype=torch.int32)
        max_query_len = 2
    elif case_name == "mtp3_contig_alt1234_2048":
        x = torch.randn(batch * 4, dim, device=device, dtype=dtype)
        conv_state = torch.randn(batch, dim, 6, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
        query_start_loc = torch.arange(0, (batch + 1) * 4, 4, device=device,
                                       dtype=torch.int32)
        num_accepted_tokens = torch.tensor(([1, 2, 3, 4] * (batch // 4)),
                                           device=device,
                                           dtype=torch.int32)
        max_query_len = 4
    else:
        raise ValueError(f"unsupported case: {case_name}")

    return {
        "batch": batch,
        "dim": dim,
        "x": x,
        "x_base": x.clone(),
        "conv_state": conv_state,
        "conv_state_base": conv_state.clone(),
        "weight": weight,
        "bias": bias,
        "conv_state_indices": conv_state_indices,
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": num_accepted_tokens,
        "max_query_len": max_query_len,
    }


def _measure_latency_us(run, setup_fn=None, warmup=100, measure=1000):
    starter = torch.npu.Event(enable_timing=True)
    ender = torch.npu.Event(enable_timing=True)

    for _ in range(warmup):
        if setup_fn is not None:
            setup_fn()
        run()
    torch.npu.synchronize()

    samples = []
    for _ in range(measure):
        if setup_fn is not None:
            setup_fn()
        torch.npu.synchronize()
        starter.record()
        run()
        ender.record()
        torch.npu.synchronize()
        samples.append(starter.elapsed_time(ender) * 1000.0)

    samples_tensor = torch.tensor(samples, dtype=torch.float64)
    return {
        "mean_us": samples_tensor.mean().item(),
        "p50_us": samples_tensor.quantile(0.5).item(),
        "p95_us": samples_tensor.quantile(0.95).item(),
    }


def _resolve_dispatch(case: dict) -> tuple[str | None, tuple[int, int, int] | None]:
    path = _select_causal_conv1d_update_sla_fast_path(
        x=case["x"],
        conv_state=case["conv_state"],
        weight=case["weight"],
        bias=case["bias"],
        activation="silu",
        conv_state_indices=case["conv_state_indices"],
        num_accepted_tokens=case["num_accepted_tokens"],
        query_start_loc=case["query_start_loc"],
        max_query_len=case["max_query_len"],
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
    )
    if path is not None:
        return path, _pick_causal_conv1d_update_sla_launch_params(
            path,
            batch=case["batch"],
            dim=case["dim"],
        )

    path = _select_causal_conv1d_update_fast_path(
        x=case["x"],
        conv_state=case["conv_state"],
        weight=case["weight"],
        bias=case["bias"],
        activation="silu",
        conv_state_indices=case["conv_state_indices"],
        num_accepted_tokens=case["num_accepted_tokens"],
        query_start_loc=case["query_start_loc"],
        max_query_len=case["max_query_len"],
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
    )
    if path is not None:
        return path, _pick_causal_conv1d_update_fast_path_launch_params(
            path,
            batch=case["batch"],
            dim=case["dim"],
        )
    return None, None


def benchmark_case(case_name: str, batch: int, warmup: int, measure: int,
                   device: str) -> None:
    case = _build_case(case_name, batch, device)

    def setup():
        case["x"].copy_(case["x_base"])
        case["conv_state"].copy_(case["conv_state_base"])

    def run():
        causal_conv1d_update_npu(
            case["x"],
            case["conv_state"],
            case["weight"],
            case["bias"],
            activation="silu",
            conv_state_indices=case["conv_state_indices"],
            num_accepted_tokens=case["num_accepted_tokens"],
            query_start_loc=case["query_start_loc"],
            max_query_len=case["max_query_len"],
            pad_slot_id=PAD_SLOT_ID,
        )

    path, launch = _resolve_dispatch(case)
    stats = _measure_latency_us(run, setup_fn=setup, warmup=warmup,
                                measure=measure)
    print(f"CASE={case_name}")
    print(f"PATH={path}")
    print(f"LAUNCH={launch}")
    print(f"MEAN_US={stats['mean_us']:.3f}")
    print(f"P50_US={stats['p50_us']:.3f}")
    print(f"P95_US={stats['p95_us']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",
                        default="all",
                        choices=["all", "hard", "soft"] + HARD_CASES +
                        SOFT_CASES)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure", type=int, default=1000)
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    if args.case == "all":
        cases = HARD_CASES + SOFT_CASES
    elif args.case == "hard":
        cases = HARD_CASES
    elif args.case == "soft":
        cases = SOFT_CASES
    else:
        cases = [args.case]

    for case_name in cases:
        try:
            benchmark_case(case_name, args.batch, args.warmup, args.measure,
                           args.device)
        except Exception as exc:
            print(f"CASE={case_name}")
            print(f"ERROR={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

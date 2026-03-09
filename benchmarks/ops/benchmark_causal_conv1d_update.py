import argparse

import torch
import torch_npu  # noqa: F401
import vllm  # noqa: F401

import vllm_ascend.platform  # noqa: F401
from vllm_ascend.ops.triton.mamba.causal_conv1d import (
    PAD_SLOT_ID,
    _pick_causal_conv1d_update_fast_path_launch_params,
    _select_causal_conv1d_update_fast_path,
    causal_conv1d_update_npu,
)


def _make_regular_stride_view(base: torch.Tensor) -> torch.Tensor:
    if base.dim() == 3:
        return base[:, 1::2, :]
    return base[:, 1::2]


def _build_case(case_name: str, batch: int, dim: int, device: str) -> dict:
    dtype = torch.bfloat16
    if case_name == "decode_contig":
        x = torch.randn(batch, dim, 1, device=device, dtype=dtype)
        conv_state = torch.randn(batch, 3, dim, device=device, dtype=dtype).transpose(1, 2)
        weight = torch.randn(dim, 4, device=device, dtype=dtype)
        query_start_loc = None
        num_accepted_tokens = None
    elif case_name == "decode_stride":
        x = _make_regular_stride_view(
            torch.randn(batch, dim * 2, 1, device=device, dtype=dtype))
        conv_state = _make_regular_stride_view(
            torch.randn(batch, dim * 2, 3, device=device, dtype=dtype))
        weight = torch.randn(dim * 2, 4, device=device, dtype=dtype)[1::2, :]
        query_start_loc = None
        num_accepted_tokens = None
    elif case_name == "update_contig":
        x = torch.randn(batch, dim, 3, device=device, dtype=dtype)
        conv_state = torch.randn(batch, 3, dim, device=device, dtype=dtype).transpose(1, 2)
        weight = torch.randn(dim, 4, device=device, dtype=dtype)
        query_start_loc = None
        num_accepted_tokens = None
    elif case_name == "update_stride":
        x = _make_regular_stride_view(
            torch.randn(batch, dim * 2, 3, device=device, dtype=dtype))
        conv_state = _make_regular_stride_view(
            torch.randn(batch, dim * 2, 3, device=device, dtype=dtype))
        weight = torch.randn(dim * 2, 4, device=device, dtype=dtype)[1::2, :]
        query_start_loc = None
        num_accepted_tokens = None
    elif case_name == "mtp_contig":
        x = torch.randn(batch * 4, dim, device=device, dtype=dtype)
        conv_state = torch.randn(batch, 6, dim, device=device, dtype=dtype).transpose(1, 2)
        weight = torch.randn(dim, 4, device=device, dtype=dtype)
        query_start_loc = torch.arange(0, (batch + 1) * 4, 4, device=device, dtype=torch.int32)
        num_accepted_tokens = torch.tensor(([1, 2, 4, 1] * (batch // 4)),
                                           device=device,
                                           dtype=torch.int32)
    elif case_name == "mtp_stride":
        x = _make_regular_stride_view(
            torch.randn(batch * 4, dim * 2, device=device, dtype=dtype))
        conv_state = _make_regular_stride_view(
            torch.randn(batch, dim * 2, 6, device=device, dtype=dtype))
        weight = torch.randn(dim * 2, 4, device=device, dtype=dtype)[1::2, :]
        query_start_loc = torch.arange(0, (batch + 1) * 4, 4, device=device, dtype=torch.int32)
        num_accepted_tokens = torch.tensor(([1, 2, 4, 1] * (batch // 4)),
                                           device=device,
                                           dtype=torch.int32)
    else:
        raise ValueError(f"Unsupported case: {case_name}")

    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
    return {
        "x": x,
        "x_base": x.clone(),
        "conv_state": conv_state,
        "conv_state_base": conv_state.clone(),
        "weight": weight,
        "bias": bias,
        "conv_state_indices": conv_state_indices,
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": num_accepted_tokens,
        "max_query_len": 4 if case_name.startswith("mtp_") else -1,
    }


def _measure_latency_us(run, setup, warmup: int, measure: int) -> dict[str, float]:
    starter = torch.npu.Event(enable_timing=True)
    ender = torch.npu.Event(enable_timing=True)

    for _ in range(warmup):
        setup()
        run()
    torch.npu.synchronize()

    samples = []
    for _ in range(measure):
        setup()
        torch.npu.synchronize()
        starter.record()
        run()
        ender.record()
        torch.npu.synchronize()
        samples.append(starter.elapsed_time(ender) * 1000.0)

    stats = torch.tensor(samples, dtype=torch.float64)
    return {
        "mean_us": stats.mean().item(),
        "p50_us": stats.quantile(0.5).item(),
        "p95_us": stats.quantile(0.95).item(),
    }


def benchmark_case(case_name: str, batch: int, dim: int, warmup: int,
                   measure: int, device: str) -> None:
    case = _build_case(case_name, batch, dim, device)

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

    fast_path = _select_causal_conv1d_update_fast_path(
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
    stats = _measure_latency_us(run, setup, warmup, measure)

    if fast_path is not None:
        launch_config = _pick_causal_conv1d_update_fast_path_launch_params(
            fast_path,
            batch=batch,
            dim=dim,
        )
    else:
        launch_config = None

    print(f"CASE={case_name}")
    print(f"PATH={fast_path}")
    print(f"LAUNCH={launch_config}")
    print(f"MEAN_US={stats['mean_us']:.3f}")
    print(f"P50_US={stats['p50_us']:.3f}")
    print(f"P95_US={stats['p95_us']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="all",
        choices=[
            "all",
            "decode_contig",
            "decode_stride",
            "update_contig",
            "update_stride",
            "mtp_contig",
            "mtp_stride",
        ],
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure", type=int, default=1000)
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    cases = [
        "decode_contig",
        "decode_stride",
        "update_contig",
        "update_stride",
        "mtp_contig",
        "mtp_stride",
    ]
    if args.case != "all":
        cases = [args.case]

    for case_name in cases:
        benchmark_case(case_name, args.batch, args.dim, args.warmup,
                       args.measure, args.device)


if __name__ == "__main__":
    main()

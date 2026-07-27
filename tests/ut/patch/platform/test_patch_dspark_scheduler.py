from types import SimpleNamespace

from vllm_ascend.patch.platform import patch_dspark_scheduler


def test_remote_decode_dspark_skips_local_prefix_cache():
    manager = SimpleNamespace(
        _ascend_dspark_skip_local_prefix_cache=True,
        empty_kv_cache_blocks="empty",
    )
    request = SimpleNamespace(
        kv_transfer_params={"do_remote_decode": True},
    )

    assert patch_dspark_scheduler._get_computed_blocks(manager, request) == (
        "empty",
        0,
    )


def test_regular_request_keeps_existing_prefix_cache_path(monkeypatch):
    manager = SimpleNamespace(
        _ascend_dspark_skip_local_prefix_cache=True,
        empty_kv_cache_blocks="empty",
    )
    request = SimpleNamespace(kv_transfer_params=None)
    monkeypatch.setattr(
        patch_dspark_scheduler,
        "_ORIGINAL_GET_COMPUTED_BLOCKS",
        lambda current_manager, current_request: (
            current_manager,
            current_request,
        ),
    )

    assert patch_dspark_scheduler._get_computed_blocks(manager, request) == (
        manager,
        request,
    )

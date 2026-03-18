from types import SimpleNamespace

import numpy as np
import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerGDNMetadata(TestBase):
    def test_build_gdn_common_attn_metadata_uses_real_requests(self):
        runner = object.__new__(NPUModelRunner)
        runner.gdn_query_start_loc = SimpleNamespace(
            cpu=torch.tensor([0, 4, 9, 9], dtype=torch.int32),
            gpu=torch.tensor([0, 4, 9, 9], dtype=torch.int32),
        )
        runner.seq_lens = SimpleNamespace(
            cpu=torch.tensor([12, 21, 0], dtype=torch.int32),
            gpu=torch.tensor([12, 21, 0], dtype=torch.int32),
        )
        runner.input_batch = SimpleNamespace(
            num_computed_tokens_cpu_tensor=torch.tensor([8, 17, -1], dtype=torch.int32),
        )

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 4, 9, 9], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 4, 9, 9], dtype=torch.int32),
            seq_lens=torch.tensor([12, 21, 0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([12, 21, 0], dtype=torch.int32),
            num_computed_tokens_cpu=torch.tensor([8, 17, -1], dtype=torch.int32),
            num_reqs=3,
            num_actual_tokens=9,
            max_query_len=5,
            max_seq_len=21,
            block_table_tensor=torch.arange(12, dtype=torch.int32).reshape(3, 4),
            slot_mapping=torch.arange(9, dtype=torch.int64),
            encoder_seq_lens=torch.tensor([30, 40, 0], dtype=torch.int32),
            encoder_seq_lens_cpu=np.array([30, 40, 0], dtype=np.int32),
            actual_seq_lengths_q=[1] * 9,
            positions=torch.arange(9, dtype=torch.int32),
            attn_state=None,
            decode_token_per_req=1,
        )

        gdn_common_attn_metadata = runner._build_gdn_common_attn_metadata(common_attn_metadata, num_reqs=2)

        self.assertEqual(gdn_common_attn_metadata.num_reqs, 2)
        self.assertTrue(
            torch.equal(
                gdn_common_attn_metadata.query_start_loc_cpu,
                torch.tensor([0, 4, 9], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                gdn_common_attn_metadata.query_start_loc,
                torch.tensor([0, 4, 9], dtype=torch.int32),
            )
        )
        self.assertTrue(torch.equal(gdn_common_attn_metadata.seq_lens_cpu, torch.tensor([12, 21], dtype=torch.int32)))
        self.assertTrue(torch.equal(gdn_common_attn_metadata.seq_lens, torch.tensor([12, 21], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                gdn_common_attn_metadata.num_computed_tokens_cpu,
                torch.tensor([8, 17], dtype=torch.int32),
            )
        )
        self.assertEqual(gdn_common_attn_metadata.block_table_tensor.shape[0], 2)
        self.assertTrue(
            torch.equal(
                gdn_common_attn_metadata.encoder_seq_lens,
                torch.tensor([30, 40], dtype=torch.int32),
            )
        )
        self.assertTrue(np.array_equal(gdn_common_attn_metadata.encoder_seq_lens_cpu, np.array([30, 40])))
        self.assertTrue(torch.equal(gdn_common_attn_metadata.slot_mapping, torch.arange(9, dtype=torch.int64)))

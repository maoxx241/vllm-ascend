from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from .types import EvidenceAtom, Recipe


class OpenWorldCompiler:
    """Compile repo facts into typed evidence instead of final answers.

    Design rules:
    - missing support-matrix rows are never negative evidence
    - local source and upstream mirror evidence are first-class
    - docs/tutorials provide positive evidence, constraints, or hints
    - recipes are reusable deployment exemplars, not the final answer
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.vllm_root = workspace_root / 'vllm'
        self.ascend_root = workspace_root / 'vllm-ascend'

    def compile(self, out_dir: Path) -> dict[str, int]:
        out_dir.mkdir(parents=True, exist_ok=True)
        evidence = list(self._collect_evidence())
        recipes = list(self._collect_recipes())
        self._write_jsonl(out_dir / 'evidence.jsonl', (e.to_dict() for e in evidence))
        self._write_jsonl(out_dir / 'recipes.jsonl', (r.to_dict() for r in recipes))
        self._write_sqlite(out_dir / 'catalog.sqlite', evidence, recipes)
        summary = {
            'evidence_count': len(evidence),
            'recipe_count': len(recipes),
            'local_source_evidence': sum(1 for e in evidence if e.source_tier == 'local_source'),
            'local_docs_evidence': sum(1 for e in evidence if e.source_tier == 'local_docs'),
            'upstream_repo_mirror_evidence': sum(1 for e in evidence if e.source_tier == 'upstream_repo_mirror'),
        }
        (out_dir / 'SUMMARY.md').write_text(self._render_summary(summary), encoding='utf-8')
        return summary

    def _collect_evidence(self) -> Iterable[EvidenceAtom]:
        yield from self._collect_qwen35_local_source_evidence()
        yield from self._collect_upstream_supported_models()
        yield from self._collect_ascend_support_matrix()
        yield from self._collect_qwen3_dense_tutorial()
        yield from self._collect_qwen3_w4a4_tutorial()
        yield from self._collect_310p_constraints()
        yield from self._collect_quantization_capabilities()
        yield from self._collect_deepseek_v31_tutorial()
        yield from self._collect_nightly_test_hints()

    def _collect_recipes(self) -> Iterable[Recipe]:
        yield Recipe(
            recipe_id='r.qwen3_32b_a3_tp4_throughput',
            subject='model:qwen3-32b',
            scenario='single_instance_throughput_a3',
            evidence_refs=['e.qwen3dense.tutorial.exists', 'e.qwen3dense.tp4.example', 'e.qwen3dense.full_decode_only', 'e.qwen3dense.longseq.128k'],
            command_template=(
                'vllm serve {weight_path} --served-model-name qwen3 --trust-remote-code '
                '--async-scheduling --distributed-executor-backend mp '
                '--tensor-parallel-size 4 {dp_flag} '
                '--max-model-len {max_model_len} --max-num-batched-tokens {max_num_batched_tokens} '
                '--compilation-config {compilation_config} --additional-config {additional_config} '
                '--gpu-memory-utilization 0.9 --block-size 128'
            ),
            env={
                'TASK_QUEUE_ENABLE': '1',
                'HCCL_OP_EXPANSION_MODE': 'AIV',
                'VLLM_ASCEND_ENABLE_FLASHCOMM1': '1',
            },
            flags={'tp': 4},
            assumptions=['single instance', 'dense qwen3 family', 'throughput-oriented serving'],
            constraints=['A3 single node defaults to 8 cards/16 chips', 'TP8 is not assumed optimal for 8-card single-instance serving'],
            note='Derived from Qwen3-Dense tutorial and nightly feature-stack examples.',
        )
        yield Recipe(
            recipe_id='r.qwen3_32b_w4a4_single',
            subject='model:qwen3-32b-w4a4',
            scenario='single_instance_basic',
            evidence_refs=['e.qwen3w4a4.deploy.command'],
            command_template='vllm serve {weight_path} --served-model-name qwen3-32b-w4a4 --max-model-len 4096 --quantization ascend',
            assumptions=['weights are already quantized to W4A4'],
            note='Single-instance serving command from the official Qwen3-32B-W4A4 tutorial.',
        )
        yield Recipe(
            recipe_id='r.deepseek_v31_w8a8_a3_single',
            subject='model:deepseek-v3.1-w8a8',
            scenario='single_instance_a3_quantized',
            evidence_refs=['e.deepseekv31.a3.single_quantized'],
            command_template=(
                'vllm serve {weight_path} --host 0.0.0.0 --port {port} '
                '--data-parallel-size 4 --tensor-parallel-size 4 --quantization ascend '
                '--seed 1024 --served-model-name deepseek_v3 --enable-expert-parallel'
            ),
            env={
                'HCCL_OP_EXPANSION_MODE': 'AIV',
                'VLLM_ASCEND_BALANCE_SCHEDULING': '1',
                'PYTORCH_NPU_ALLOC_CONF': 'expandable_segments:True',
            },
            assumptions=['A3 64G×16', 'quantized DeepSeek-V3.1 variant'],
            note='Single-node A3 quantized DeepSeek-V3.1 tutorial path.',
        )

    # ---------- evidence collectors ----------

    def _collect_qwen35_local_source_evidence(self) -> Iterable[EvidenceAtom]:
        local_paths = [
            self.vllm_root / 'vllm/model_executor/models/qwen3_5.py',
            self.vllm_root / 'vllm/model_executor/models/qwen3_5_mtp.py',
            self.vllm_root / 'vllm/transformers_utils/configs/qwen3_5.py',
            self.ascend_root / 'vllm_ascend/patch/worker/patch_qwen3_5.py',
            self.ascend_root / 'vllm_ascend/quantization/modelslim_config.py',
        ]
        idx = 0
        for path in local_paths:
            if path.exists():
                idx += 1
                yield EvidenceAtom(
                    evidence_id=f'e.qwen35.local_surface.{idx}',
                    subject='model_family:qwen3.5',
                    predicate='has_local_surface',
                    value='true',
                    source_tier='local_source',
                    polarity='positive',
                    source_ref=self._rel(path),
                    note='Local implementation/patch presence for Qwen3.5 support surfaces.',
                )
        registry = self.vllm_root / 'vllm/model_executor/models/registry.py'
        if registry.exists() and 'Qwen3_5ForConditionalGeneration' in registry.read_text(encoding='utf-8', errors='ignore'):
            yield EvidenceAtom(
                evidence_id='e.qwen35.registry',
                subject='model_family:qwen3.5',
                predicate='registered_upstream_model',
                value='Qwen3_5ForConditionalGeneration',
                source_tier='upstream_repo_mirror',
                polarity='positive',
                source_ref=self._rel(registry),
                note='Local upstream mirror registers Qwen3.5 model class.',
            )

    def _collect_upstream_supported_models(self) -> Iterable[EvidenceAtom]:
        path = self.vllm_root / 'docs/models/supported_models.md'
        if not path.exists():
            return
        text = path.read_text(encoding='utf-8', errors='ignore')
        patterns = {
            'Qwen3_5ForConditionalGeneration': 'model_family:qwen3.5',
            'Qwen3_5MoeForConditionalGeneration': 'model_family:qwen3.5-moe',
            'Qwen2VLForConditionalGeneration': 'model_family:qwen2-vl',
        }
        for arch, subject in patterns.items():
            if arch in text:
                yield EvidenceAtom(
                    evidence_id=f'e.upstream.{arch.lower()}',
                    subject=subject,
                    predicate='listed_in_upstream_supported_models',
                    value=arch,
                    source_tier='upstream_repo_mirror',
                    polarity='positive',
                    source_ref=self._rel(path),
                    note='Upstream vLLM support page is strong positive evidence.',
                )

    def _collect_ascend_support_matrix(self) -> Iterable[EvidenceAtom]:
        path = self.ascend_root / 'docs/source/user_guide/support_matrix/supported_models.md'
        if not path.exists():
            return
        rows = [line for line in path.read_text(encoding='utf-8', errors='ignore').splitlines() if line.strip().startswith('|')]
        checks = {
            'support_matrix:qwen3-dense': ['qwen3 dense'],
            'support_matrix:qwen2-vl': ['qwen2-vl'],
            'support_matrix:deepseek-v3.1': ['deepseek v3/3.1', 'deepseek-v3.1'],
        }
        idx = 0
        for row in rows:
            low = row.lower()
            for subject, aliases in checks.items():
                if any(a in low for a in aliases):
                    idx += 1
                    yield EvidenceAtom(
                        evidence_id=f'e.support_matrix.{idx}',
                        subject=subject,
                        predicate='listed_supported',
                        value=row.strip(),
                        source_tier='local_docs',
                        polarity='positive',
                        source_ref=self._rel(path),
                        note='Support matrix is positive evidence, not a closed-world ceiling.',
                    )

    def _collect_qwen3_dense_tutorial(self) -> Iterable[EvidenceAtom]:
        path = self.ascend_root / 'docs/source/tutorials/models/Qwen3-Dense.md'
        if not path.exists():
            return
        text = path.read_text(encoding='utf-8', errors='ignore')
        yield EvidenceAtom(
            evidence_id='e.qwen3dense.tutorial.exists',
            subject='model_family:qwen3-dense',
            predicate='has_model_tutorial',
            value='true',
            source_tier='local_docs',
            polarity='positive',
            source_ref=self._rel(path),
            note='Qwen3 dense family has an explicit vLLM-Ascend tutorial.',
        )
        if 'DP=1&TP=4' in text:
            yield EvidenceAtom(
                evidence_id='e.qwen3dense.tp4.example',
                subject='scenario:qwen3-32b-throughput-a3',
                predicate='verified_parallel_shape',
                value='tp4_dp1_core_example',
                source_tier='local_docs',
                polarity='positive',
                source_ref=self._rel(path),
                note='Tutorial example for Qwen3-32B-W8A8 uses TP4 as the core shape.',
            )
        if 'If the machine environment is an **Atlas 800I A2(64G*8)**, the deployment approach stays identical.' in text:
            yield EvidenceAtom(
                evidence_id='e.qwen3dense.a3_a2_same_approach',
                subject='scenario:qwen3-32b-throughput-a2a3',
                predicate='same_deployment_approach',
                value='a3_example_transfers_to_a2_64g_x8',
                source_tier='local_docs',
                polarity='hint',
                source_ref=self._rel(path),
                note='Tutorial explicitly says the A2(64G×8) approach stays identical to the A3 example.',
            )
        if 'FULL_DECODE_ONLY' in text:
            yield EvidenceAtom(
                evidence_id='e.qwen3dense.full_decode_only',
                subject='feature:full_decode_only',
                predicate='used_in_qwen3dense_tutorial',
                value='true',
                source_tier='local_docs',
                polarity='positive',
                source_ref=self._rel(path),
                note='Optimized Qwen3-Dense serving example uses FULL_DECODE_ONLY.',
            )
        rel_notes = self.ascend_root / 'docs/source/user_guide/release_notes.md'
        if rel_notes.exists():
            notes = rel_notes.read_text(encoding='utf-8', errors='ignore')
            if 'Qwen3-32B' in notes and '128K input case' in notes:
                yield EvidenceAtom(
                    evidence_id='e.qwen3dense.longseq.128k',
                    subject='model:qwen3-32b',
                    predicate='known_long_seq_limit',
                    value='128k_input_case_needs_pcp_dcp',
                    source_tier='local_docs',
                    polarity='constraint',
                    source_ref=self._rel(rel_notes),
                    note='Long-seq special handling is called out for 128K input, not ordinary 40K max-context serving.',
                )

    def _collect_qwen3_w4a4_tutorial(self) -> Iterable[EvidenceAtom]:
        path = self.ascend_root / 'docs/source/tutorials/models/Qwen3-32B-W4A4.md'
        if not path.exists():
            return
        text = path.read_text(encoding='utf-8', errors='ignore')
        if 'The following steps will show how to quantize Qwen3 32B to W4A4.' in text:
            yield EvidenceAtom(
                evidence_id='e.qwen3w4a4.quant_flow_exists',
                subject='quantization:qwen3-32b-w4a4',
                predicate='has_official_quant_flow',
                value='true',
                source_tier='local_docs',
                polarity='positive',
                source_ref=self._rel(path),
                note='Official quantization flow exists for Qwen3-32B-W4A4.',
            )
        m = re.search(r'vllm serve /home/models/Qwen3-32B-w4a4.*--quantization ascend', text)
        if m:
            yield EvidenceAtom(
                evidence_id='e.qwen3w4a4.deploy.command',
                subject='model:qwen3-32b-w4a4',
                predicate='has_verified_single_instance_command',
                value='vllm serve /home/models/Qwen3-32B-w4a4 --served-model-name qwen3-32b-w4a4 --max-model-len 4096 --quantization ascend',
                source_tier='local_docs',
                polarity='positive',
                source_ref=self._rel(path),
                note='Tutorial contains a concrete W4A4 serving command.',
            )

    def _collect_310p_constraints(self) -> Iterable[EvidenceAtom]:
        path = self.ascend_root / 'docs/source/tutorials/hardwares/310p.md'
        if not path.exists():
            return
        text = path.read_text(encoding='utf-8', errors='ignore')
        if 'only supports eager mode and the float16 data type' in text:
            yield EvidenceAtom(
                evidence_id='e.310p.eager_float16',
                subject='hardware:310p',
                predicate='execution_constraint',
                value='eager_only_float16_only',
                source_tier='local_docs',
                polarity='constraint',
                source_ref=self._rel(path),
                note='310P serving path currently requires eager mode and float16.',
            )
        if 'do not rely on `max-model-len` auto detection' in text:
            yield EvidenceAtom(
                evidence_id='e.310p.max_model_len',
                subject='hardware:310p',
                predicate='serving_constraint',
                value='set_conservative_max_model_len_explicitly',
                source_tier='local_docs',
                polarity='constraint',
                source_ref=self._rel(path),
                note='310P attention path can OOM if max-model-len is auto-detected.',
            )
        release_notes = self.ascend_root / 'docs/source/user_guide/release_notes.md'
        if release_notes.exists():
            notes = release_notes.read_text(encoding='utf-8', errors='ignore')
            if 'Qwen3-0.6B/Qwen3-4B/Qwen3-8B' in notes:
                yield EvidenceAtom(
                    evidence_id='e.310p.example_dense_models',
                    subject='hardware:310p',
                    predicate='example_verified_models',
                    value='Qwen3-0.6B,Qwen3-4B,Qwen3-8B,Qwen2.5-7B-Instruct,Qwen2.5-0.5B',
                    source_tier='local_docs',
                    polarity='hint',
                    source_ref=self._rel(release_notes),
                    note='Examples show some small dense models exercised on 310P; this is a hint, not a blanket ceiling.',
                )

    def _collect_quantization_capabilities(self) -> Iterable[EvidenceAtom]:
        path = self.ascend_root / 'docs/source/faqs.md'
        if not path.exists():
            return
        text = path.read_text(encoding='utf-8', errors='ignore')
        if 'w8a8, w4a8, and w4a4 quantization methods are already supported' in text:
            yield EvidenceAtom(
                evidence_id='e.quant.general_methods',
                subject='quantization:general',
                predicate='supported_methods',
                value='w8a8,w4a8,w4a4',
                source_tier='local_docs',
                polarity='positive',
                source_ref=self._rel(path),
                note='Generic quantization capability does not mean every model combination is officially validated.',
            )

    def _collect_deepseek_v31_tutorial(self) -> Iterable[EvidenceAtom]:
        path = self.ascend_root / 'docs/source/tutorials/models/DeepSeek-V3.1.md'
        if not path.exists():
            return
        text = path.read_text(encoding='utf-8', errors='ignore')
        if 'require at least 2 Atlas 800 A2 (64G × 8)' in text:
            yield EvidenceAtom(
                evidence_id='e.deepseekv31.a2.minimum',
                subject='model:deepseek-v3.1',
                predicate='minimum_a2_footprint',
                value='at_least_2x_atlas800_a2_64g_x8',
                source_tier='local_docs',
                polarity='constraint',
                source_ref=self._rel(path),
                note='A2 deployment guidance is far above single-node dual-card footprint.',
            )
        if 'can be deployed on 1 Atlas 800 A3 (64G × 16)' in text:
            yield EvidenceAtom(
                evidence_id='e.deepseekv31.a3.single_quantized',
                subject='model:deepseek-v3.1-w8a8',
                predicate='known_a3_single_node_footprint',
                value='1x_atlas800_a3_64g_x16',
                source_tier='local_docs',
                polarity='positive',
                source_ref=self._rel(path),
                note='Single-node A3 guidance exists for quantized DeepSeek-V3.1.',
            )

    def _collect_nightly_test_hints(self) -> Iterable[EvidenceAtom]:
        # Optional hints from nightly configs: stronger than generic docs, weaker than direct user facts.
        configs = [
            self.ascend_root / 'tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8.yaml',
            self.ascend_root / 'tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8-A3-Feature-Stack3.yaml',
        ]
        idx = 0
        for path in configs:
            if not path.exists():
                continue
            text = path.read_text(encoding='utf-8', errors='ignore')
            if 'Qwen3-32B-W8A8' in text:
                idx += 1
                yield EvidenceAtom(
                    evidence_id=f'e.nightly.qwen3_32b_int8.{idx}',
                    subject='scenario:qwen3-32b-nightly',
                    predicate='has_nightly_config',
                    value=path.name,
                    source_tier='local_docs',
                    polarity='hint',
                    source_ref=self._rel(path),
                    note='Nightly config hints can strengthen deployment candidates without becoming hard ceilings.',
                )

    # ---------- writers ----------

    def _write_jsonl(self, path: Path, rows: Iterable[dict]) -> None:
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def _write_sqlite(self, path: Path, evidence: list[EvidenceAtom], recipes: list[Recipe]) -> None:
        if path.exists():
            path.unlink()
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute(
            'create table evidence ('
            'evidence_id text primary key, subject text, predicate text, value text, '
            'source_tier text, polarity text, source_ref text, note text, tags_json text)'
        )
        cur.execute(
            'create table recipes ('
            'recipe_id text primary key, subject text, scenario text, evidence_refs_json text, '
            'command_template text, env_json text, flags_json text, assumptions_json text, constraints_json text, note text)'
        )
        for e in evidence:
            cur.execute(
                'insert into evidence values (?,?,?,?,?,?,?,?,?)',
                (e.evidence_id, e.subject, e.predicate, e.value, e.source_tier, e.polarity, e.source_ref, e.note, json.dumps(e.tags, ensure_ascii=False)),
            )
        for r in recipes:
            cur.execute(
                'insert into recipes values (?,?,?,?,?,?,?,?,?,?)',
                (r.recipe_id, r.subject, r.scenario, json.dumps(r.evidence_refs, ensure_ascii=False), r.command_template, json.dumps(r.env, ensure_ascii=False), json.dumps(r.flags, ensure_ascii=False), json.dumps(r.assumptions, ensure_ascii=False), json.dumps(r.constraints, ensure_ascii=False), r.note),
            )
        con.commit()
        con.close()

    # ---------- helpers ----------

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.workspace_root))

    def _render_summary(self, summary: dict[str, int]) -> str:
        lines = ['# Deployment compiler sample build', '']
        for k, v in summary.items():
            lines.append(f'- {k}: {v}')
        lines.append('')
        lines.append('This build stores evidence atoms and recipes instead of final answers.')
        return '\n'.join(lines) + '\n'

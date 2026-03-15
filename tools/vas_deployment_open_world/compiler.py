from __future__ import annotations

from pathlib import Path
from .models import EvidenceAtom, Recipe


class EvidenceCompiler:
    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root)
        self.workspace_root = self.repo_root.parent
        self.upstream_vllm_root = self.workspace_root / 'vllm'

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            try:
                return str(path.relative_to(self.workspace_root))
            except ValueError:
                return str(path)

    def _read(self, rel: str):
        path = self.repo_root / rel
        if not path.exists():
            return None, ''
        return path, path.read_text(encoding='utf-8', errors='ignore')

    def compile(self) -> tuple[list[EvidenceAtom], list[Recipe]]:
        evidence: list[EvidenceAtom] = []
        recipes: list[Recipe] = []
        evidence.extend(self._support_matrix_rows())
        evidence.extend(self._glm_rows())
        evidence.extend(self._qwen_rows())
        evidence.extend(self._deepseek_rows())
        evidence.extend(self._hardware_310p_rows())
        evidence.extend(self._local_qwen35_rows())
        evidence.extend(self._generic_quant_rows())
        recipes.extend(self._glm_recipes())
        recipes.extend(self._qwen_recipes())
        return evidence, recipes

    def _support_matrix_rows(self) -> list[EvidenceAtom]:
        path, text = self._read('docs/source/user_guide/support_matrix/supported_models.md')
        if not path:
            return []
        out: list[EvidenceAtom] = []
        targets = {
            'glm4.x': 'GLM-4.x',
            'qwen3': 'Qwen3',
            'qwen2-vl': 'Qwen2-VL',
            'deepseek-v3.1': 'DeepSeek V3/3.1',
        }
        idx = 0
        for line in text.splitlines():
            if not line.strip().startswith('|'):
                continue
            for subject, needle in targets.items():
                if needle.lower() in line.lower():
                    idx += 1
                    out.append(EvidenceAtom(
                        evidence_id=f'e.matrix.{idx}',
                        subject=subject,
                        predicate='support_matrix_positive',
                        value=line.strip(),
                        source_ref=self._rel(path),
                        source_tier='repo_docs',
                        polarity='positive',
                        note='Support matrix is positive evidence, not a closed-world ceiling.',
                    ))
        return out

    def _glm_rows(self) -> list[EvidenceAtom]:
        path, text = self._read('docs/source/tutorials/models/GLM4.x.md')
        if not path:
            return []
        out: list[EvidenceAtom] = []
        if 'GLM-4.6' in text:
            out.append(EvidenceAtom('e.glm.doc.1', 'glm4.x', 'tutorial_covers_family', 'GLM-4.5/4.6/4.7', self._rel(path), 'repo_docs', 'positive', 'GLM4.x family tutorial exists.'))
        if 'dp1tp16' in text.lower() or 'tensor-parallel-size 16' in text:
            out.append(EvidenceAtom('e.glm.doc.2', 'glm4.x', 'single_node_low_latency_shape', 'dp1tp16_ep_off', self._rel(path), 'repo_docs', 'positive', 'Single-node low-latency route exists in tutorial.'))
        if 'turn off expert parallel' in text.lower():
            out.append(EvidenceAtom('e.glm.doc.3', 'glm4.x', 'single_node_low_latency_ep', 'off', self._rel(path), 'repo_docs', 'positive', 'Low-latency single-node route recommends EP off.'))
        nightly, nightly_text = self._read('tests/e2e/nightly/single_node/models/configs/GLM-4.5.yaml')
        if nightly and 'TP8-DP2-fullgraph' in nightly_text:
            out.append(EvidenceAtom('e.glm.nightly.1', 'glm4.x', 'single_node_throughput_shape', 'tp8dp2_fullgraph', self._rel(nightly), 'repo_tests', 'positive', 'Nightly config demonstrates TP8+DP2 route.'))
        if nightly and '--enable-expert-parallel' in nightly_text:
            out.append(EvidenceAtom('e.glm.nightly.2', 'glm4.x', 'single_node_throughput_ep', 'on', self._rel(nightly), 'repo_tests', 'positive', 'Nightly config demonstrates EP enabled on TP8+DP2 route.'))
        return out

    def _glm_recipes(self) -> list[Recipe]:
        path, text = self._read('docs/source/tutorials/models/GLM4.x.md')
        nightly, nightly_text = self._read('tests/e2e/nightly/single_node/models/configs/GLM-4.5.yaml')
        out: list[Recipe] = []
        if path and 'tensor-parallel-size 16' in text:
            out.append(Recipe(
                recipe_id='r.glm4x.a3.single.low_latency',
                subject='glm4.x',
                hardware=['A3', 'A2'],
                scenario_kind='low_latency_single_instance',
                topology={'tp': 16, 'dp': 1, 'ep': False},
                feature_policy={'full_graph': True, 'async_scheduling': True, 'mtp': 'normal_if_weight_support_present'},
                source_ref=self._rel(path),
                note='Tutorial-backed low-latency single-node route.',
            ))
        if nightly and 'TP8-DP2-fullgraph' in nightly_text:
            out.append(Recipe(
                recipe_id='r.glm4x.a3.single.throughput',
                subject='glm4.x',
                hardware=['A3'],
                scenario_kind='high_throughput_single_instance',
                topology={'tp': 8, 'dp': 2, 'ep': True},
                feature_policy={'full_graph': True, 'async_scheduling': True, 'mtp': 'normal_if_weight_support_present'},
                source_ref=self._rel(nightly),
                note='Repo-test-backed single-node throughput route.',
            ))
        return out

    def _qwen_rows(self) -> list[EvidenceAtom]:
        path, text = self._read('docs/source/tutorials/models/Qwen3-Dense.md')
        if not path:
            return []
        out: list[EvidenceAtom] = []
        if 'Qwen3-32B' in text:
            out.append(EvidenceAtom('e.qwen.doc.1', 'qwen3', 'tutorial_covers_family', 'true', self._rel(path), 'repo_docs', 'positive', 'Qwen3 Dense tutorial exists.'))
        if 'Qwen3-32B-W8A8' in text and 'tensor-parallel-size 4' in text:
            out.append(EvidenceAtom('e.qwen.doc.2', 'qwen3', 'throughput_shape', 'tp4', self._rel(path), 'repo_docs', 'positive', 'Qwen3 throughput example uses TP4 on A3/A2.'))
        if '3.5K' in text and '1.5K' in text:
            out.append(EvidenceAtom('e.qwen.doc.3', 'qwen3', 'example_length_shape', 'input3500_output1500', self._rel(path), 'repo_docs', 'hint', 'Qwen3 throughput example matches 3.5k/1.5k pattern.'))
        return out

    def _qwen_recipes(self) -> list[Recipe]:
        path, text = self._read('docs/source/tutorials/models/Qwen3-Dense.md')
        out: list[Recipe] = []
        if path and 'tensor-parallel-size 4' in text:
            out.append(Recipe(
                recipe_id='r.qwen3.a3.single.throughput',
                subject='qwen3',
                hardware=['A3', 'A2'],
                scenario_kind='high_throughput_single_instance',
                topology={'tp': 4, 'dp': 1, 'ep': False},
                feature_policy={'full_graph': True, 'async_scheduling': True, 'flashcomm1': True, 'mtp': 'not_applicable'},
                source_ref=self._rel(path),
                note='Doc-backed TP4 throughput route; can be scaled with DP on larger single-instance deployments.',
            ))
        return out

    def _deepseek_rows(self) -> list[EvidenceAtom]:
        path, text = self._read('docs/source/tutorials/models/DeepSeek-V3.1.md')
        if not path:
            return []
        out: list[EvidenceAtom] = []
        if 'require at least 2 Atlas 800 A2 (64G × 8)' in text:
            out.append(EvidenceAtom('e.deepseek.doc.1', 'deepseek-v3.1', 'minimum_a2_footprint', '2x_atlas800_a2_64g_x8', self._rel(path), 'repo_docs', 'constraint', 'Documented minimum A2 footprint is far above dual-card single-node.'))
        if 'dp4tp4' in text.lower() and 'instead of `dp2tp8`' in text.lower():
            out.append(EvidenceAtom('e.deepseek.doc.2', 'deepseek-v3.1', 'single_node_a3_shape', 'dp4tp4', self._rel(path), 'repo_docs', 'positive', 'Single-node A3 guidance recommends dp4tp4.'))
        return out

    def _hardware_310p_rows(self) -> list[EvidenceAtom]:
        path, text = self._read('docs/source/tutorials/hardwares/310p.md')
        if not path:
            return []
        out: list[EvidenceAtom] = []
        if 'only supports eager mode and the float16 data type' in text:
            out.append(EvidenceAtom('e.310p.doc.1', 'hardware:310p', 'execution_constraint', 'eager_float16_only', self._rel(path), 'repo_docs', 'constraint', '310P only supports eager + float16.'))
        if 'set `max-model-len` to a small value' in text.lower() or 'auto detection' in text.lower():
            out.append(EvidenceAtom('e.310p.doc.2', 'hardware:310p', 'serving_constraint', 'set_explicit_small_max_model_len', self._rel(path), 'repo_docs', 'constraint', '310P requires explicit conservative max-model-len.'))
        return out

    def _local_qwen35_rows(self) -> list[EvidenceAtom]:
        out: list[EvidenceAtom] = []
        local_paths = [
            self.upstream_vllm_root / 'vllm/model_executor/models/qwen3_5.py',
            self.repo_root / 'vllm_ascend/patch/worker/patch_qwen3_5.py',
        ]
        idx = 0
        for path in local_paths:
            if path.exists():
                idx += 1
                out.append(EvidenceAtom(f'e.qwen35.local.{idx}', 'qwen3.5', 'has_local_source_surface', 'true', self._rel(path), 'local_source', 'positive', 'Local source or patch exists for Qwen3.5.'))
        return out

    def _generic_quant_rows(self) -> list[EvidenceAtom]:
        path, text = self._read('docs/source/faqs.md')
        if not path:
            return []
        out: list[EvidenceAtom] = []
        if 'w8a8, w4a8, and w4a4 quantization methods are already supported' in text.lower():
            out.append(EvidenceAtom('e.quant.doc.1', 'quantization:general', 'supported_methods', 'w8a8,w4a8,w4a4', self._rel(path), 'repo_docs', 'positive', 'Generic quant methods are supported; this does not verify every model combination.'))
        return out

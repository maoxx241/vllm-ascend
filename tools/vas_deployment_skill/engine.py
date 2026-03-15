from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parser import parse_request
from .types import DeploymentResult, LaunchCandidate, ParsedRequest


class EvidenceStore:
    def __init__(self, build_dir: Path):
        self.evidence = [json.loads(line) for line in (build_dir / 'evidence.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        self.recipes = [json.loads(line) for line in (build_dir / 'recipes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]

    def find_evidence(self, subject_prefix: str | None = None, predicate: str | None = None) -> list[dict[str, Any]]:
        rows = self.evidence
        if subject_prefix is not None:
            rows = [r for r in rows if r['subject'].startswith(subject_prefix)]
        if predicate is not None:
            rows = [r for r in rows if r['predicate'] == predicate]
        return rows

    def find_recipes(self, subject_prefix: str | None = None, scenario: str | None = None) -> list[dict[str, Any]]:
        rows = self.recipes
        if subject_prefix is not None:
            rows = [r for r in rows if r['subject'].startswith(subject_prefix)]
        if scenario is not None:
            rows = [r for r in rows if r['scenario'] == scenario]
        return rows


class OpenWorldDeploymentEngine:
    def __init__(self, store: EvidenceStore):
        self.store = store

    def evaluate(self, request: ParsedRequest) -> DeploymentResult:
        # Strong alias/identity gate only when the name itself is unresolved.
        if request.model_name == 'owen3':
            return DeploymentResult(
                result_class='blocked.identity',
                resolved_subject={'model_name': 'owen3', 'status': 'unresolved_alias'},
                required_questions=['`owen3` 是打错字、别名，还是你们自定义模型？如果是自定义模型，请给权重路径或 config。'],
                why_not_exact=['名字相近不等于允许自动纠正；需要用户确认 typo 或给出本地证据。'],
                report_sections=[{'title': 'identity_gate', 'content': '名称未被自动纠正，先进入用户对齐。'}],
            )

        evidence = self._collect_relevant_evidence(request)

        # Hard blockers that can be decided after self-acquire but before any extra question.
        hard = self._hard_blockers(request, evidence)
        if hard is not None:
            return hard

        questions = self._missing_user_only_facts(request, evidence)
        if questions:
            return DeploymentResult(
                result_class='blocked.user_only_fact',
                resolved_subject=self._subject_view(request),
                evidence_summary=evidence,
                required_questions=questions,
                why_not_exact=['关键部署事实属于用户专属信息，系统无法自行从 repo/源码/文档中获取。'],
                report_sections=[{'title': 'self_acquire_done', 'content': '已先完成 repo/doc/source 取证，再进入最小化追问。'}],
            )

        if request.model_name == 'qwen3' and request.model_size_b == 32 and request.hardware == 'A3':
            return self._qwen3_32b_a3_single_instance(request, evidence)
        if request.model_name == 'qwen2-vl' and request.model_size_b == 72 and request.quantization == 'W4A4':
            return self._qwen2_vl_w4a4_existing_weights_candidate(request, evidence)
        if request.model_name == 'qwen3' and request.model_size_b == 560 and request.hardware == 'A3' and request.accepts_experimental:
            return self._qwen3_560b_candidate(request, evidence)
        if request.model_name == 'qwen3.5' and request.hardware == '310P':
            return self._qwen35_310p(request, evidence)

        return self._generic_candidate(request, evidence)

    def _collect_relevant_evidence(self, request: ParsedRequest) -> list[dict[str, Any]]:
        subjects: list[str] = []
        if request.model_name == 'qwen3.5':
            subjects += ['model_family:qwen3.5', 'hardware:310p']
        elif request.model_name == 'qwen2-vl':
            subjects += ['support_matrix:qwen2-vl', 'quantization:general', 'model_family:qwen2-vl']
        elif request.model_name == 'deepseek-v3.1':
            subjects += ['model:deepseek-v3.1', 'model:deepseek-v3.1-w8a8', 'support_matrix:deepseek-v3.1']
        elif request.model_name == 'qwen3':
            subjects += ['model_family:qwen3-dense', 'scenario:qwen3-32b-throughput-a3']
        results: list[dict[str, Any]] = []
        for subject in subjects:
            results.extend(self.store.find_evidence(subject_prefix=subject))
        seen = set()
        uniq = []
        for row in results:
            if row['evidence_id'] not in seen:
                uniq.append(row)
                seen.add(row['evidence_id'])
        return uniq

    def _missing_user_only_facts(self, request: ParsedRequest, evidence: list[dict[str, Any]]) -> list[str]:
        questions: list[str] = []
        if request.model_name is None:
            questions.append('请确认具体模型名，或给权重路径/config。')
        if request.hardware is None:
            questions.append('请确认硬件是 A2、A3、310P 还是其他规格。')
        if request.weight_path is None and request.local_weights is not False and request.model_name != 'qwen3.5':
            questions.append('请确认权重是否在本地；如果在本地，请给路径；如果不是本地，请说明是否直接使用远端权重。')
        if request.hardware == 'A2' and request.cards is None:
            questions.append('A2 单机规格不唯一，请补充卡数。')
        if request.model_name == 'qwen3' and request.model_size_b == 32:
            if request.objective == 'unknown':
                questions.append('请确认目标更偏高吞吐、低时延，还是平衡模式。')
            if request.max_context is None:
                questions.append('请确认最大上下文长度；这会影响是否进入长序列或保守内存策略。')
        if request.model_name == 'qwen2-vl' and request.model_size_b == 72 and request.quantization == 'W4A4' and request.has_existing_quantized_weights is None:
            questions.append('你是已经有 W4A4 权重，还是希望系统先做量化再部署？这会改变结论。')
        if request.model_name == 'qwen3.5' and request.hardware == '310P':
            if request.model_size_b is None:
                questions.append('请确认 `Qwen3.5` 的具体规模；310P 不能按家族名直接给脚本。')
            if request.weight_path is None:
                questions.append('请给具体权重路径或明确是远端权重。')
        return questions

    def _hard_blockers(self, request: ParsedRequest, evidence: list[dict[str, Any]]) -> DeploymentResult | None:
        if request.model_name == 'deepseek-v3.1' and request.hardware == 'A2' and request.cards is not None and request.cards < 16:
            est = self._estimate_weight_memory(request.model_size_b or 671.0, request.quantization)
            hw = self._hardware_capacity_gb('A2', request.cards)
            return DeploymentResult(
                result_class='blocked.resource',
                resolved_subject=self._subject_view(request),
                evidence_summary=evidence,
                why_not_exact=['当前硬件规模低于 repo 文档给出的已知 A2 部署下限。'],
                validation_checklist=['若坚持部署，请改用更大资源规格后再进入脚本合成。'],
                report_sections=[{'title': 'resource_blocker', 'content': 'DeepSeek-V3.1 的已知 A2 路径远高于双卡单机；当前请求应直接停止，不继续合成脚本。'}],
                derived_metrics={
                    'estimated_weight_memory_gb_lower_bound': est,
                    'visible_device_memory_gb': hw,
                    'weight_to_device_memory_ratio': round(est / hw, 2) if hw else None,
                },
            )
        if request.model_name == 'qwen2-vl' and request.model_size_b == 72 and request.quantization == 'W4A4' and request.has_existing_quantized_weights is False:
            return DeploymentResult(
                result_class='blocked.scope_mismatch',
                resolved_subject=self._subject_view(request),
                evidence_summary=evidence,
                why_not_exact=['当前请求不是纯 deployment；它包含量化生成与精度确认，超出 deployment skill 闭环。'],
                required_questions=['如果你已经有 W4A4 权重，可以继续按 deployment 流程；否则需要单独量化 flow。'],
                report_sections=[{'title': 'scope_mismatch', 'content': '当前 skill 只覆盖部署与脚本综合，不覆盖量化生成与精度验证。'}],
            )
        return None

    def _qwen3_32b_a3_single_instance(self, request: ParsedRequest, evidence: list[dict[str, Any]]) -> DeploymentResult:
        max_ctx = request.max_context or 40960
        long_seq = max_ctx >= 128_000
        avg_in = request.average_input_len or 3500
        avg_out = request.average_output_len or 1500
        compilation_json = '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
        additional_json = '{"weight_prefetch_config":{"enabled":true}}'
        dp_flag = '--data-parallel-size 2' if (request.cards or 8) >= 8 else ''
        launch = [
            LaunchCandidate(
                name='primary_tp4_dp2_throughput',
                script_kind='single_instance',
                command=(
                    f'vllm serve {request.weight_path} --served-model-name qwen3 --trust-remote-code '
                    f'--async-scheduling --distributed-executor-backend mp '
                    f'--tensor-parallel-size 4 {dp_flag} '
                    f'--max-model-len {max_ctx} --max-num-batched-tokens {max_ctx} '
                    f'--compilation-config \'{compilation_json}\' '
                    f'--additional-config \'{additional_json}\' '
                    f'--gpu-memory-utilization 0.9 --block-size 128'
                ).replace('  ', ' '),
                env={
                    'TASK_QUEUE_ENABLE': '1',
                    'HCCL_OP_EXPANSION_MODE': 'AIV',
                    'VLLM_ASCEND_ENABLE_FLASHCOMM1': '1',
                },
                risk_level='medium',
                rationale=['TP4 is the strongest documented core shape for Qwen3-32B throughput serving.', 'Single-instance 8-card serving prefers TP4 + DP2 over assuming TP8 is optimal.'],
            ),
            LaunchCandidate(
                name='tp4_dp2_graph_capture',
                script_kind='single_instance',
                command=(
                    f'vllm serve {request.weight_path} --served-model-name qwen3 --trust-remote-code '
                    f'--async-scheduling --distributed-executor-backend mp '
                    f'--tensor-parallel-size 4 {dp_flag} '
                    f'--max-model-len {max_ctx} --max-num-batched-tokens {max_ctx} '
                    f'--compilation-config \'{{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,12,16,20,24,32,48,60,64,68,72,76,80]}}\' '
                    f'--additional-config \'{{"weight_prefetch_config":{{"enabled":true}}}}\' '
                    f'--gpu-memory-utilization 0.9 --block-size 128'
                ).replace('  ', ' '),
                env={
                    'TASK_QUEUE_ENABLE': '1',
                    'HCCL_OP_EXPANSION_MODE': 'AIV',
                    'VLLM_ASCEND_ENABLE_FLASHCOMM1': '1',
                },
                risk_level='medium',
                rationale=['Nightly configs suggest graph capture sizes for feature-stack serving on A3.', 'Use this only after the primary script starts cleanly.'],
            ),
        ]
        why = ['仓中的最强直接证据更接近 TP4 单实例配方；当前 8 卡单实例方案是基于该配方叠加 DP 的派生。']
        if long_seq:
            why.append('当前上下文已经落入 128K 级别长序列，应考虑 pcp/dcp 等专门路径，而不是复用常规吞吐配方。')
        return DeploymentResult(
            result_class='compatible',
            resolved_subject=self._subject_view(request),
            evidence_summary=evidence,
            assumptions=['默认按单实例处理。', 'A3 单机默认 8 卡 / 16 芯，不额外追问卡数。'],
            why_not_exact=why,
            launch_candidates=launch,
            validation_checklist=[
                '确认权重 dtype/quantization 与命令一致。',
                '先做启动 smoke 和吞吐试跑。',
                '观察 TP4 切分下的 KV / 吞吐表现，再决定是否需要更激进调参。',
            ],
            report_sections=[
                {'title': 'intent_analysis', 'content': f'根据平均 {avg_in} 输入 / {avg_out} 输出、最大上下文 {max_ctx} 和高吞吐目标，优先按常规吞吐服务而不是 128K 长序列特殊路径处理。'},
                {'title': 'topology_reasoning', 'content': '采用 TP4 作为核心形状，再在单实例内叠加 DP2；不直接假设 TP8 是最优。'},
            ],
            derived_metrics={'avg_in': avg_in, 'avg_out': avg_out, 'max_context': max_ctx},
        )

    def _qwen2_vl_w4a4_existing_weights_candidate(self, request: ParsedRequest, evidence: list[dict[str, Any]]) -> DeploymentResult:
        return DeploymentResult(
            result_class='candidate',
            resolved_subject=self._subject_view(request),
            evidence_summary=evidence,
            assumptions=['用户已持有 W4A4 权重；当前流程只负责部署，不负责任何额外量化或精度背书。'],
            why_not_exact=['repo 中存在 Qwen2-VL 支持与通用 W4A4 能力，但没有该具体组合的明确已验证配方。'],
            launch_candidates=[
                LaunchCandidate(
                    name='qwen2_vl_72b_w4a4_candidate',
                    script_kind='single_instance',
                    command=f'vllm serve {request.weight_path} --max-model-len {request.max_context or 4096} --quantization ascend',
                    risk_level='high',
                    rationale=['Model support and quant capability are positive evidence.', 'The model × quantization combination is not closed as an officially verified recipe.'],
                )
            ],
            validation_checklist=['先确认模型可被当前工作区 registry 正确识别。', '先做视觉输入 smoke，再做功能/精度检查。', '将结果明确标记为未官方验证组合。'],
        )

    def _qwen35_310p(self, request: ParsedRequest, evidence: list[dict[str, Any]]) -> DeploymentResult:
        safe_len = request.max_context or 4096
        size_b = request.model_size_b or 0.0
        if size_b and size_b > 8.0:
            return DeploymentResult(
                result_class='blocked.resource',
                resolved_subject=self._subject_view(request),
                evidence_summary=evidence,
                why_not_exact=['310P 当前公开示例集中在小模型；对于你给定的 Qwen3.5 规模，这条路径风险过高，不建议继续合成脚本。'],
                report_sections=[{'title': 'resource_risk', 'content': '即使存在 Qwen3.5 本地实现与 patch，310P 侧公开示例仍集中在小规格模型；当前规模超出保守接入范围。'}],
                validation_checklist=['若坚持尝试，请换更强硬件，或先缩小模型规模后重试。'],
            )
        return DeploymentResult(
            result_class='candidate',
            resolved_subject=self._subject_view(request),
            evidence_summary=evidence,
            assumptions=['Qwen3.5 在本地 upstream mirror 与 ascend patch 空间均有实现/补丁痕迹。', '310P 只能走 eager + float16，并且需要显式保守 max-model-len。'],
            why_not_exact=['310P 路径仍属实验性；当前没有针对你给定具体规格的已验证单一配方。'],
            launch_candidates=[
                LaunchCandidate(
                    name='310p_conservative_candidate',
                    script_kind='single_instance',
                    command=f'vllm serve {request.weight_path} --dtype float16 --enforce-eager --max-model-len {safe_len}',
                    risk_level='high',
                    rationale=['Self-acquire found local source + patch surfaces for Qwen3.5.', '310P path remains heavily constrained and should start conservatively.'],
                )
            ],
            validation_checklist=['先做启动 smoke，再做短上下文功能验证。', '不要依赖 max-model-len 自动探测。'],
            report_sections=[{'title': 'open_world_reasoning', 'content': '没有因为 support matrix 缺项直接否定，而是先使用本地源码/补丁证据和 310P 约束做综合。'}],
        )

    def _qwen3_560b_candidate(self, request: ParsedRequest, evidence: list[dict[str, Any]]) -> DeploymentResult:
        return DeploymentResult(
            result_class='candidate',
            resolved_subject=self._subject_view(request),
            evidence_summary=evidence,
            assumptions=['用户明确接受实验性尝试。', '该模型是基于已知 Qwen3 大模型的自定义扩展版本。'],
            why_not_exact=['当前工作区没有你这个 560B 自定义版本的已验证配方；只能基于相邻家族与规模经验生成保守到激进的候选集。'],
            launch_candidates=[
                LaunchCandidate(name='conservative_no_graph', script_kind='single_instance', command=f'vllm serve {request.weight_path} --tensor-parallel-size 8 --disable-custom-all-reduce --enforce-eager --max-model-len 8192', risk_level='high', rationale=['Prefer minimal moving parts for first boot.']),
                LaunchCandidate(name='graph_moderate', script_kind='single_instance', command=f'vllm serve {request.weight_path} --tensor-parallel-size 8 --max-model-len 8192 --compilation-config \'{{"cudagraph_mode":"FULL_DECODE_ONLY"}}\'', risk_level='high', rationale=['Add graph mode only after conservative launch is understood.']),
                LaunchCandidate(name='graph_feature_stack', script_kind='single_instance', command=f'vllm serve {request.weight_path} --tensor-parallel-size 8 --max-model-len 8192 --compilation-config \'{{"cudagraph_mode":"FULL_DECODE_ONLY"}}\' --async-scheduling --enable-mlp-weight-prepack', risk_level='very_high', rationale=['Most aggressive stack for exploratory throughput experiments.']),
            ],
            validation_checklist=['先做 config/arch smoke：确认当前源码能加载该自定义 checkpoint。', '先跑单请求启动，再扩到批量。', '记录失败点，便于回退到更保守配置。'],
            report_sections=[{'title': 'risk_notice', 'content': '该候选集用于实验，不代表当前工作区已验证可用。'}],
        )

    def _generic_candidate(self, request: ParsedRequest, evidence: list[dict[str, Any]]) -> DeploymentResult:
        return DeploymentResult(
            result_class='candidate',
            resolved_subject=self._subject_view(request),
            evidence_summary=evidence,
            why_not_exact=['已有正证据但尚未匹配到更具体的已验证组合；返回实验性候选。'],
            launch_candidates=[
                LaunchCandidate(name='generic_candidate', script_kind='single_instance', command=self._generic_command(request), risk_level='medium', rationale=['Fallback candidate generated after self-acquire and blocker checks.'])
            ],
            validation_checklist=['确认模型配置能被当前 vLLM/vLLM-Ascend 工作区识别。', '先做启动 smoke，再做功能验证。'],
        )

    def _subject_view(self, request: ParsedRequest) -> dict[str, Any]:
        return {
            'model_name': request.model_name,
            'model_size_b': request.model_size_b,
            'hardware': request.hardware,
            'cards': request.cards,
            'quantization': request.quantization,
            'weight_path': request.weight_path,
            'objective': request.objective,
        }

    def _generic_command(self, request: ParsedRequest) -> str:
        path = request.weight_path or '/path/to/model'
        parts = ['vllm serve', path]
        if request.quantization:
            parts += ['--quantization', 'ascend' if request.quantization.startswith('W') else request.quantization.lower()]
        if request.hardware == '310P':
            parts += ['--dtype', 'float16', '--enforce-eager', '--max-model-len', str(request.max_context or 4096)]
        return ' '.join(parts)

    def _estimate_weight_memory(self, size_b: float, quantization: str | None) -> float:
        bytes_per_param = 2.0  # bf16/fp16 lower-bound proxy
        if quantization and quantization.upper().startswith('W8'):
            bytes_per_param = 1.0
        elif quantization and quantization.upper().startswith('W4'):
            bytes_per_param = 0.5
        return round(size_b * bytes_per_param, 2)

    def _hardware_capacity_gb(self, hardware: str | None, cards: int | None) -> int | None:
        if hardware in {'A2', 'A3'} and cards:
            return 64 * cards
        return None


def evaluate_text(build_dir: Path, text: str, *, overrides: dict[str, Any] | None = None) -> DeploymentResult:
    store = EvidenceStore(build_dir)
    engine = OpenWorldDeploymentEngine(store)
    req = parse_request(text, overrides=overrides)
    return engine.evaluate(req)

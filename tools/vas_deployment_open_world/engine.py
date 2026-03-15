from __future__ import annotations

from .models import RequestFacts, DeploymentResult, EvidenceAtom, Recipe


class DeploymentEngine:
    def __init__(self, evidence: list[EvidenceAtom], recipes: list[Recipe]):
        self.evidence = evidence
        self.recipes = recipes

    def _ev(self, subject: str) -> list[EvidenceAtom]:
        return [e for e in self.evidence if e.subject == subject or e.subject.startswith(subject)]

    def _recipes(self, subject: str, scenario_kind: str | None = None) -> list[Recipe]:
        rows = [r for r in self.recipes if r.subject == subject]
        if scenario_kind:
            rows = [r for r in rows if r.scenario_kind == scenario_kind]
        return rows

    def _scenario_kind(self, req: RequestFacts) -> str | None:
        if req.tpot_ms is None:
            return None
        if req.tpot_ms <= 30:
            return 'low_latency_single_instance'
        if req.tpot_ms >= 50:
            return 'high_throughput_single_instance'
        return None

    def _scenario_view(self, req: RequestFacts) -> dict:
        return {
            'model': req.model_variant or req.model_family,
            'hardware': req.hardware,
            'input_length': {
                'avg_input_tokens': req.avg_input_tokens,
                'avg_output_tokens': req.avg_output_tokens,
                'max_context_tokens': req.max_context_tokens,
            },
            'sla': {'tpot_ms': req.tpot_ms},
            'deployment_form': req.deployment_form,
        }

    def _subject_view(self, req: RequestFacts) -> dict:
        return {
            'model_family': req.model_family,
            'model_variant': req.model_variant,
            'model_size_b': req.model_size_b,
            'hardware': req.hardware,
            'cards': req.cards,
            'weight_path': req.weight_path,
            'quantization': req.quantization,
        }

    def _alignment_questions(self, req: RequestFacts) -> list[str]:
        questions: list[str] = []
        if req.alias_suspect:
            questions.append(f'`{req.alias_suspect}` 是打错字、内部别名，还是自定义模型？请确认模型名或给本地权重路径/config。')
            return questions
        if not req.model_family:
            questions.append('请确认模型名，或者直接给本地权重路径/config。')
        if not req.hardware:
            questions.append('请确认硬件型号，例如 A2、A3、310P。')
        if req.hardware == 'A2' and req.cards is None:
            questions.append('A2 单机规格不唯一，请补充卡数。')
        if not req.weight_path:
            questions.append('请确认权重路径，或给远端模型标识。')
        if not req.quantization:
            questions.append('请确认当前权重是 BF16/FP16 还是量化权重（如 W8A8/W4A4）。')
        if req.avg_input_tokens is None or req.avg_output_tokens is None:
            questions.append('请给平均输入长度和平均输出长度。')
        if req.max_context_tokens is None:
            questions.append('请给最大上下文长度。')
        if req.tpot_ms is None:
            questions.append('请给 TPOT/SLA 目标（毫秒）。场景只基于 TPOT 判断。')
        elif 30 < req.tpot_ms < 50:
            questions.append('当前 TPOT 介于 30ms 和 50ms 之间。请明确是要更紧的低时延还是放宽到高吞吐。')
        if req.model_family == 'qwen2-vl' and req.quantization == 'W4A4' and req.existing_quantized_weights is None:
            questions.append('你是已经有 W4A4 权重，还是需要先做量化？这会改变结论。')
        if req.model_family == 'qwen3.5' and req.hardware == '310P' and req.model_size_b is None:
            questions.append('请确认 Qwen3.5 的具体规模/权重路径。310P 不能只按家族名给脚本。')
        return questions

    def evaluate(self, req: RequestFacts) -> DeploymentResult:
        questions = self._alignment_questions(req)
        if questions:
            return DeploymentResult(
                result_class='needs_alignment',
                scenario=self._scenario_view(req),
                resolved_subject=self._subject_view(req),
                evidence_summary=[e.to_dict() for e in self._relevant_evidence(req)],
                required_questions=questions,
                why=['系统已经先做自取证；当前缺的是用户专属高影响事实。'],
            )

        blocked = self._hard_blockers(req)
        if blocked is not None:
            return blocked

        if req.model_family == 'glm4.x' and req.hardware == 'A3' and req.deployment_form == 'single_instance':
            return self._glm_single_a3(req)
        if req.model_family == 'qwen3' and req.model_size_b == 32 and req.hardware == 'A3' and req.deployment_form == 'single_instance':
            return self._qwen3_32b_a3(req)
        if req.model_family == 'qwen2-vl' and req.model_size_b == 72 and req.quantization == 'W4A4' and req.existing_quantized_weights:
            return self._qwen2_vl_w4a4_existing(req)
        if req.model_family == 'qwen3.5' and req.hardware == '310P':
            return self._qwen35_310p(req)
        if req.model_family == 'qwen3' and req.model_size_b == 560 and req.hardware == 'A3' and req.accepts_candidate:
            return self._qwen3_560b_candidate(req)

        return DeploymentResult(
            result_class='candidate',
            scenario=self._scenario_view(req),
            resolved_subject=self._subject_view(req),
            evidence_summary=[e.to_dict() for e in self._relevant_evidence(req)],
            assumptions=['已有正证据，但当前没有更具体的已验证组合。'],
            why=['返回实验性候选，而不是假装 exact_verified。'],
            launch_candidates=[self._generic_launch_candidate(req)],
            validation_checklist=['先做 smoke 启动', '确认 feature 开关与权重相匹配', '确认服务可正常返回'],
        )

    def _relevant_evidence(self, req: RequestFacts) -> list[EvidenceAtom]:
        subjects = []
        if req.model_family:
            subjects.append(req.model_family)
        if req.hardware == '310P':
            subjects.append('hardware:310p')
        if req.model_family == 'qwen2-vl' and req.quantization == 'W4A4':
            subjects.append('quantization:general')
        found = []
        seen = set()
        for s in subjects:
            for e in self._ev(s):
                if e.evidence_id not in seen:
                    seen.add(e.evidence_id)
                    found.append(e)
        return found

    def _hard_blockers(self, req: RequestFacts) -> DeploymentResult | None:
        if req.model_family == 'deepseek-v3.1' and req.hardware == 'A2' and req.cards is not None and req.cards <= 2:
            return DeploymentResult(
                result_class='blocked.resource',
                scenario=self._scenario_view(req),
                resolved_subject=self._subject_view(req),
                evidence_summary=[e.to_dict() for e in self._relevant_evidence(req)],
                blockers=['A2 双卡资源规模低于已知 DeepSeek-V3.1 路线的资源下界。'],
                why=['这是资源不可能，不是文档闭世界判断。'],
                validation_checklist=['若要继续，请改为更大资源规格后重做合成。'],
            )
        if req.model_family == 'qwen2-vl' and req.model_size_b == 72 and req.quantization == 'W4A4' and req.existing_quantized_weights is False:
            return DeploymentResult(
                result_class='blocked.conflict',
                scenario=self._scenario_view(req),
                resolved_subject=self._subject_view(req),
                evidence_summary=[e.to_dict() for e in self._relevant_evidence(req)],
                blockers=['当前请求包含“量化生成 + 部署”，不属于纯 deployment 闭环。'],
                why=['如果已经有 W4A4 权重，可以进入 deployment；否则需要单独量化流程。'],
            )
        return None

    def _glm_single_a3(self, req: RequestFacts) -> DeploymentResult:
        kind = self._scenario_kind(req)
        ev = [e.to_dict() for e in self._relevant_evidence(req)]
        if kind == 'low_latency_single_instance':
            recipe = self._recipes('glm4.x', 'low_latency_single_instance')[0]
            return DeploymentResult(
                result_class='exact_verified',
                scenario=self._scenario_view(req),
                resolved_subject=self._subject_view(req),
                evidence_summary=ev,
                assumptions=['A3 单机默认单实例，硬件形状默认为 8 卡 / 16 芯。', 'MTP / full graph 作为正常 feature policy 处理。'],
                why=['GLM4.x 教程存在单机低时延路线，且当前场景由 TPOT<=30ms 闭合。'],
                launch_candidates=[self._render_recipe_candidate(req, recipe)],
                validation_checklist=['确认权重包含或兼容 MTP 路径', '确认 TPOT 目标确实是 <=30ms', '验证启动后 TTFT/TPOT'],
            )
        recipe = self._recipes('glm4.x', 'high_throughput_single_instance')
        if recipe:
            return DeploymentResult(
                result_class='compatible',
                scenario=self._scenario_view(req),
                resolved_subject=self._subject_view(req),
                evidence_summary=ev,
                assumptions=['TP8+DP2 来自 repo nightly 配置，是强证据但不是同等级教程锚点。', '高吞吐场景由 TPOT>=50ms 闭合。'],
                why=['GLM4.x 在 repo 中存在 TP8+DP2 单机路线；当前 SLA 更接近高吞吐。'],
                launch_candidates=[self._render_recipe_candidate(req, recipe[0])],
                validation_checklist=['验证 TPOT 与吞吐是否满足预期', '验证 expert parallel 行为与显存利用'],
            )
        return DeploymentResult(
            result_class='candidate',
            scenario=self._scenario_view(req),
            resolved_subject=self._subject_view(req),
            evidence_summary=ev,
            assumptions=['当前只有高层支持证据，缺少更强拓扑锚点。'],
            why=['返回候选而不是强行给 exact。'],
            launch_candidates=[self._generic_launch_candidate(req)],
            validation_checklist=['先做启动 smoke'],
        )

    def _qwen3_32b_a3(self, req: RequestFacts) -> DeploymentResult:
        ev = [e.to_dict() for e in self._relevant_evidence(req)]
        tp = 4
        dp = 2 if (req.cards or 8) >= 8 else 1
        command = self._build_command(req, tp=tp, dp=dp, ep=False, full_graph=True, async_scheduling=True, flashcomm1=True, mtp=False)
        return DeploymentResult(
            result_class='compatible',
            scenario=self._scenario_view(req),
            resolved_subject=self._subject_view(req),
            evidence_summary=ev,
            assumptions=['Qwen3 Dense tutorial gives TP4 throughput route; on A3 单机 8 卡默认单实例时可放大为 DP2+TP4。'],
            why=['当前场景是 A3 + 单实例 + 高吞吐；TP4 结合 DP 扩展比直接 TP8 更符合 repo 和经验约束。'],
            launch_candidates=[{
                'name': 'primary',
                'topology': {'tp': tp, 'dp': dp, 'ep': False},
                'command': command,
                'script_kind': 'single_instance',
                'features': {'full_graph': True, 'async_scheduling': True, 'flashcomm1': True},
            }],
            validation_checklist=['验证 max-model-len 与 lengths 匹配', '验证吞吐与 TPOT', '检查显存利用是否稳定'],
        )

    def _qwen2_vl_w4a4_existing(self, req: RequestFacts) -> DeploymentResult:
        ev = [e.to_dict() for e in self._relevant_evidence(req)]
        command = self._build_command(req, tp=max(1, req.cards or 1), dp=1, ep=False, full_graph=True, async_scheduling=True, flashcomm1=False, mtp=False)
        return DeploymentResult(
            result_class='candidate',
            scenario=self._scenario_view(req),
            resolved_subject=self._subject_view(req),
            evidence_summary=ev,
            assumptions=['Qwen2-VL family is supported and generic W4A4 capability exists.', 'This exact combination is not treated as officially verified.'],
            why=['已有量化权重时，可以给实验性部署候选；不能因为组合未在教程中闭合就直接 hard-block。'],
            launch_candidates=[{
                'name': 'candidate_existing_weights',
                'topology': {'tp': max(1, req.cards or 1), 'dp': 1, 'ep': False},
                'command': command,
                'script_kind': 'single_instance',
                'features': {'full_graph': True, 'async_scheduling': True},
            }],
            validation_checklist=['先做图像/多模态 smoke', '做基础精度冒烟，不要默认官方验证等级'],
        )

    def _qwen35_310p(self, req: RequestFacts) -> DeploymentResult:
        ev = [e.to_dict() for e in self._relevant_evidence(req)]
        safe_len = min(req.max_context_tokens or 4096, 4096)
        command = self._build_command(req, tp=1, dp=1, ep=False, full_graph=False, async_scheduling=False, flashcomm1=False, mtp=False, force_float16=True, force_eager=True, max_model_len=safe_len)
        return DeploymentResult(
            result_class='candidate',
            scenario=self._scenario_view(req),
            resolved_subject=self._subject_view(req),
            evidence_summary=ev,
            assumptions=['Qwen3.5 has local source surfaces in the workspace.', '310P imposes eager+float16 and conservative max-model-len constraints.'],
            why=['This is not blocked by a KB omission; it is an exploratory candidate under 310P hard constraints.'],
            launch_candidates=[{
                'name': 'conservative_310p_candidate',
                'topology': {'tp': 1, 'dp': 1, 'ep': False},
                'command': command,
                'script_kind': 'single_instance',
                'features': {'eager': True, 'float16_only': True},
            }],
            validation_checklist=['确认模型规模是否适合 310P', '确认 max-model-len 保守设置', '先做最小 smoke'],
        )

    def _qwen3_560b_candidate(self, req: RequestFacts) -> DeploymentResult:
        ev = [e.to_dict() for e in self._relevant_evidence(req)]
        candidates = []
        for name, full_graph, ep in [
            ('conservative', False, False),
            ('graph_default', True, False),
            ('graph_feature_stack', True, True),
        ]:
            candidates.append({
                'name': name,
                'topology': {'tp': 8, 'dp': 1, 'ep': ep},
                'command': self._build_command(req, tp=8, dp=1, ep=ep, full_graph=full_graph, async_scheduling=True, flashcomm1=True, mtp=True),
                'script_kind': 'single_instance',
                'features': {'full_graph': full_graph, 'async_scheduling': True, 'mtp': True, 'expert_parallel': ep},
            })
        return DeploymentResult(
            result_class='candidate',
            scenario=self._scenario_view(req),
            resolved_subject=self._subject_view(req),
            evidence_summary=ev,
            assumptions=['This is a custom expanded model, so family inheritance is partial rather than exact.'],
            why=['User accepts experimental deployment; return a conservative-to-aggressive candidate ladder.'],
            launch_candidates=candidates,
            validation_checklist=['从 conservative 开始启动', '逐步升级到 graph / feature stack', '每一步都验证能否加载和返回'],
        )

    def _render_recipe_candidate(self, req: RequestFacts, recipe: Recipe) -> dict:
        topo = recipe.topology
        command = self._build_command(
            req,
            tp=topo.get('tp', 1),
            dp=topo.get('dp', 1),
            ep=bool(topo.get('ep', False)),
            full_graph=bool(recipe.feature_policy.get('full_graph', False)),
            async_scheduling=bool(recipe.feature_policy.get('async_scheduling', False)),
            flashcomm1=bool(recipe.feature_policy.get('flashcomm1', False)),
            mtp=recipe.feature_policy.get('mtp') == 'normal_if_weight_support_present',
        )
        return {
            'name': 'primary',
            'topology': topo,
            'command': command,
            'script_kind': 'single_instance',
            'features': recipe.feature_policy,
        }

    def _generic_launch_candidate(self, req: RequestFacts) -> dict:
        return {
            'name': 'generic_candidate',
            'topology': {'tp': max(1, req.cards or 1), 'dp': 1, 'ep': False},
            'command': self._build_command(req, tp=max(1, req.cards or 1), dp=1, ep=False, full_graph=True, async_scheduling=True, flashcomm1=False, mtp=False),
            'script_kind': 'single_instance',
            'features': {'full_graph': True, 'async_scheduling': True},
        }

    def _build_command(self, req: RequestFacts, *, tp: int, dp: int, ep: bool, full_graph: bool, async_scheduling: bool, flashcomm1: bool, mtp: bool, force_float16: bool = False, force_eager: bool = False, max_model_len: int | None = None) -> str:
        model_ref = req.weight_path or (req.model_variant or req.model_family or 'MODEL_PATH')
        parts = ['vllm', 'serve', model_ref]
        parts += ['--served-model-name', (req.model_variant or req.model_family or 'model').replace('.', '').replace('-', '_')]
        parts += ['--trust-remote-code']
        parts += ['--tensor-parallel-size', str(tp)]
        if dp > 1:
            parts += ['--data-parallel-size', str(dp)]
        if ep:
            parts += ['--enable-expert-parallel']
        if req.quantization and req.quantization.startswith('W'):
            parts += ['--quantization', 'ascend']
        if force_float16:
            parts += ['--dtype', 'float16']
        if async_scheduling:
            parts += ['--async-scheduling']
        if force_eager:
            parts += ['# eager_only']
        if max_model_len or req.max_context_tokens:
            parts += ['--max-model-len', str(max_model_len or req.max_context_tokens)]
        if req.avg_input_tokens and req.avg_output_tokens:
            max_batched = max(req.avg_input_tokens + req.avg_output_tokens, 4096) * max(dp, 1)
            parts += ['--max-num-batched-tokens', str(max_batched)]
        if full_graph and not force_eager:
            parts += ['--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}']
        if flashcomm1:
            parts += ['# env: VLLM_ASCEND_ENABLE_FLASHCOMM1=1']
        if mtp:
            parts += ['--speculative-config', '{"num_speculative_tokens":3,"method":"mtp"}']
        return ' '.join(parts)

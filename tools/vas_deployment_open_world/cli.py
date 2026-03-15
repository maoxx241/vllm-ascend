from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.vas_deployment_open_world.parser import parse_request
    from tools.vas_deployment_open_world.compiler import EvidenceCompiler
    from tools.vas_deployment_open_world.engine import DeploymentEngine
    from tools.vas_deployment_open_world.bundle import write_bundle
    from tools.vas_deployment_open_world.workspace import write_workspace
else:
    from .parser import parse_request
    from .compiler import EvidenceCompiler
    from .engine import DeploymentEngine
    from .bundle import write_bundle
    from .workspace import write_workspace


def run_assistant(repo_root: str, request_text: str, out: str | None = None) -> dict:
    req = parse_request(request_text)
    compiler = EvidenceCompiler(repo_root)
    evidence, recipes = compiler.compile()
    engine = DeploymentEngine(evidence, recipes)
    result = engine.evaluate(req)
    payload = result.to_dict()
    if out:
        out_path = Path(out)
        write_workspace(out_path, req, result)
        write_bundle(out_path, result)
        payload['bundle_dir'] = str(out_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(prog='vas_deployment_open_world')
    sub = parser.add_subparsers(dest='cmd', required=True)

    pa = sub.add_parser('assistant')
    pa.add_argument('--repo-root', required=True)
    pa.add_argument('--request', required=True)
    pa.add_argument('--out')

    ps = sub.add_parser('synthesize')
    ps.add_argument('--repo-root', required=True)
    ps.add_argument('--request', required=True)
    ps.add_argument('--out', required=True)

    pb = sub.add_parser('build')
    pb.add_argument('--repo-root', required=True)

    args = parser.parse_args()
    if args.cmd in {'assistant', 'synthesize'}:
        payload = run_assistant(args.repo_root, args.request, getattr(args, 'out', None))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == 'build':
        compiler = EvidenceCompiler(args.repo_root)
        evidence, recipes = compiler.compile()
        print(json.dumps({'evidence_count': len(evidence), 'recipe_count': len(recipes)}, ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())

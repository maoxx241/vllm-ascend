from __future__ import annotations

import os
import sys

# Support the acceptance-documented invocation:
#   python tools/vas_deployment_skill/cli.py ...
# When executed that way, sys.path[0] points at this package directory and can
# shadow the stdlib `types` module with our local `types.py`.
if __package__ in {None, ""}:
    _PKG_DIR = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == _PKG_DIR:
        sys.path.pop(0)
    _PARENT = os.path.dirname(_PKG_DIR)
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    from vas_deployment_skill.assistant_entry import vllm_ascend_assistant
    from vas_deployment_skill.compiler import OpenWorldCompiler
    from vas_deployment_skill.engine import evaluate_text
    from vas_deployment_skill.renderer import write_bundle
else:
    from .assistant_entry import vllm_ascend_assistant
    from .compiler import OpenWorldCompiler
    from .engine import evaluate_text
    from .renderer import write_bundle


def _json_or_none(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    return json.loads(text)


def build_cmd(args: argparse.Namespace) -> int:
    summary = OpenWorldCompiler(Path(args.workspace_root)).compile(Path(args.out_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def eval_cmd(args: argparse.Namespace) -> int:
    result = evaluate_text(Path(args.build_dir), args.text, overrides=_json_or_none(args.overrides))
    bundle_dir = Path(args.bundle_dir)
    manifest = write_bundle(bundle_dir, result, request_text=args.text, case_id=args.case_id)
    print(json.dumps({'result': result.to_dict(), 'bundle_manifest': manifest}, ensure_ascii=False, indent=2))
    return 0


def entry_cmd(args: argparse.Namespace) -> int:
    payload = vllm_ascend_assistant(
        workspace_root=Path(args.workspace_root),
        build_dir=Path(args.build_dir),
        text=args.text,
        overrides=_json_or_none(args.overrides),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def case_cmd(args: argparse.Namespace) -> int:
    case = json.loads(Path(args.case_file).read_text(encoding='utf-8'))
    if args.entry:
        payload = vllm_ascend_assistant(
            workspace_root=Path(args.workspace_root),
            build_dir=Path(args.build_dir),
            text=case['text'],
            overrides=case.get('overrides'),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    result = evaluate_text(Path(args.build_dir), case['text'], overrides=case.get('overrides'))
    bundle_dir = Path(args.bundle_dir)
    manifest = write_bundle(bundle_dir, result, request_text=case['text'], case_id=case.get('case_id'))
    print(json.dumps({'result': result.to_dict(), 'bundle_manifest': manifest}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog='vas-deployment-skill')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('build', help='compile repo evidence into a build directory')
    p.add_argument('--workspace-root', required=True)
    p.add_argument('--out-dir', required=True)
    p.set_defaults(func=build_cmd)

    p = sub.add_parser('eval', help='evaluate a deployment request and write a bundle')
    p.add_argument('--build-dir', required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--bundle-dir', required=True)
    p.add_argument('--case-id', default='adhoc')
    p.add_argument('--overrides')
    p.set_defaults(func=eval_cmd)

    p = sub.add_parser('entry', help='evaluate through vllm-ascend-assistant entry')
    p.add_argument('--workspace-root', required=True)
    p.add_argument('--build-dir', required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--overrides')
    p.set_defaults(func=entry_cmd)

    p = sub.add_parser('run-case', help='run a case file either directly or via entry')
    p.add_argument('--workspace-root', required=True)
    p.add_argument('--build-dir', required=True)
    p.add_argument('--case-file', required=True)
    p.add_argument('--bundle-dir', required=False, default='bundle_out')
    p.add_argument('--entry', action='store_true')
    p.set_defaults(func=case_cmd)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())

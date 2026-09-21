"""GraphRAG local-query subprocess entry point with provider thinking disabled."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .backends.graphrag import (
    _GraphRagAsyncRuntime,
    _json_safe_context,
    _run_graphrag_local_search,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--community-level", type=int, required=True)
    parser.add_argument("--response-type", required=True)
    parser.add_argument("--method", choices=["local"], default="local")
    parser.add_argument("--query", required=True)
    parser.add_argument("--disable-api-thinking", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = _GraphRagAsyncRuntime()
    try:
        response, context_data = _run_graphrag_local_search(
            config_filepath=None,
            data_dir=Path(args.data),
            root_dir=Path(args.root),
            community_level=args.community_level,
            response_type=args.response_type,
            streaming=False,
            query=args.query,
            verbose=False,
            disable_api_thinking=args.disable_api_thinking,
            runtime=runtime,
        )
        log_path = os.getenv("GRAPHRAG_LOG_PATH")
        if log_path:
            context_path = Path(log_path)
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context_path.write_text(
                json.dumps(
                    {
                        "capture_status": "captured",
                        "query_method": args.method,
                        "query": args.query,
                        "tables": _json_safe_context(context_data),
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        print(str(response))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

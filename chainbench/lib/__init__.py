"""Shared utilities for ChainBench pipeline and baseline scripts."""

from .config import default_workspace_root, load_json, relative_to_workspace, resolve_path
from .cli import (
    LOG_LEVEL_CHOICES,
    LANGUAGE_CHOICES,
    add_language_filter_argument,
    add_limit_argument,
    add_log_level_argument,
    load_rows_with_filters,
    resolve_config_argument,
    resolve_worker_count,
)
from .chain_keys import (
    OPERATOR_MULTISET_KEY_CACHE_FIELD,
    operator_multiset_key,
    operator_signature_sequence,
    parse_operator_params,
    parse_operator_seq,
    path_endpoint_key,
)
from .execution import run_bounded_tasks
from .io import load_csv_rows, load_jsonl, write_csv, write_jsonl
from .conda import (
    conda_run_python_command,
    is_conda_prefix_path,
    list_conda_env_names,
    normalize_conda_env_ref,
    require_conda_envs,
)
from .proc import run_command, run_command_streaming
from .audio import ffprobe_audio
from .logging import setup_logging, get_logger, clean_stream_line, format_elapsed
from .runtime_stats import runtime_snapshot
from .structural_metadata import (
    OPERATOR_MULTISET_FIELD,
    OPERATOR_SUBSTITUTION_DETAIL_FIELD,
    OPERATOR_SUBSTITUTION_GROUP_FIELD,
    ORDER_SWAP_GROUP_FIELD,
    PARAMETER_PERTURBATION_AXIS_FIELD,
    PARAMETER_PERTURBATION_GROUP_FIELD,
    PATH_ENDPOINT_FIELD,
    PATH_GROUP_FIELD,
    PATH_STEP_FIELD,
    annotate_structural_group_fields,
    lineage_bucket_key,
    stable_row_token,
)
from .summary import json_dumps, print_json, utc_now_iso, write_timestamped_json

__all__ = [
    "default_workspace_root",
    "load_json",
    "resolve_path",
    "relative_to_workspace",
    "LOG_LEVEL_CHOICES",
    "LANGUAGE_CHOICES",
    "add_log_level_argument",
    "add_language_filter_argument",
    "add_limit_argument",
    "load_rows_with_filters",
    "resolve_config_argument",
    "resolve_worker_count",
    "load_csv_rows",
    "write_csv",
    "load_jsonl",
    "write_jsonl",
    "parse_operator_seq",
    "parse_operator_params",
    "operator_signature_sequence",
    "operator_multiset_key",
    "path_endpoint_key",
    "OPERATOR_MULTISET_KEY_CACHE_FIELD",
    "run_bounded_tasks",
    "conda_run_python_command",
    "is_conda_prefix_path",
    "list_conda_env_names",
    "normalize_conda_env_ref",
    "require_conda_envs",
    "run_command",
    "run_command_streaming",
    "ffprobe_audio",
    "setup_logging",
    "get_logger",
    "clean_stream_line",
    "format_elapsed",
    "runtime_snapshot",
    "OPERATOR_MULTISET_FIELD",
    "OPERATOR_SUBSTITUTION_GROUP_FIELD",
    "OPERATOR_SUBSTITUTION_DETAIL_FIELD",
    "PARAMETER_PERTURBATION_GROUP_FIELD",
    "PARAMETER_PERTURBATION_AXIS_FIELD",
    "ORDER_SWAP_GROUP_FIELD",
    "PATH_GROUP_FIELD",
    "PATH_ENDPOINT_FIELD",
    "PATH_STEP_FIELD",
    "annotate_structural_group_fields",
    "stable_row_token",
    "lineage_bucket_key",
    "utc_now_iso",
    "json_dumps",
    "print_json",
    "write_timestamped_json",
]

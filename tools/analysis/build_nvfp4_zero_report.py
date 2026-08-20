#!/usr/bin/env python3
"""Build a canonical Data Analytics report artifact from NVFP4 zero stats."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        DROP TABLE IF EXISTS zero_summary;
        DROP TABLE IF EXISTS block_zero_histogram;

        CREATE TABLE zero_summary (
            scope_type TEXT NOT NULL,
            scope TEXT NOT NULL,
            tensor_count INTEGER NOT NULL,
            blocks INTEGER NOT NULL,
            logical_values INTEGER NOT NULL,
            zero_values INTEGER NOT NULL,
            zero_rate REAL NOT NULL,
            positive_zero_codes INTEGER NOT NULL,
            negative_zero_codes INTEGER NOT NULL,
            zero_free_blocks INTEGER NOT NULL,
            blocks_with_any_zero INTEGER NOT NULL,
            any_zero_rate REAL NOT NULL,
            all_zero_blocks INTEGER NOT NULL,
            mean_zeros_per_block REAL NOT NULL,
            PRIMARY KEY (scope_type, scope)
        );

        CREATE TABLE block_zero_histogram (
            scope_type TEXT NOT NULL,
            scope TEXT NOT NULL,
            zero_count INTEGER NOT NULL,
            block_count INTEGER NOT NULL,
            block_share REAL NOT NULL,
            zero_values_contributed INTEGER NOT NULL,
            PRIMARY KEY (scope_type, scope, zero_count)
        );
        """
    )
    return connection


def _all_records(stats: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        stats["overall"],
        *stats["projections"],
        *stats["layers"],
        *stats["tensors"],
    ]


def _populate_database(
    connection: sqlite3.Connection, stats: dict[str, Any]
) -> None:
    summary_sql = """
        INSERT INTO zero_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    histogram_sql = """
        INSERT INTO block_zero_histogram VALUES (?, ?, ?, ?, ?, ?)
    """
    for record in _all_records(stats):
        blocks = int(record["blocks"])
        zero_values = int(record["dequant_zero_values"])
        connection.execute(
            summary_sql,
            (
                record["scope_type"],
                record["scope"],
                int(record["tensor_count"]),
                blocks,
                int(record["logical_values"]),
                zero_values,
                zero_values / int(record["logical_values"]),
                int(record["positive_zero_codes"]),
                int(record["negative_zero_codes"]),
                int(record["zero_free_blocks"]),
                int(record["blocks_with_any_zero"]),
                int(record["blocks_with_any_zero"]) / blocks,
                int(record["all_zero_blocks"]),
                float(record["mean_zeros_per_block"]),
            ),
        )
        histogram = record["dequant_block_histogram_0_to_16"]
        for zero_count, block_count in enumerate(histogram):
            connection.execute(
                histogram_sql,
                (
                    record["scope_type"],
                    record["scope"],
                    zero_count,
                    int(block_count),
                    int(block_count) / blocks,
                    zero_count * int(block_count),
                ),
            )
    connection.commit()


def _query(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql).fetchall()]


def _source(
    source_id: str,
    label: str,
    database_path: str,
    sql: str,
    description: str,
    executed_at: str,
    metric_definitions: list[str],
    filters: list[str],
    tables_used: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": database_path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": executed_at,
            "tables_used": tables_used,
            "filters": filters,
            "metric_definitions": metric_definitions,
        },
    }


def _artifact(
    stats: dict[str, Any], database_path: str, connection: sqlite3.Connection
) -> dict[str, Any]:
    generated_at = stats["source"]["scanned_at"]
    overall_sql = """SELECT blocks, logical_values, zero_values, zero_rate,
       positive_zero_codes, negative_zero_codes, blocks_with_any_zero,
       any_zero_rate, all_zero_blocks, mean_zeros_per_block
FROM zero_summary
WHERE scope_type = 'overall' AND scope = 'all_nvfp4'"""
    histogram_sql = """SELECT zero_count, block_count, block_share,
       zero_values_contributed
FROM block_zero_histogram
WHERE scope_type = 'overall' AND scope = 'all_nvfp4'
ORDER BY zero_count"""
    projection_sql = """SELECT scope AS projection, blocks, zero_values, zero_rate,
       blocks_with_any_zero, any_zero_rate, all_zero_blocks
FROM zero_summary
WHERE scope_type = 'projection'
ORDER BY CASE scope
    WHEN 'gate' THEN 1 WHEN 'up' THEN 2 WHEN 'down' THEN 3 ELSE 4 END"""
    layer_sql = """SELECT scope AS layer, blocks, zero_values, zero_rate,
       blocks_with_any_zero, any_zero_rate, all_zero_blocks
FROM zero_summary
WHERE scope_type = 'layer' AND scope LIKE 'layer_%'
ORDER BY zero_rate DESC"""

    overall_rows = _query(connection, overall_sql)
    histogram_rows = _query(connection, histogram_sql)
    projection_rows = _query(connection, projection_sql)
    layer_rows = _query(connection, layer_sql)
    if len(overall_rows) != 1 or len(histogram_rows) != 17:
        raise AssertionError("report SQL did not return the expected overall grain")
    if len(projection_rows) != 4 or len(layer_rows) != 64:
        raise AssertionError("report SQL did not return the expected segment grain")

    metric_definitions = [
        "Zero value: strict FP32 dequantized value equals 0.0; for this checkpoint it is exactly E2M1 code 0x0 or 0x8.",
        "Zero rate = dequantized zero values / all logical NVFP4 values.",
        "Any-zero block rate = size-16 quantization blocks with at least one zero / all size-16 blocks.",
    ]
    common_filters = [
        "Include only targets marked W4A16_NVFP4 in hf_quant_config.json.",
        "Exclude FP8 attention, vision, BF16, and MTP tensors.",
        "Checkpoint revision 0893e1606ff3d5f97a441f405d5fc541a6bdf404.",
    ]
    sources = [
        _source(
            "overall_sql",
            "NVFP4 overall zero summary",
            database_path,
            overall_sql,
            "Returns the reviewed checkpoint-wide zero metrics used by the report cards and summary.",
            generated_at,
            metric_definitions,
            common_filters,
            ["zero_summary"],
        ),
        _source(
            "histogram_sql",
            "NVFP4 block zero-count distribution",
            database_path,
            histogram_sql,
            "Returns all 17 exact zero-count bins for the size-16 quantization blocks.",
            generated_at,
            metric_definitions,
            common_filters,
            ["block_zero_histogram"],
        ),
        _source(
            "projection_sql",
            "NVFP4 projection summary",
            database_path,
            projection_sql,
            "Returns exact zero statistics for gate, up, down, and lm_head projections.",
            generated_at,
            metric_definitions,
            common_filters,
            ["zero_summary"],
        ),
        _source(
            "layer_sql",
            "NVFP4 Transformer-layer summary",
            database_path,
            layer_sql,
            "Returns exact zero statistics for the 64 Transformer MLP layers.",
            generated_at,
            metric_definitions,
            common_filters,
            ["zero_summary"],
        ),
        {
            "id": "scan_script",
            "label": "NVFP4 memmap statistics implementation",
            "path": "tools/analysis/nvfp4_per_block_zeros.py",
            "query": {
                "engine": "python",
                "language": "python",
                "description": "Canonical safetensors header parsing, E2M1 decoding, scale validation, and per-block aggregation implementation.",
                "executed_at": generated_at,
                "filters": common_filters,
                "metric_definitions": metric_definitions,
            },
        },
    ]
    manifest_sources = [
        {"id": source["id"], "label": source["label"], "path": source["path"]}
        for source in sources
    ]
    title = "Qwen3.6-27B-NVFP4 分块零值统计"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "固定 revision 的 size-16 NVFP4 quant-block 精确零值分布与校验。",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "zero_rate",
                "description": "严格反量化零值占全部逻辑 NVFP4 权重的比例。",
                "dataset": "overall",
                "sourceId": "overall_sql",
                "metrics": [{"label": "零值率", "field": "zero_rate", "format": "percent"}],
            },
            {
                "id": "any_zero_rate",
                "description": "至少含一个零的 size-16 量化块占比。",
                "dataset": "overall",
                "sourceId": "overall_sql",
                "metrics": [
                    {"label": "含零 Block", "field": "any_zero_rate", "format": "percent"}
                ],
            },
            {
                "id": "all_zero_blocks",
                "description": "16 个值全部为零的量化块数量。",
                "dataset": "overall",
                "sourceId": "overall_sql",
                "metrics": [
                    {"label": "全零 Block", "field": "all_zero_blocks", "format": "number"}
                ],
            },
            {
                "id": "mean_zeros",
                "description": "每个 size-16 量化块中的平均零值数量。",
                "dataset": "overall",
                "sourceId": "overall_sql",
                "metrics": [
                    {"label": "平均零/Block", "field": "mean_zeros_per_block", "format": "number"}
                ],
            },
        ],
        "charts": [
            {
                "id": "block_zero_distribution",
                "title": "size-16 量化块的零值个数分布",
                "subtitle": "17 个精确桶，覆盖 1,149,009,920 个量化块；纵轴从 0 开始。",
                "type": "bar",
                "dataset": "block_histogram",
                "sourceId": "histogram_sql",
                "valueFormat": "compact",
                "layout": "full",
                "encodings": {
                    "x": {"field": "zero_count", "type": "ordinal", "label": "每块零值个数"},
                    "y": {
                        "field": "block_count",
                        "type": "quantitative",
                        "label": "Block 数",
                        "format": "compact",
                    },
                    "tooltip": [
                        {"field": "block_share", "type": "quantitative", "label": "Block 占比", "format": "percent"},
                        {"field": "zero_values_contributed", "type": "quantitative", "label": "贡献零值", "format": "compact"},
                    ],
                },
            }
        ],
        "tables": [
            {
                "id": "projection_summary",
                "title": "按投影汇总",
                "subtitle": "gate、up、down 与 lm_head 的精确零值统计。",
                "dataset": "projections",
                "sourceId": "projection_sql",
                "defaultSort": {"field": "zero_rate", "direction": "desc"},
                "columns": [
                    {"field": "projection", "label": "投影", "type": "text"},
                    {"field": "zero_rate", "label": "零值率", "format": "percent"},
                    {"field": "any_zero_rate", "label": "含零 Block", "format": "percent"},
                    {"field": "zero_values", "label": "零值数", "format": "number"},
                    {"field": "all_zero_blocks", "label": "全零 Block", "format": "number"},
                ],
            },
            {
                "id": "layer_summary",
                "title": "64 层 MLP 明细",
                "subtitle": "按零值率降序；每层包含 gate、up、down 三个 NVFP4 张量。",
                "dataset": "layers",
                "sourceId": "layer_sql",
                "defaultSort": {"field": "zero_rate", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "layer", "label": "层", "type": "text"},
                    {"field": "zero_rate", "label": "零值率", "format": "percent"},
                    {"field": "any_zero_rate", "label": "含零 Block", "format": "percent"},
                    {"field": "zero_values", "label": "零值数", "format": "number"},
                    {"field": "blocks", "label": "Block 数", "format": "number"},
                ],
            },
        ],
        "sources": manifest_sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "overall_sql",
                "body": (
                    "## 技术结论：零值率 7.948%，但不存在 size-16 全零块\n\n"
                    "- 18,384,158,720 个逻辑 NVFP4 权重中，1,461,170,821 个严格反量化为零。\n"
                    "- 72.175% 的量化块至少含一个零，但 1,149,009,920 个块中没有任何全零块。\n"
                    "- 因此，按 size-16 block 做整块跳过没有可利用空间；若目标是稀疏加速，需要检查更细粒度或结构化模式。"
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": ["zero_rate", "any_zero_rate", "all_zero_blocks", "mean_zeros"],
            },
            {
                "id": "distribution_finding",
                "type": "markdown",
                "sourceId": "histogram_sql",
                "body": (
                    "## 大多数含零块只有 1–2 个零\n\n"
                    "直方图展示每个 size-16 量化块里的严格零值个数。0、1、2 个零三档合计占 86.45%；"
                    "高零密度块极少，且 16/16 桶为零。这个分布支持‘细粒度零较普遍、整块稀疏不存在’的判断。"
                ),
            },
            {"id": "distribution_chart", "type": "chart", "chartId": "block_zero_distribution", "layout": "full"},
            {
                "id": "projection_finding",
                "type": "markdown",
                "sourceId": "projection_sql",
                "body": (
                    "## 投影间差异很小，lm_head 略高\n\n"
                    "四类投影的零值率都落在 7.916%–7.981% 之间；lm_head 最高，gate 最低。"
                    "差异仅约 0.065 个百分点，不足以改变 size-16 整块跳过的结论。"
                ),
            },
            {"id": "projection_table", "type": "table", "tableId": "projection_summary"},
            {
                "id": "layer_finding",
                "type": "markdown",
                "sourceId": "layer_sql",
                "body": (
                    "## 64 层零值率集中在窄区间\n\n"
                    "最高为 layer_63 的 8.1566%，最低为 layer_03 的 7.8423%。层间变动存在，"
                    "但所有层都保持同一数量级，未出现可单独启用整块稀疏路径的异常层。"
                ),
            },
            {"id": "layer_table", "type": "table", "tableId": "layer_summary", "layout": "full"},
            {
                "id": "scope_definitions",
                "type": "markdown",
                "sourceId": "scan_script",
                "body": (
                    "## 统计范围与定义\n\n"
                    "统计范围是固定 revision 的 193 个 W4A16_NVFP4 张量：64 层 × gate/up/down，加 lm_head。"
                    "每个 quant block 是同一输出行沿 K 连续的 16 个逻辑权重。E2M1 的 `0x0`（+0）和 `0x8`（-0）均计为数值零；"
                    "FP8 attention、vision、BF16 与 MTP 权重不在本次分母中。"
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "scan_script",
                "body": (
                    "## 方法：按 safetensors 原生布局逐块扫描\n\n"
                    "脚本解析 index/header 后，以只读 memmap 扫描 9.19 GB packed U8 权重；每 8 bytes 还原一个 block，"
                    "同时读取对应 E4M3 block scale 与 F32 tensor scale。输出在 tensor、layer、projection、overall 四个粒度上聚合，"
                    "并用直方图求和、nibble 码总数、block 总数和 scale 有效性做内部交叉校验。"
                ),
            },
            {
                "id": "robustness",
                "type": "markdown",
                "sourceId": "scan_script",
                "body": (
                    "## 鲁棒性：两种零口径完全一致\n\n"
                    "全部 1,149,009,920 个 block scale 和 193 个 tensor scale 都是有限正数，"
                    "所以没有 scale 诱发零或 FP32 下溢；严格反量化零与 E2M1 `0x0/0x8` 计数逐项相同。"
                    "结果是描述性权重统计，不直接等价于可获得的运行时加速。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 建议：不要为 size-16 全零块设计跳过路径\n\n"
                    "若目的是 Orin kernel 优化，当前数据否定了整块稀疏跳过。下一步更有价值的是统计 2:4 命中率、"
                    "K32/K64 tile 内的零位置分布，以及 backend repack 后是否出现可合并的零模式，再把节省的指令/访存与分支开销一起建模。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## 后续问题\n\n"
                    "逐块 `uint8` 明细未默认落盘，因为 11.49 亿行即使每块只存 1 byte 也约 1.15 GB。"
                    "如果需要研究具体零位置或结构化稀疏，可按 tensor 生成压缩 sidecar，并只保留目标层/投影。"
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "overall": overall_rows,
                "block_histogram": histogram_rows,
                "projections": projection_rows,
                "layers": layer_rows,
            },
        },
        "sources": sources,
        "package_info": {},
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    stats_path = args.stats.resolve()
    output_path = args.output.resolve()
    database_path = args.database.resolve()
    stats = _load_json(stats_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect_database(database_path)
    try:
        _populate_database(connection, stats)
        try:
            relative_database = database_path.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            relative_database = database_path.name
        artifact = _artifact(stats, relative_database, connection)
    finally:
        connection.close()
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path, database_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", type=Path, help="stats.json from nvfp4_per_block_zeros.py")
    parser.add_argument("--output", type=Path, required=True, help="canonical artifact.json")
    parser.add_argument("--database", type=Path, required=True, help="SQLite report source")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    artifact, database = run(parse_args(argv))
    print(f"artifact: {artifact}")
    print(f"database: {database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

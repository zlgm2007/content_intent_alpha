"""ES 数据导入器：过滤 → 按时间排序拉取（search_after 严格增量）→ 清洗 → 落库。

导入在后台线程执行，进度写入 import_tasks 表。
增量位点存 sync_state（JSON 的 search_after sort 键），保证不重不漏。
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from src.web import db
from src.web.config import get_es_config
from src.web.es_client import ESClient


def _none_if_null(v: Any) -> Any:
    """把字符串 'null' 归一化为 None。"""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() == "null":
        return None
    return v


def build_query(fields: dict, only_core: bool) -> dict:
    """构建 ES 查询。only_core=True 时只拉有意图分数或意图实体的核心样本。"""
    must: List[dict] = []
    if only_core:
        should = [
            {"exists": {"field": fields["intent_score"]}},
            {"exists": {"field": fields["intent_brand"]}},
            {"exists": {"field": fields["intent_category"]}},
            {"exists": {"field": fields["intent_product"]}},
        ]
        must.append({"bool": {"should": should, "minimum_should_match": 1}})
    if must:
        return {"bool": {"must": must}}
    return {"match_all": {}}


def clean_doc(source: dict, fields: dict) -> Optional[dict]:
    """提取并清洗单条文档，返回可落库的 dict；无有效内容返回 None。"""
    f = fields
    comment = _none_if_null(source.get(f["text_comment"]))
    desc = _none_if_null(source.get(f["text_desc"]))
    # 评论和描述都为空则丢弃
    if not comment and not desc:
        return None

    score = _none_if_null(source.get(f["intent_score"]))
    try:
        score = int(score) if score is not None else None
    except (ValueError, TypeError):
        score = None

    return {
        "source_index": source.get("_index"),
        "comment_content": (comment or "")[:2000],
        "note_desc": (desc or "")[:5000],
        "note_title": (_none_if_null(source.get(f["text_title"])) or "")[:500],
        "note_topics": (_none_if_null(source.get(f["text_topics"])) or "")[:1000],
        "intent_content": (_none_if_null(source.get(f["intent_content"])) or "")[:2000],
        "intent_brand": _none_if_null(source.get(f["intent_brand"])),
        "intent_category": _none_if_null(source.get(f["intent_category"])),
        "intent_product": _none_if_null(source.get(f["intent_product"])),
        "intent_score": score,
        "update_time": str(_none_if_null(source.get(f["update_time"])) or ""),
        "import_batch": "",
    }


def run_import(task_id: int, params: dict):
    """后台导入任务主函数。

    params: {indices: [...], only_core: bool, incremental: bool, max_docs: int}
    """
    cfg = get_es_config()
    fields = cfg["fields"]
    client = ESClient(cfg["hosts"])
    batch = uuid.uuid4().hex[:12]

    indices = params.get("indices") or cfg["indices"]
    only_core = params.get("only_core", cfg.get("only_core", True))
    incremental = params.get("incremental", True)
    max_docs = params.get("max_docs") or 0
    sort_field = fields["update_time"]

    db.update_import_task(task_id, status="running", message="开始导入")

    total_imported = 0
    try:
        for idx in indices:
            # 读增量位点（search_after sort 键）
            last_sort = None
            if incremental:
                raw = db.get_sync_state(idx)
                if raw:
                    try:
                        last_sort = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        last_sort = None

            query = build_query(fields, only_core)
            count = client.count(idx, query)
            db.update_import_task(task_id, message=f"索引 {idx} 命中 {count} 条，开始拉取")

            buffer: List[dict] = []
            source_fields = list(fields.values())
            new_last_sort = last_sort

            for hit in client.search_after_all(
                idx, query,
                size=cfg["scroll_size"],
                source_fields=source_fields,
                sort_field=sort_field,
                search_after=last_sort,
            ):
                new_last_sort = hit.get("_sort")
                doc = clean_doc({**hit.get("_source", {}), "_index": hit.get("_index", idx)}, fields)
                if doc is None:
                    continue
                doc["import_batch"] = batch
                buffer.append(doc)

                if len(buffer) >= 500:
                    db.insert_es_docs(buffer, batch)
                    total_imported += len(buffer)
                    buffer = []
                    db.update_import_task(task_id, total=total_imported,
                                          message=f"已导入 {total_imported} 条")

                if max_docs and (total_imported + len(buffer)) >= max_docs:
                    break

            if buffer:
                db.insert_es_docs(buffer, batch)
                total_imported += len(buffer)

            if new_last_sort:
                db.set_sync_state(idx, json.dumps(new_last_sort))

            if max_docs and total_imported >= max_docs:
                break

        db.update_import_task(task_id, status="done", total=total_imported,
                              message=f"导入完成，共 {total_imported} 条", finished=True)
    except Exception as e:  # noqa: BLE001
        db.update_import_task(task_id, status="failed",
                              message=f"导入失败: {e}", finished=True)

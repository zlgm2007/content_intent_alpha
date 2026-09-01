"""Elasticsearch 客户端封装（基于 requests 直连 REST API）。

支持：连接测试、计数、terms 聚合、scroll 顺序拉取、slice 并发分片拉取。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional

import requests


class ESClient:
    def __init__(self, hosts: List[str], timeout: int = 30):
        self.hosts = hosts
        self.base = hosts[0].rstrip("/")
        self.timeout = timeout

    # ---------- 基础 ----------

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def _request(self, method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None,
                 timeout: Optional[int] = None) -> dict:
        url = self._url(path)
        resp = requests.request(
            method, url,
            json=body,
            params=params,
            headers={"Content-Type": "application/json"},
            timeout=timeout or self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def ping(self) -> dict:
        return self._request("GET", "/")

    def count(self, index: str, query: Optional[dict] = None) -> int:
        r = self._request("POST", f"/{index}/_count", body={"query": query} if query else {})
        return r["count"]

    def terms_agg(self, index: str, field: str, size: int = 5000, query: Optional[dict] = None) -> List[dict]:
        """对 keyword 字段做 terms 聚合，返回 [{key, count}, ...]。"""
        body = {
            "size": 0,
            "query": query or {"match_all": {}},
            "aggs": {"terms": {"terms": {"field": field, "size": size}}},
        }
        r = self._request("POST", f"/{index}/_search", body=body, timeout=60)
        buckets = r["aggregations"]["terms"]["buckets"]
        return [{"key": b["key"], "count": b["doc_count"]} for b in buckets]

    # ---------- scroll ----------

    def _scroll_init(self, index: str, query: dict, size: int, timeout: str,
                     source_fields: Optional[List[str]], slice_spec: Optional[dict] = None) -> tuple:
        body: Dict[str, Any] = {
            "size": size,
            "query": query,
        }
        if source_fields:
            body["_source"] = source_fields
        if slice_spec:
            body["slice"] = slice_spec
        params = {"scroll": timeout}
        r = self._request("POST", f"/{index}/_search", body=body, params=params, timeout=120)
        return r.get("_scroll_id"), r.get("hits", {}).get("hits", [])

    def _scroll_continue(self, scroll_id: str, timeout: str) -> tuple:
        r = self._request(
            "POST", "/_search/scroll",
            body={"scroll": timeout, "scroll_id": scroll_id}, timeout=120,
        )
        return r.get("_scroll_id"), r.get("hits", {}).get("hits", [])

    def _clear_scroll(self, scroll_id: str):
        try:
            self._request("DELETE", "/_search/scroll", body={"scroll_id": [scroll_id]}, timeout=10)
        except Exception:
            pass

    def scroll(self, index: str, query: dict, size: int = 1000, timeout: str = "2m",
               source_fields: Optional[List[str]] = None) -> Iterator[dict]:
        """顺序 scroll 拉取全部命中文档。"""
        scroll_id, hits = self._scroll_init(index, query, size, timeout, source_fields)
        try:
            while hits:
                for h in hits:
                    yield h
                scroll_id, hits = self._scroll_continue(scroll_id, timeout)
        finally:
            if scroll_id:
                self._clear_scroll(scroll_id)

    def slice_scroll(self, index: str, query: dict, size: int = 1000, timeout: str = "2m",
                     slices: int = 4, source_fields: Optional[List[str]] = None) -> Iterator[dict]:
        """并发 slice scroll 拉取，返回全部命中文档（顺序不定）。"""
        def _slice(i: int) -> List[dict]:
            out = []
            scroll_id, hits = self._scroll_init(
                index, query, size, timeout, source_fields,
                slice_spec={"id": i, "max": slices},
            )
            try:
                while hits:
                    out.extend(hits)
                    scroll_id, hits = self._scroll_continue(scroll_id, timeout)
            finally:
                if scroll_id:
                    self._clear_scroll(scroll_id)
            return out

        with ThreadPoolExecutor(max_workers=slices) as pool:
            futures = [pool.submit(_slice, i) for i in range(slices)]
            for fut in as_completed(futures):
                for h in fut.result():
                    yield h

    def search_after_all(self, index: str, query: dict, size: int = 1000,
                         source_fields: Optional[List[str]] = None,
                         sort_field: str = "updateTime",
                         search_after: Optional[list] = None) -> Iterator[dict]:
        """按 sort_field 升序 + _id 升序，用 search_after 顺序翻页。

        相比 scroll，search_after 保证按时间顺序遍历，可做严格的断点增量。
        每条 hit 会附带 "sort" 键（[sort_value, _id]），供下次 search_after 续传。
        """
        body: Dict[str, Any] = {
            "size": size,
            "query": query,
            "sort": [{sort_field: {"order": "asc", "unmapped_type": "long"}}, {"_id": "asc"}],
        }
        if source_fields:
            body["_source"] = source_fields

        while True:
            if search_after:
                body["search_after"] = search_after
            r = self._request("POST", f"/{index}/_search", body=body, timeout=120)
            hits = r["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                h["_sort"] = h.get("sort")
                yield h
            if len(hits) < size:
                break
            search_after = hits[-1].get("sort")

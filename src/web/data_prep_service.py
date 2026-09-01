"""数据准备服务：从已导入的 es_docs 生成训练数据。

- NER：正文+评论拼接文本 + intent 实体对齐 → BIO conll
- 意图：正文+评论拼接文本 + intent_score（无分=0）→ 6 分类 jsonl（支持采样平衡）

正文与评论统一通过 src.common.compose.compose_note_comment 拼接，
保证训练与在线推理的特征格式一致。
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

from src.common.compose import compose_note_comment
from src.web import db
from src.web.config import BASE_DIR, DATA_DIR

# 型号：字母数字混合（与 lexicon_service 保持一致）
MODEL_RE = re.compile(r"[A-Za-z]{1,8}\d{1,5}[A-Za-z0-9]{0,12}|\d{2,6}[A-Za-z]{1,8}[A-Za-z0-9]{0,12}")

NER_OUT = os.path.join(DATA_DIR, "ner", "annotated", "train.conll")
# gold 标注快照：词典冲突消解与短词白名单都以它为证据源。
# 必须固定用快照而非 NER_OUT——否则切到词典标注后，证据会变成"自己标注自己"，
# 形成自指循环，冲突消解和噪声过滤会逐渐失去外部依据。
NER_GOLD_OUT = os.path.join(DATA_DIR, "ner", "annotated", "train_gold.conll")
# 留出集：在「去重采样之前」随机预留，用于评估泛化。
# 必须早于 max_per_pattern 切分，否则留出集全是"被去重淘汰"的同质样本。
NER_HOLDOUT_OUT = os.path.join(DATA_DIR, "ner", "annotated", "holdout.conll")
# 留出集副本：额外存 ES 真值字段，供双轨评测还原 A 轨（gold）标注
NER_HOLDOUT_JSONL = os.path.join(DATA_DIR, "ner", "annotated", "holdout.jsonl")
INTENT_OUT = os.path.join(DATA_DIR, "intent", "labeled", "train.jsonl")
INTENT_HOLDOUT_OUT = os.path.join(DATA_DIR, "intent", "labeled", "holdout.jsonl")
# 留出集：必须在【平衡之前】随机预留，否则 0/2/4 这类小类会被 100% 纳入训练
# （实测 2 分、4 分的留出样本数为 0），6 分类退化成 3 分类，评估结论不可信。
# 实体 -> 规范类型 映射（消除「同词异标」，详见 build_entity_type_map）
ENTITY_TYPE_MAP_FILE = os.path.join(DATA_DIR, "ner", "annotated", "entity_type_map.json")


def build_entity_type_map(
    conll_path: str = NER_OUT, min_count: int = 5, min_ratio: float = 0.6
) -> Dict[str, str]:
    """统计每个实体文本的主类型，生成「实体 -> 规范类型」映射。

    为什么需要：ES 的 intent_category / intent_product 字段存在大量交叉，
    同一个词在不同样本里被填进不同字段，导致训练标注「同词异标」。
    实测原训练数据 **64.9% 的实体标注存在此冲突**（如"空调" 90% 标 CATEGORY
    却有 10% 标 PRODUCT、"小米" 在 BRAND/PRODUCT 间摇摆）。模型学到模糊边界后，
    推理时只能给出 0.49 vs 0.44 这类五五开的预测，产生碎片实体。

    本函数把「出现次数 >= min_count 且主类型占比 >= min_ratio」的实体固定为
    其主类型，标注时强制归一化，保证同词同标。
    """
    stat: Dict[str, Counter] = {}
    cur_type: List[str] = []
    cur_chars: List[str] = []

    def _flush():
        nonlocal cur_type, cur_chars
        if cur_type:
            w = "".join(cur_chars)
            if w:
                stat.setdefault(w, Counter())[cur_type] += 1
        cur_type, cur_chars = "", []

    with open(conll_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                _flush()
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ch, tag = parts[0], parts[1]
            if tag.startswith("B-"):
                _flush()
                cur_type, cur_chars = tag[2:], [ch]
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur_chars.append(ch)
            else:
                _flush()
    _flush()

    tmap: Dict[str, str] = {}
    for w, c in stat.items():
        total = sum(c.values())
        if total < min_count:
            continue
        etype, n = c.most_common(1)[0]
        if n / total >= min_ratio:
            tmap[w] = etype
    return tmap


def save_entity_type_map(tmap: Dict[str, str], path: str = ENTITY_TYPE_MAP_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tmap, f, ensure_ascii=False, indent=0)


def load_entity_type_map(path: str = ENTITY_TYPE_MAP_FILE) -> Dict[str, str]:
    """加载实体类型映射；文件不存在时返回空 dict（退化为原标注逻辑）。"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _align(
    text: str, brand, category, product, type_map: Dict[str, str] = None
) -> Tuple[List[str], List[str]]:
    """把意图实体在文本中定位，生成字符级 BIO 标签。

    type_map 非空时，实体类型会被强制归一化为该映射中的规范类型，
    用于消除「同词异标」（详见 build_entity_type_map 文档）。
    """
    tokens = list(text)
    tags = ["O"] * len(tokens)
    occupied = [False] * len(tokens)

    items = []
    if product:
        items.append(("PRODUCT", str(product)))
    if brand:
        items.append(("BRAND", str(brand)))
    if category:
        items.append(("CATEGORY", str(category)))
    if product:
        for m in MODEL_RE.findall(str(product)):
            if len(m) >= 2:
                items.append(("MODEL", m))

    # 去重，长实体优先标注
    seen = set()
    uniq = []
    for t, v in items:
        if (t, v) not in seen:
            seen.add((t, v))
            uniq.append((t, v))
    uniq.sort(key=lambda x: -len(x[1]))

    # 类型归一化：把实体强制映射为其规范类型，消除同词异标
    if type_map:
        merged = []
        seen2 = set()
        for t, v in uniq:
            nt = type_map.get(v, t)
            if (nt, v) not in seen2:
                seen2.add((nt, v))
                merged.append((nt, v))
        uniq = merged

    for etype, val in uniq:
        start = 0
        while True:
            idx = text.find(val, start)
            if idx == -1:
                break
            seg = occupied[idx : idx + len(val)]
            if not any(seg):
                for j in range(len(val)):
                    pos = idx + j
                    tags[pos] = ("B-" if j == 0 else "I-") + etype
                    occupied[pos] = True
            start = idx + 1
    return tokens, tags


# ==================== 词典全量一致标注（方案 1） ====================
# 原 _align 只标「该样本 ES gold 字段里出现的字符串」，实测导致两个硬伤：
#   1) span 不一致：同一词在不同样本里时标时不标（「汽车」出现 5324 次只标 41.3%）
#   2) 同词异标    ：同一词在不同样本里标成不同类型（64.9% 实体存在冲突）
# 模型学到模糊边界 + 不完整召回后，推理时给出 0.49 vs 0.44 五五开的预测并产生碎片实体。
#
# 改为「词典全量一致标注」：词典里的词**每次出现都标**，且类型全局固定。
# 实测词典覆盖了 95.4% 的实体出现次数（不同实体数口径 56.5%，因为大量长尾只出现 1 次），
# 词典外的 4.6% 主要是单字（鞋/酒/床）和多品牌合并串（"西门子/美的/海尔"），
# 后者本身就是 ES 打标程序的脏输出，丢弃反而更干净。

DICT_MAP_FILE = os.path.join(DATA_DIR, "ner", "annotated", "dict_entity_map.json")
LEXICON_DIR = os.path.join(DATA_DIR, "ner", "lexicon")
# 人工屏蔽词表：误入词典的通用词/修饰词。自动识别不可行，需人工维护。
# 详见 load_blocklist 文档。
BLOCKLIST_FILE = os.path.join(DATA_DIR, "ner", "lexicon", "blocklist.txt")
_LEX_FILES = {
    "BRAND": "brand.txt",
    "CATEGORY": "category.txt",
    "PRODUCT": "product.txt",
    "MODEL": "model.txt",
}
# 冲突消解的默认优先级（仅用于 ES gold 里没有任何标注记录的词）
# 品牌/商品比品类更具体，故优先；MODEL 由正则提取、噪声最多，排最后。
_TYPE_PRIORITY = ["BRAND", "PRODUCT", "MODEL", "CATEGORY"]

_PURE_CN = re.compile(r"^[一-龥]+$")
# ES 打标程序会把解释性文字写进字段，这类词条不是真实体
_DIRTY = re.compile(r"[（）()「」【】]|从(文章|评论)内容|未提及|可推断|等识别|识别$")
_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")


def _is_boundary_ok(text: str, start: int, end: int) -> bool:
    """ASCII 词边界检查：防止 "Mac" 匹配进 "MacBook"、"ID" 匹配进 "IDEA"。

    仅当匹配片段的首/尾字符是 ASCII 字母数字时才校验，中文无词边界、不做检查。
    """
    if _ASCII_ALNUM.match(text[start]):
        if start > 0 and _ASCII_ALNUM.match(text[start - 1]):
            return False
    if _ASCII_ALNUM.match(text[end - 1]):
        if end < len(text) and _ASCII_ALNUM.match(text[end]):
            return False
    return True


def load_blocklist(path: str = BLOCKLIST_FILE) -> Set[str]:
    """读取人工屏蔽词表（通用词 / 误入词典的修饰词）。

    为什么需要人工维护：实测自动识别不可行。试过两种统计量都失败——
      - 左右邻接熵：高频实体天然邻接多样，把「空调/美的/海尔」全误判为通用词
      - 按词频归一化后：正确词与错误词数值重叠（美的 0.82 vs 国产 0.82、
        京东 1.14 vs 新品 1.14），完全不可分
    根因是本语料里所有词都出现在相似的商品讨论语境，分布信号区分不了
    「品牌」和「修饰品牌的形容词」。这类判断依赖业务语义，只能人工维护。

    文件格式：每行一个词，空行忽略。支持两种注释：
      - 整行注释：以 # 开头
      - 行内注释：词后面跟 # 说明（便于记录「为什么屏蔽这个词」）
    解析时取第一个 # 之前的内容并 strip，所以「国产    # 形容词」只取「国产」。
    """
    words: Set[str] = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.split("#", 1)[0].strip()
                if w:
                    words.add(w)
    return words


def load_lexicon(lexicon_dir: str = LEXICON_DIR) -> Dict[str, Set[str]]:
    """读取四类词典文件，返回 {类型: 词条集合}。"""
    lex: Dict[str, Set[str]] = {}
    for t, fname in _LEX_FILES.items():
        path = os.path.join(lexicon_dir, fname)
        words = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w and not w.startswith("#"):
                        words.add(w)
        lex[t] = words
    return lex


def build_dict_entity_map(
    conll_path: str = NER_GOLD_OUT,
    min_gold_count: int = 2,
) -> Dict[str, str]:
    """构建「词 -> 规范类型」的全局映射，用于全量一致标注。

    三步：
      1. 清洗：剔除解释性脏词条（如「洗发水（从文章内容"…"等识别）」）
      2. 消解冲突：一个词同时属于多个词典时（实测 8.0%，如 MODEL+PRODUCT 826 例），
         优先采纳 ES gold 里被标注最多的类型；无 gold 记录时按 _TYPE_PRIORITY 兜底。
         实测与 ES gold 主类型一致率 93.0%。
      3. 过滤短词风险：含英数且长度 <= 3 的词（如 of/or/On/ID/TO）在正文里极易误匹配，
         要求至少有 min_gold_count 次 gold 标注证据才保留。纯中文双字词歧义低，不受此限。

    min_gold_count=2 的取值依据：实测 1 次证据的词里混有 up/max/ren 等噪声，
    2 次能有效滤掉偶发标注，同时保住 TCL(507)/P20(309)/SUV(90) 等真实体。
    """
    lex = load_lexicon()
    gold = _collect_entity_stats(conll_path)
    blocked = load_blocklist()

    owner: Dict[str, List[str]] = {}
    for t, words in lex.items():
        for w in words:
            if w in blocked:            # 0. 人工屏蔽词优先于一切
                continue
            if _DIRTY.search(w):        # 1. 清洗脏词条
                continue
            owner.setdefault(w, []).append(t)

    wmap: Dict[str, str] = {}
    for w, types in owner.items():
        # 3. 短词风险过滤（含英数 + 长度<=3 需要 gold 证据）
        if len(w) <= 3 and not _PURE_CN.match(w):
            if sum(gold.get(w, Counter()).values()) < min_gold_count:
                continue
        # 2. 冲突消解
        c = gold.get(w)
        picked = None
        if c and sum(c.values()) > 0:
            cands = [(t, c.get(t, 0)) for t in types]
            best_t, best_n = max(cands, key=lambda x: (x[1], -_TYPE_PRIORITY.index(x[0])))
            if best_n > 0:
                picked = best_t
        if picked is None:
            for t in _TYPE_PRIORITY:
                if t in types:
                    picked = t
                    break
        wmap[w] = picked

    # 4. 丢弃「类型冲突的复合词」——最长匹配会把组分吞掉，反而降低召回
    #    例："美的空调"(PRODUCT) 会吞掉 "美的"(BRAND) 与 "空调"(CATEGORY)，
    #    而 gold 标准是把两者分开标。删除复合词后两者各自标出。
    #    品牌变体不受影响：KONO/KO、MacBook/Mac 同类型，不满足丢弃条件。
    #    实测（3000 条）：R 69.3%->81.2%、P 37.4%->39.6%、F1 48.6%->53.2%。
    wmap = _drop_conflicting_composites(wmap)
    return wmap


def _drop_conflicting_composites(wmap: Dict[str, str]) -> Dict[str, str]:
    """丢弃能被完整拆成 2+ 个词典词、且组分类型与自身不同的复合词。

    只删「可完整覆盖」的（每个组分长度 >= 2），避免把 "小米11" 这种
    品牌+型号拆得七零八落。
    """
    from functools import lru_cache

    wset = set(wmap)

    def full_split(w: str):
        """把 w 完整拆成若干长度>=2 的词典词，拆不开则返回 None。"""
        n = len(w)

        @lru_cache(maxsize=None)
        def go(i: int):
            if i == n:
                return ()
            for j in range(i + 2, n + 1):
                if w[i:j] in wset:
                    r = go(j)
                    if r is not None:
                        return (w[i:j],) + r
            return None

        return go(0)

    drop = set()
    for w, t in wmap.items():
        if len(w) < 4:
            continue
        parts = full_split(w)
        if not parts or len(parts) < 2:
            continue
        part_types = {wmap[p] for p in parts}
        # 自身类型不在组分类型里，或组分跨多个类型 -> 粒度冲突，丢弃
        if t not in part_types or len(part_types) > 1:
            drop.add(w)
    return {k: v for k, v in wmap.items() if k not in drop}


def _collect_entity_stats(conll_path: str) -> Dict[str, Counter]:
    """从 conll 统计每个实体文本被标成各类型的次数。

    行格式是 `字\\tTAG`（制表符）——必须用 split('\\t')，用 split() 会被
    含空格的 token 破坏并让还原文本混入 "O"/"B-BRAND" 等标签字符（踩过的坑）。
    """
    stat: Dict[str, Counter] = {}
    cur_type, cur_chars = "", []

    def flush():
        nonlocal cur_type, cur_chars
        if cur_type and cur_chars:
            stat.setdefault("".join(cur_chars), Counter())[cur_type] += 1
        cur_type, cur_chars = "", []

    if not os.path.exists(conll_path):
        return stat
    with open(conll_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ch, tag = parts[0], parts[1]
            if tag.startswith("B-"):
                flush()
                cur_type, cur_chars = tag[2:], [ch]
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur_chars.append(ch)
            else:
                flush()
    flush()
    return stat


def save_dict_entity_map(wmap: Dict[str, str], path: str = DICT_MAP_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wmap, f, ensure_ascii=False, indent=0)


def load_dict_entity_map(path: str = DICT_MAP_FILE) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class DictAnnotator:
    """基于词典的全量一致标注器。

    为什么用「首字索引」而不是逐词 find：
      词典 1.4 万词 × 语料 10 万条，逐词 find 是 O(W×T)，纯 Python 下要跑数小时。
      首字索引把每个位置的候选词降到平均 ~5 个（1.4 万词 / 约 3 千个不同首字），
      整体降到 O(T×5)，实测可在 1~2 分钟内完成全量标注。
    """

    def __init__(self, wmap: Dict[str, str]):
        self.wmap = wmap
        self.char_index: Dict[str, List[str]] = {}
        for w in wmap:
            self.char_index.setdefault(w[0], []).append(w)
        # 长词优先，保证「小米空调」优先于「小米」匹配
        for k in self.char_index:
            self.char_index[k].sort(key=len, reverse=True)

    def annotate(self, text: str) -> Tuple[List[str], List[str]]:
        n = len(text)
        tags = ["O"] * n
        occupied = [False] * n
        i = 0
        while i < n:
            cands = self.char_index.get(text[i])
            matched = 0
            if cands:
                for w in cands:
                    L = len(w)
                    end = i + L
                    if end > n or text[i:end] != w:
                        continue
                    if any(occupied[i:end]):
                        continue
                    if not _is_boundary_ok(text, i, end):
                        continue
                    et = self.wmap[w]
                    tags[i] = "B-" + et
                    occupied[i] = True
                    for j in range(i + 1, end):
                        tags[j] = "I-" + et
                        occupied[j] = True
                    matched = L
                    break
            i += matched if matched else 1
        return list(text), tags


def _entity_pattern(tokens: List[str], tags: List[str]) -> tuple:
    """提取样本的「实体组合签名」=(类型, 实体文本) 的有序元组。

    用于按标注模式去重采样：同一组合的样本只是正文措辞不同、标注模式完全一样。
    """
    out = []
    cur_type: List[str] = []
    cur_chars: List[str] = []
    for ch, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if cur_type:
                out.append((cur_type, "".join(cur_chars)))
            cur_type, cur_chars = tag[2:], [ch]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_chars.append(ch)
        else:
            if cur_type:
                out.append((cur_type, "".join(cur_chars)))
            cur_type, cur_chars = "", []
    if cur_type:
        out.append((cur_type, "".join(cur_chars)))
    return tuple(out)


def build_ner_data(
    limit: int = 0,
    min_entities: int = 1,
    max_len: int = 0,
    rebuild_type_map: bool = False,
    max_per_pattern: int = 0,
    seed: int = 42,
    annotator: str = "gold",
    holdout_ratio: float = 0.0,
) -> dict:
    """生成 NER 训练 conll。输入为「正文+评论」拼接文本，实体在拼接文本中定位。

    只保留至少标注到 min_entities 个实体的样本。max_len=0 表示不额外截断
    （拼接函数已控制各字段长度）。

    annotator 选择标注方式：
      "gold"（旧）：只标该样本 ES gold 字段里出现的字符串，再做类型归一化。
                   实测漏标 51.1% —— 正文里出现了但 gold 没写的实体一个都不标，
                   模型学到「该词可标可不标」，推理时给出五五开概率并产生碎片实体。
      "dict"（新）：词典全量一致标注。词典里的词每次出现都标，类型全局固定。
                   详见 DictAnnotator 文档。

    max_per_pattern > 0 时按「实体组合」去重采样：每种标注模式最多保留 N 条。
    全量 12.97 万条其实只有 3.2 万种标注模式（平均每种重复 4 条），
    设 N=5 可把数据压到 5.4 万条（41.6%），3 epoch 训练从 ~52 分钟降到 ~22 分钟，
    效果损失极小。0 表示不采样（生成全量）。
    """
    if annotator == "dict":
        annot = DictAnnotator(load_dict_entity_map() or build_dict_entity_map())
    else:
        if rebuild_type_map or not os.path.exists(ENTITY_TYPE_MAP_FILE):
            if os.path.exists(NER_OUT):
                save_entity_type_map(build_entity_type_map(NER_OUT))
        type_map = load_entity_type_map()
        annot = None

    samples = db.fetch_ner_samples(limit)
    if max_per_pattern > 0 or holdout_ratio > 0:
        # 洗牌后再按配额保留，保证每组留下的是随机子集，
        # 避免 ES 按索引/时间的排列顺序带来采样偏差
        random.Random(seed).shuffle(samples)

    # 留出集必须在去重采样【之前】切走，否则留出集会被"模式去重"过滤成同质样本
    holdout_samples: list = []
    if holdout_ratio > 0:
        n_hold = int(len(samples) * holdout_ratio)
        holdout_samples, samples = samples[:n_hold], samples[n_hold:]

    def _annotate(s) -> Tuple[List[str], List[str]] | None:
        text = compose_note_comment(
            s.get("note_title"), s.get("note_desc"), s.get("comment_content")
        )
        if not text:
            return None
        if max_len and len(text) > max_len:
            text = text[:max_len]
        if annot is not None:
            return annot.annotate(text)
        return _align(
            text, s["intent_brand"], s["intent_category"], s["intent_product"], type_map
        )

    pattern_count: Counter = Counter()
    pattern_dropped = 0
    lines: List[str] = []
    annotated = 0
    for s in samples:
        pair = _annotate(s)
        if pair is None:
            continue
        tokens, tags = pair
        n_entities = sum(1 for t in tags if t.startswith("B-"))
        if n_entities < min_entities:
            continue

        if max_per_pattern > 0:
            sig = _entity_pattern(tokens, tags)
            if pattern_count[sig] >= max_per_pattern:
                pattern_dropped += 1
                continue
            pattern_count[sig] += 1

        annotated += 1
        for tok, tag in zip(tokens, tags):
            lines.append(f"{tok}\t{tag}")
        lines.append("")

    # 留出集：同样标注，但【不做】min_entities 过滤与模式去重，
    # 保证它反映的是真实分布（含 0 实体样本），评估才有意义。
    hold_lines: List[str] = []
    hold_entities = 0
    hold_rows: List[dict] = []
    for s in holdout_samples:
        pair = _annotate(s)
        if pair is None:
            continue
        tokens, tags = pair
        hold_entities += sum(1 for t in tags if t.startswith("B-"))
        for tok, tag in zip(tokens, tags):
            hold_lines.append(f"{tok}\t{tag}")
        hold_lines.append("")
        # 同时保留 ES 真值字段：评估时要在【同一批文本】上算两个轨
        #   A 轨 gold：_align(text, brand, category, product)
        #   B 轨 dict：annotator.annotate(text)
        # 没有原始字段就无法还原 A 轨，评测会退化成单一口径。
        hold_rows.append({
            "text": "".join(tokens),
            "intent_brand": s.get("intent_brand"),
            "intent_category": s.get("intent_category"),
            "intent_product": s.get("intent_product"),
        })

    os.makedirs(os.path.dirname(NER_OUT), exist_ok=True)
    with open(NER_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    if hold_lines:
        with open(NER_HOLDOUT_OUT, "w", encoding="utf-8") as f:
            f.write("\n".join(hold_lines) + "\n")
        with open(NER_HOLDOUT_JSONL, "w", encoding="utf-8") as f:
            for r in hold_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "samples": annotated,
        "out": NER_OUT,
        "annotator": annotator,
        "type_map_size": len(annot.wmap) if annot is not None else len(type_map),
        "patterns": len(pattern_count) if max_per_pattern > 0 else None,
        "max_per_pattern": max_per_pattern or None,
        "dedup_dropped": pattern_dropped if max_per_pattern > 0 else 0,
        "holdout": len(holdout_samples),
        "holdout_entities": hold_entities,
        "holdout_out": NER_HOLDOUT_OUT if hold_lines else None,
        "holdout_jsonl": NER_HOLDOUT_JSONL if hold_lines else None,
    }


def _balance(rows: List[dict], max_per_class: int = 0) -> List[dict]:
    """按 label 采样平衡，避免 0 分样本淹没其他类别。"""
    groups: Dict[int, List[dict]] = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    counts = sorted(len(g) for g in groups.values())
    if max_per_class:
        target = min(max_per_class, counts[len(counts) // 2])
    else:
        target = min(counts)
    balanced: List[dict] = []
    for label, g in groups.items():
        if len(g) > target:
            balanced.extend(random.sample(g, target))
        else:
            balanced.extend(g)
    random.shuffle(balanced)
    return balanced


def build_intent_data(limit: int = 0, balance: bool = True, max_per_class: int = 20000,
                      holdout_ratio: float = 0.0, seed: int = 42) -> dict:
    """生成意图 6 分类训练 jsonl。输入为「正文+评论」拼接文本，标签为 intent_score。无分样本按 0 分处理。

    留出集（holdout_ratio > 0）的切分顺序是**关键点**，顺序错了整个评估就废了：

      数据准备会对 1/3/5 类截断到 max_per_class，而 0/2/4 类数量本就不足、会被 100% 纳入训练。
      若在全部数据上先平衡再留出，留出集里 0/2/4 的样本数为 0（实测 2 分 0 条、4 分 0 条），
      6 分类退化成 3 分类 —— 此时「未见组准确率反而更高」只是任务变简单的假象。

    正确顺序：去重 → 随机留出 → 再平衡。
    另外按 text 去重，避免同一条文本同时出现在训练集和留出集造成泄漏。
    """
    samples = db.fetch_intent_samples(limit)
    rows: List[dict] = []
    seen_text = set()
    n_dup = 0
    for s in samples:
        text = compose_note_comment(
            s.get("note_title"), s.get("note_desc"), s.get("comment_content")
        )
        if not text:
            continue
        if text in seen_text:          # 去重：同一文本不得同时进训练和留出集
            n_dup += 1
            continue
        seen_text.add(text)
        score = s["intent_score"]
        score = 0 if score is None else max(0, min(5, int(score)))
        rows.append({"text": text, "label": score})

    dist_before = dict(Counter(r["label"] for r in rows))

    holdout: List[dict] = []
    if holdout_ratio > 0:
        random.Random(seed).shuffle(rows)
        n_hold = int(len(rows) * holdout_ratio)
        holdout, rows = rows[:n_hold], rows[n_hold:]

    if balance:
        rows = _balance(rows, max_per_class)

    os.makedirs(os.path.dirname(INTENT_OUT), exist_ok=True)
    with open(INTENT_OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if holdout:
        with open(INTENT_HOLDOUT_OUT, "w", encoding="utf-8") as f:
            for r in holdout:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist_after = dict(Counter(r["label"] for r in rows))
    return {
        "samples": len(rows),
        "dist_before": dist_before,
        "dist_after": dist_after,
        "out": INTENT_OUT,
        "dups_dropped": n_dup,
        "holdout": len(holdout),
        "holdout_dist": dict(sorted(Counter(r["label"] for r in holdout).items())),
        "holdout_out": INTENT_HOLDOUT_OUT if holdout else None,
    }

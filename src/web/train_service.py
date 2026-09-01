"""训练服务：后台线程异步训练，进度/日志写入事件缓冲供 SSE 实时推送。"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Optional

from src.web import db
from src.web.config import load_yaml

# task_id -> deque of event dicts（type: log / progress / done / failed）
_task_events: defaultdict = defaultdict(lambda: deque(maxlen=2000))
_events_lock = threading.Lock()


def emit(task_id: int, event_type: str, **data):
    with _events_lock:
        _task_events[task_id].append({"type": event_type, **data})


def get_events_after(task_id: int, last_seq: int):
    """返回 last_seq 之后的新事件，返回 (events, new_last_seq)。

    last_seq < 0 时返回全部历史事件（用于前端重新连接时回放）。
    """
    with _events_lock:
        events = list(_task_events[task_id])
    if last_seq < 0:
        return events, len(events)
    if last_seq >= len(events):
        return [], last_seq
    return events[last_seq:], len(events)


def clear_events(task_id: int):
    with _events_lock:
        _task_events.pop(task_id, None)


# 停止标志：task_id -> threading.Event
_stop_events: dict = {}
_import_lock = threading.Lock()
# 训练启动锁：保证「检查是否有任务在跑」+「创建任务」是原子的，
# 避免并发请求同时通过检查（全局同一时间只允许一个训练任务）
_train_start_lock = threading.Lock()


def stop_training(task_id: int) -> bool:
    """请求停止训练任务。返回是否有正在运行的任务被标记停止。"""
    ev = _stop_events.get(task_id)
    if ev is None:
        return False
    ev.set()
    db.update_train_task(task_id, message="正在停止…")
    return True


def _make_callback(task_id: int):
    def cb(progress: float, message: str, loss: Optional[float] = None):
        db.update_train_task(task_id, status="running", progress=round(progress, 4), message=message)
        emit(task_id, "progress", progress=round(progress, 4), message=message, loss=loss)
    return cb


def _apply_train_params(cfg: dict, params: dict):
    t = cfg.setdefault("train", {})
    for k in ("epochs", "batch_size", "learning_rate"):
        if params.get(k):
            t[k] = params[k]
    return cfg


def run_ner_training(task_id: int, params: dict):
    stop_event = threading.Event()
    _stop_events[task_id] = stop_event
    try:
        cfg = _apply_train_params(load_yaml("ner.yaml"), params)
        from src.ner.train import train as ner_train
        emit(task_id, "log", message="开始 NER 训练，加载模型…")
        db.update_train_task(task_id, status="running", progress=0.0, message="加载模型…")
        ner_train(cfg, progress_callback=_make_callback(task_id), stop_event=stop_event)
        if stop_event.is_set():
            emit(task_id, "log", message="训练已被手动停止，保留当前最优模型")
            db.update_train_task(task_id, status="stopped", message="已停止（保留当前最优模型）", finished=True)
            emit(task_id, "stopped", message="已停止（保留当前最优模型）")
        else:
            emit(task_id, "log", message="NER 训练完成")
            db.update_train_task(task_id, status="done", progress=1.0, message="NER 训练完成", finished=True)
            emit(task_id, "done", message="NER 训练完成")
    except Exception as e:  # noqa: BLE001
        msg = f"NER 训练失败: {e}"
        emit(task_id, "failed", message=msg)
        db.update_train_task(task_id, status="failed", message=msg, finished=True)
    finally:
        _stop_events.pop(task_id, None)


def run_intent_training(task_id: int, params: dict):
    stop_event = threading.Event()
    _stop_events[task_id] = stop_event
    try:
        cfg = _apply_train_params(load_yaml("intent.yaml"), params)
        from src.intent.train import train as intent_train
        emit(task_id, "log", message="开始意图训练，加载模型…")
        db.update_train_task(task_id, status="running", progress=0.0, message="加载模型…")
        intent_train(cfg, progress_callback=_make_callback(task_id), stop_event=stop_event)
        if stop_event.is_set():
            emit(task_id, "log", message="训练已被手动停止，保留当前最优模型")
            db.update_train_task(task_id, status="stopped", message="已停止（保留当前最优模型）", finished=True)
            emit(task_id, "stopped", message="已停止（保留当前最优模型）")
        else:
            emit(task_id, "log", message="意图训练完成")
            db.update_train_task(task_id, status="done", progress=1.0, message="意图训练完成", finished=True)
            emit(task_id, "done", message="意图训练完成")
    except Exception as e:  # noqa: BLE001
        msg = f"意图训练失败: {e}"
        emit(task_id, "failed", message=msg)
        db.update_train_task(task_id, status="failed", message=msg, finished=True)
    finally:
        _stop_events.pop(task_id, None)


def start_training(task_type: str, params: dict) -> int:
    """创建训练任务并启动后台线程，返回 task_id。

    全局同一时间只允许一个训练任务在跑：若检测到已有 running 任务，
    本次任务会被创建但立即标记 failed 并在 message 中说明原因（保留可追溯记录）。
    """
    if task_type == "ner":
        fn = run_ner_training
    elif task_type == "intent":
        fn = run_intent_training
    else:
        task_id = db.create_train_task(task_type, params)
        db.update_train_task(task_id, status="failed", message=f"未知任务类型 {task_type}", finished=True)
        return task_id

    # 检查 + 创建 + 启动放在同一把锁内，避免并发请求同时通过检查
    with _train_start_lock:
        running = db.get_running_train_task()
        task_id = db.create_train_task(task_type, params)
        if running:
            msg = (
                f"已有训练任务 #{running['id']}（{_TASK_CN.get(running['task_type'], running['task_type'])}）"
                f"正在运行，同一时间只允许一个训练任务"
            )
            db.update_train_task(task_id, status="failed", message=msg, finished=True)
            emit(task_id, "failed", message=msg)
            return task_id

        # 预热 import（加锁串行化），避免训练线程内首次 import 的并发竞态
        try:
            with _import_lock:
                if task_type == "ner":
                    import src.ner.train  # noqa: F401
                else:
                    import src.intent.train  # noqa: F401
        except Exception as e:  # noqa: BLE001
            msg = f"加载训练模块失败: {e}"
            db.update_train_task(task_id, status="failed", message=msg, finished=True)
            return task_id

        threading.Thread(target=fn, args=(task_id, params), daemon=True).start()
        return task_id


_TASK_CN = {"ner": "实体识别", "intent": "意图识别"}

"""意图概率校准：搜索最佳二分类阈值，可选等渗回归校准概率。

读入 train.py 保存的 val_probs.json（验证集概率与 0/1 标签），
输出 calibration.json 供 inference.py 使用。

用法：
    python -m src.intent.calibrate --config configs/intent.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """网格搜索使 F1 最大的二分类阈值。"""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (probs >= t).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def fit_isotonic(probs: np.ndarray, labels: np.ndarray, out_dir: str) -> str:
    """等渗回归校准，保存校准器。返回保存路径。"""
    import pickle

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels)
    path = os.path.join(out_dir, "isotonic.pkl")
    with open(path, "wb") as f:
        pickle.dump(iso, f)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/intent.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["train"]["output_dir"]
    probs_path = os.path.join(out_dir, "val_probs.json")
    if not os.path.exists(probs_path):
        print(f"[calibrate] 未找到 {probs_path}，请先运行 train.py")
        return

    with open(probs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    probs = np.array(data["probs"])
    labels = np.array(data["labels"])

    method = cfg["score"]["method"]
    calib = {
        "method": method,
        "best_threshold": find_best_threshold(probs, labels),
        "score_thresholds": cfg["score"]["thresholds"],
    }

    if method == "isotonic":
        calib["isotonic_path"] = fit_isotonic(probs, labels, out_dir)

    out_path = os.path.join(out_dir, "calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    print(f"[calibrate] 最佳二分类阈值 = {calib['best_threshold']:.3f}")
    print(f"[calibrate] 校准配置已保存 -> {out_path}")


if __name__ == "__main__":
    main()

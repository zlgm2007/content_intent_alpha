-- ============================================================
-- 监督模型自训练：自动标注未标注评论 — 数据库变更
-- 分支: dev_br20260806    日期: 2026-08-07
--
-- 说明: 本文件为数据库结构变更的文档性脚本。
--       实际迁移由 intent_app/db.py 的 SCHEMA（新库建表）+
--       _MIGRATE_COLUMNS 幂等补列（老库）执行，此处 SQL 与之一致。
-- ============================================================

-- ------------------------------------------------------------
-- 1. 手工评论表 comments 增加 auto_labeled 标记列
--    自动标注写库时置 1，用于区分人工标注 vs 自动标注，便于日后复核/过滤
--    （仅 SQLite 3.35+ 支持 ADD COLUMN IF NOT EXISTS；db.py 用 PRAGMA 判断幂等）
-- ------------------------------------------------------------
ALTER TABLE comments ADD COLUMN auto_labeled INTEGER DEFAULT 0;

-- ------------------------------------------------------------
-- 2. 外部同步标注表 annotations 增加 auto_labeled 标记列（同上）
-- ------------------------------------------------------------
ALTER TABLE annotations ADD COLUMN auto_labeled INTEGER DEFAULT 0;

-- 说明: 自动标注写库语义
--   - comments / annotations 均以 status='labeled' 计入训练数据（与人工标注一致）
--   - auto_labeled=1 标记为"由自动标注产生"，供日后按来源复核/剔除
--   - 自动标注 UPDATE 带 AND status='pending'，避免覆盖人工已标注数据

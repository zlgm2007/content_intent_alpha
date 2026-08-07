-- ============================================================
-- 外部平台数据同步（ES -> SQLite）数据库变更
-- 分支: dev_br20260806    日期: 2026-08-07
--
-- 说明: 本文件为数据库结构变更的文档性脚本。
--       实际建表/迁移由 intent_app/db.py 的 SCHEMA + _migrate() 幂等执行，
--       此处 SQL 与之一致，可用于手工初始化/对账。
-- ============================================================

-- ------------------------------------------------------------
-- 1. 作品表（ES 同步的外部平台作品）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS works (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id         INTEGER NOT NULL REFERENCES intent_goals(id) ON DELETE CASCADE,
    note_id         TEXT NOT NULL,                 -- ES noteId 外部作品ID
    note_title      TEXT,                          -- ES noteTitle
    content         TEXT,                          -- ES noteDesc 作品内容
    platform        TEXT,                          -- ES platform
    author_nickname TEXT,                          -- ES authorNickname
    author_id       TEXT,                          -- ES authorId
    note_time       INTEGER,                       -- ES noteTime (epoch_millis)
    note_type       TEXT,                          -- ES noteType
    note_cover      TEXT,                          -- ES noteCover
    note_video      TEXT,                          -- ES noteVideo
    note_url        TEXT,                          -- ES 作品链接
    topics          TEXT,                          -- ES noteTopics
    source_index    TEXT,                          -- 来源索引 ge3/lt3
    sync_id         INTEGER,                       -- 同步批次ID
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (goal_id, note_id)                      -- 同一目标下按外部作品ID去重
);

CREATE INDEX IF NOT EXISTS idx_works_goal ON works(goal_id);

-- ------------------------------------------------------------
-- 2. 标注表（ES 同步评论 + 人工标注）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS annotations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id            INTEGER NOT NULL REFERENCES intent_goals(id) ON DELETE CASCADE,
    work_id            INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    comment_id         TEXT NOT NULL,              -- ES commentId 外部评论ID
    comment            TEXT,                       -- ES commentContent 评论内容
    raw_score          INTEGER,                    -- ES intentScore 原始得分
    score              INTEGER,                    -- 标注值 0-5（人工）
    label              TEXT,                       -- has_intent / no_intent
    status             TEXT NOT NULL DEFAULT 'pending',  -- pending / labeled / skipped
    comment_create_time INTEGER,                   -- ES commentCreateTime
    comment_user_name  TEXT,                       -- ES commentUserName
    source_index       TEXT,                       -- 来源索引 ge3/lt3
    sync_id            INTEGER,                    -- 同步批次ID
    created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (work_id, comment_id)                   -- 同一作品下按外部评论ID去重
);

CREATE INDEX IF NOT EXISTS idx_ann_goal    ON annotations(goal_id, status);
CREATE INDEX IF NOT EXISTS idx_ann_work    ON annotations(work_id);
CREATE INDEX IF NOT EXISTS idx_ann_comment ON annotations(comment_id);

-- ------------------------------------------------------------
-- 3. 同步批次表（历史记录）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sync_batches (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id        INTEGER NOT NULL,
    indices        TEXT,                          -- ge3/lt3/both
    days           INTEGER,                       -- 按天数同步：最近 N 天
    sync_limit     INTEGER,                       -- 按条数同步：最近 N 条（与 days 互斥）
    state          TEXT NOT NULL DEFAULT 'running',  -- running / done / stopped / error
    works_inserted INTEGER DEFAULT 0,
    works_skipped  INTEGER DEFAULT 0,
    ann_inserted   INTEGER DEFAULT 0,
    ann_skipped    INTEGER DEFAULT 0,
    error          TEXT,
    started_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    finished_at    DATETIME
);

CREATE INDEX IF NOT EXISTS idx_sync_goal ON sync_batches(goal_id, id);

-- ------------------------------------------------------------
-- 4. 现有表补充字段（幂等迁移，供 db.py _migrate() 参考）
--    仅当列缺失时执行，SQLite 3.35+ 支持 ADD COLUMN IF NOT EXISTS
-- ------------------------------------------------------------
ALTER TABLE contents ADD COLUMN note_id      TEXT;
ALTER TABLE contents ADD COLUMN platform     TEXT;
ALTER TABLE contents ADD COLUMN note_title   TEXT;
ALTER TABLE contents ADD COLUMN note_time    INTEGER;
ALTER TABLE contents ADD COLUMN source_index TEXT;
ALTER TABLE contents ADD COLUMN sync_id      INTEGER;

ALTER TABLE comments ADD COLUMN comment_id   TEXT;
ALTER TABLE comments ADD COLUMN raw_score    INTEGER;
ALTER TABLE comments ADD COLUMN source_index TEXT;
ALTER TABLE comments ADD COLUMN sync_id      INTEGER;

-- 同步批次表补列（按条数模式；days 与 sync_limit 二选一）
ALTER TABLE sync_batches ADD COLUMN sync_limit INTEGER;

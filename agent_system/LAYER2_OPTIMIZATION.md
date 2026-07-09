# Layer2 优化 — 备份与恢复记录

## 备份清单

| 文件 | 备份位置 | 日期 |
|------|---------|------|
| agent_executor.py | agent_executor.py.backup | 2026-07-03 |
| myprompts/ (全部) | myprompts_old/ | 2026-07-03 |

## 恢复方式

```bash
# 恢复 executor
cd /media/data6/xuejj/CVRAgent/agent_system
cp agent_executor.py.backup agent_executor.py

# 恢复 prompts
rm -rf myprompts
cp -r myprompts_old myprompts
```

## 改动计划

### 1. Executor: `_validate_minimum_stages_before_answer`

**当前行为** (agent_executor.py ~984行):
- 必须 `_pss_layer2_covered_indices()` 覆盖所有 video_index
- 少一个 segment 都不给 answer

**改为**:
- 不要求全覆盖
- 只要有 `has_layer2` + `order_confidence >= 7` 就放行
- Master 自己决定哪些 chunk 需要 Layer2

### 2. Prompt: PSS contract + general prompt

**当前行为**:
- Layer2 做全部 segment 的全量重扫
- 每个 segment 完整扫一遍，只改 focus_prompt

**改为**:
- Layer2 针对 chunk 级别
- Master 从 Layer1 per-chunk 结果中提取 uncertain chunks
- Layer2 只观察不确定的 chunk（窄窗口 start_time/end_time）
- 明确标注跳过的 chunk 及理由
- 减少 VLM 调用次数

### 3. Prompt: general prompt Stage 2

**当前**: "For each video that has gaps, design concrete follow-up questions. Then re-observe those videos."

**改为**: "Only re-observe chunks with identified gaps. Do NOT re-scan chunks whose object states are already clearly established. List which chunks you are skipping and why."

---

## 已完成改动 (2026-07-03)

### 1. Executor: `_validate_minimum_stages_before_answer` (agent_executor.py ~984行)

**旧**: 必须所有 segment 被 Layer2 覆盖 (`missing_layer2_indices` 为空)

**新**: 
- Layer1 仍然必做
- Layer2 至少做一次（不要求全覆盖）
- 如果 `order_confidence >= 7` → 直接放行 answer
- 如果 `order_confidence < 7` 且有未覆盖的 segment → 提示针对性复查
- 明确告知 Master 只需观察 uncertain chunks，不需要全量重扫

### 2. Prompt: PSS contract (task_contracts/PSS.prompt)

**旧**: Layer2 针对 segments，要求全覆盖

**新**:
- Layer2 针对 CHUNKS，不是 segments
- 必须列出 skip list：哪些 chunk 跳过 + 原因
- 窄窗口观察（start_time/end_time 精确到 chunk）
- 如果所有 chunks 已有充分证据且 confidence >= 7，只需 minimal Layer2（1-2 chunks）
- 明确："Do NOT run Layer2 on all segments just to satisfy a coverage requirement"

### 3. Prompt: general prompt (master_crossvid_general.prompt)

**旧**: Stage 2 "For each video that has gaps, re-observe those videos"

**新**:
- "Target only uncertain chunks, not full segments"
- "Narrow start_time/end_time to that chunk's exact window"
- 必须在 thought 中列出 skip list
- Stage 2 后如果 confidence >= 7 可直接 answer，不需要强制 focus
- PSS: order_confidence >= 7 + ORDER_AUDIT → 直接 answer

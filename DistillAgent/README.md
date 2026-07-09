# DistillAgent

蒸馏管线：使用强 API 模型 + 诊断 Agent 自检修正，从 CrossVid 训练集生成正确的多轮 Agent 轨迹，用于 SFT 微调 Qwen3-8B 和 Qwen-VL。

## 核心流程

```
训练集问题 → Master Agent (API) 多轮推理 → 答案正确?
  ├── ✅ → 直接保留轨迹作为 SFT 数据
  └── ❌ → 诊断Agent自检回溯 → 定位错误 → 注入修正 → 继续推理
           └── 重复直到正确或达到最大修正次数
```

## 目录结构

```
DistillAgent/
├── run_distill.py           # 入口脚本
├── distill_core.py          # 蒸馏编排器（主循环 + 诊断修正循环）
├── diagnosis_agent.py       # 诊断 Agent（轨迹分析 + 修正引导）
├── task_runner.py           # 任务适配器（10个任务的视频上下文构建）
├── task_config.py           # 任务配置（查询模板、答案解析、正确性判断）
├── trajectory_converter.py  # 轨迹转 SFT 格式
├── .env.example             # 环境变量模板
└── README.md
```

## 快速开始

### 1. 环境准备

API keys 和其他配置已在 `agent_system/.env` 中配置好，`run_distill.py` 会自动加载。

确保 Python 环境已安装 `agent_system/requirements.txt` 中的依赖（decord, torch, transformers, opencv-python-headless 等）。

### 2. 确保训练数据已划分

```bash
# 如果还没划分（在 agent_system/ 下执行）：
cd ../agent_system
python split_train_test.py
# 这会生成 question/train/ 和 question/test/
```

### 3. 运行蒸馏

```bash
# 单个任务
python run_distill.py --task PSS

# 多个任务
python run_distill.py --task PSS,FSA,CC

# 全部任务
python run_distill.py --all

# 自定义设置
python run_distill.py --task PSS \
    --max-diagnosis 5 \
    --max-turns 25 \
    --start 1 --end 50 \
    --output-dir ./my_output
```

### 4. 输出

```
output/
├── sft_data.jsonl              # SFT 训练数据（可直接用于 verl SFT）
├── per_task_sft/               # 每个任务的单独 SFT 文件
│   ├── PSS_sft.jsonl
│   └── ...
├── raw_trajectories/            # 完整轨迹（调试用）
│   ├── all_trajectories.jsonl
│   └── trajectory_summary.json
├── failed_samples/              # 无法修正的样本
│   └── failed_samples.json
└── distillation_report.json     # 最终报告
```

## SFT 数据格式

输出的 `sft_data.jsonl` 每行一个 JSON：

```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<query + video info>"},
    {"role": "assistant", "content": "<JSON tool call>"},
    {"role": "user", "content": "[TOOL OBSERVATION]\n<result>"},
    {"role": "assistant", "content": "<JSON answer>"}
  ],
  "metadata": {
    "task": "PSS",
    "question_id": "42",
    "num_attempts": 1,
    "has_diagnosis": false
  }
}
```

Tool role 被转为 `user` 角色（Qwen3 兼容），内容加了 `[TOOL OBSERVATION]` 前缀标识。

## 诊断 Agent 设计

诊断 Agent 使用独立的 API（`DIAGNOSIS_API_*` 环境变量），对失败轨迹进行分析：

- **输入**：任务类型、问题、正确答案、完整对话历史、工具调用历史
- **输出**：
  - `error_turn`: 哪个轮次出了错
  - `error_type`: premature_answer | insufficient_evidence | wrong_observation | reasoning_error | format_error | missing_tool_call
  - `correction_strategy`: reobserve_specific | backtrack_and_focus | restart_with_guidance | minor_fix
  - `correction_message`: 注入对话的修正引导

修正策略：
- **reobserve_specific** / **backtrack_and_focus**：截断对话到错误点之前，注入诊断消息，继续推理
- **restart_with_guidance**：从头开始，系统提示中加入诊断引导

## 注意事项

- 蒸馏使用 API 后端（MASTER_BACKEND=api），不会加载本地模型
- 视频需要可访问（本地缓存或远程 URL）
- CCQA 任务无法自动判断正确性，直接保留轨迹
- 建议先用小范围（如 `--start 1 --end 20`）测试流程，确认无误再全量运行

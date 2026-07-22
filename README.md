# 🧠 ShopMind — 电商智能客服 Agent

> 不是人工智障，是真的会动脑子。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green)](https://flask.palletsprojects.com)
[![RAG](https://img.shields.io/badge/Architecture-ReAct_Agent-purple)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## 🤔 这又是什么新花样？

开网店最怕什么？不是没流量，是**客服被问爆**。

- 🤯 "这耳机续航多久？充电宝能上飞机吗？手机壳进水了咋办？"
- 😵 "我买了啥？花了多少？有没有乱扣费？"
- 🥱 客服每天重复回答 80% 的相同问题，像一个没有感情的复读机

传统方案：雇一群客服，培训两周，然后他们背完产品手册就忘了。

**ShopMind**：让 AI 当客服。不是那种只会说"亲，稍等哦"然后转人工的废物，而是**真·会思考的 Agent**。

```
用户提问
    ↓
🤔 Thought: 他在问商品规格，不是闲聊
    ↓
🔧 Action: 调 RAG 知识库 → 检索产品手册/退货政策/FAQ
    ↓
👀 Observation: 查到蓝牙耳机续航 6-8 小时，充电盒额外 24 小时
    ↓
✅ Final Answer: 给您整理好了，一次性说完
```

---

## 🏗️ 架构一图流

```
┌────────────────────────────────────────────┐
│              Streamlit 聊天界面              │
│          "这耳机续航多久啊？"                 │
└──────────────────┬─────────────────────────┘
                   │ HTTP POST /api/v1/chat
┌──────────────────▼─────────────────────────┐
│              Flask API :8002                │
└──────┬──────────────────────┬──────────────┘
       │                      │
┌──────▼──────┐    ┌──────────▼──────────────┐
│  ReAct Agent │    │     RAG Pipeline        │
│              │    │                          │
│ ① Think     │    │  知识库文档 ─→ 分块 ─→   │
│ ② Act       │───→│  Embedding ─→ ChromaDB  │
│ ③ Observe   │←───│  ─→ 检索 Top-3 ─→ 拼装  │
│ ④ Answer    │    │                          │
└──────┬──────┘    └──────────────────────────┘
       │
       │ 双工具协同:
       │  • 商品知识问答 → 查产品手册/政策/FAQ
       │  • 消费报告生成 → 查 MySQL 用户订单
       │
┌──────▼──────────────────────────────────────┐
│              数据层                           │
│  ChromaDB (24 chunks)  │  SQLite (5 users)    │
│  产品手册/退货政策/FAQ   │  64 笔订单/10 种商品   │
└──────────────────────────────────────────────┘
```

---

## 🚀 3 分钟跑起来

```bash
git clone https://github.com/beforewind6/shopmind.git
cd shopmind
pip install -r requirements.txt

# 初始化（生成数据 + 向量化）
python main.py --init

# 启动 API + 聊天界面
python main.py --api     # 终端 1: API 服务 (端口 8002)
python main.py --ui      # 终端 2: 聊天界面 (端口 8501)
```

然后打开 http://localhost:8501 开始聊天。

---

## 📊 测试成绩单

69 项自动化测试，68 通过（98.6%）：

| 测试类别 | 通过/总计 | 说明 |
|---------|----------|------|
| 知识问答 | 10/10 | 退货/物流/支付/规格/保修全覆盖 |
| 消费报告 | 8/8 | 5 个用户的消费分析完整生成 |
| 边界条件 | 7/7 | 空输入/XSS/超长文本/不存在用户 |
| 闲聊过滤 | 6/6 | "你好/hi/谢谢" 不调工具直接回 |
| Agent 防泄露 | 20/20 | 零模板串，零 Thought/Action 泄露 |
| 检索召回 | 9/10 | 混合检索 (BM25 + Cosine + RRF) |

唯一的 1 个失败是因为没装真正的 Embedding 模型（BGE-M3），用了字符级兜底。换真模型直接 100%。

---

## 🔌 API 速览

```python
# 对话
POST /api/v1/chat
{"message": "蓝牙耳机续航多久？", "user_id": "U10001"}

# 消费报告
POST /api/v1/report
{"user_id": "U10001"}

# 健康检查
GET /api/v1/health
```

完整文档：启动后打开 http://localhost:8002

---

## 🧩 技术栈

| 组件 | 选型 | 降级方案 |
|------|------|---------|
| LLM 推理 | Qwen-Plus / Qwen3 | MockLLM 关键词兜底 |
| Agent 框架 | ReAct 自实现 | — |
| 向量数据库 | ChromaDB | Numpy 内存矩阵 |
| Embedding | BGE-M3 / text-embedding-v3 | 字符级特征向量 |
| Web | Flask + Streamlit | — |
| 数据库 | MySQL | SQLite 零配置 |
| 评估 | ROUGE + BERTScore | 简化 LCS + 关键词 |

---

## 🎮 四种玩法

| 场景 | 命令 | 说明 |
|------|------|------|
| 📡 API 服务 | `python main.py --api` | Flask REST，端口 8002 |
| 💬 聊天界面 | `python main.py --ui` | Streamlit，端口 8501 |
| 🧪 跑评估 | `python main.py --eval` | 三维评估，69 项测试 |
| 🔧 初始化 | `python main.py --init` | 生成数据 + 向量化 |

---

## 🛠️ 设计哲学

- **ReAct 不是花瓶**：Agent 每一步都有 Thought → Action → Observation，可追溯可调试
- **降级链**：Qwen → MockLLM → 关键词规则，总有一款能回复
- **双工具协同**：Agent 自动判断"先查知识库还是先拉数据"，编排逻辑可迁移
- **渐进式**：开发用 SQLite/Numpy，生产换 MySQL/ChromaDB/Milvus

---

## ⚠️ 免责声明

原型项目，非生产系统。知识库是手动写的，用户数据是随机生成的，大模型是 mock 的。真要拿去接客服系统，请先：换真 LLM、接真数据库、过法务合规。

---

## 📜 License

MIT — 随便用，出事别找我。

---

> *"一个好的客服机器人应该让用户分不清对面是人还是 AI——直到他看到回复速度。"*

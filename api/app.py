"""Flask API 服务 —— ShopMind 智能客服系统

启动方式:
  python api/app.py

接口:
  POST /api/v1/chat     对话接口
  POST /api/v1/report   报告生成
  GET  /api/v1/health   健康检查
"""

import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.config import config
from utils.logger import setup_logger
from rag.vector_store import VectorStore
from rag.retrieval_chain import RetrievalQA, build_rag_pipeline
from agent.agent_core import ReActAgent, MockLLM
from agent.tools import build_tools

logger = setup_logger("shopmind-api")

# ============ 全局系统初始化 ============

_vector_store = None
_retrieval_qa = None
_agent = None


def init_system():
    global _vector_store, _retrieval_qa, _agent
    if _agent is not None:
        return

    logger.info("正在初始化 ShopMind 系统...")

    # 向量存储
    _vector_store = VectorStore(
        backend=config.get("vector_store.backend", "numpy"),
        persist_dir=config.get("vector_store.persist_dir", "./data/vector_db"),
    )

    # 加载知识库
    knowledge_dir = Path(__file__).parent.parent / "data" / "knowledge"
    if _vector_store.get_collection_size() == 0 and knowledge_dir.exists():
        logger.info("向量库为空，正在构建索引...")
        _retrieval_qa = build_rag_pipeline(
            str(knowledge_dir),
            vector_store=_vector_store,
            llm=MockLLM(),
        )
        _vector_store.save_all()
    else:
        _retrieval_qa = RetrievalQA(vector_store=_vector_store, llm=MockLLM())

    # Agent
    db_path = str(Path(__file__).parent.parent / "data" / "db" / "shopmind.db")
    agent_tools = build_tools(_retrieval_qa, db_path)
    _agent = ReActAgent(tools=agent_tools, max_iterations=config.get("agent.max_iterations", 5))

    logger.info(f"ShopMind 初始化完成 (向量库 {_vector_store.get_collection_size()} 条, 工具 {agent_tools.get_tool_names()})")


# ============ Flask App ============

app = Flask(__name__)
CORS(app)


@app.route("/api/v1/chat", methods=["POST"])
def chat():
    """对话接口"""
    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4())[:8])
    user_id = data.get("user_id", "")

    if not message:
        return jsonify({
            "reply": "您好！请问有什么可以帮您？请描述您的问题，我会尽力为您解答。",
            "agent_trace": [],
            "tool_calls": [],
            "session_id": session_id,
            "total_steps": 0,
            "total_time_ms": 0,
        })

    if _agent is None:
        init_system()

    # 如果提供了 user_id，优先做消费报告
    if user_id and ("消费" in message or "报告" in message or "买了" in message or "账单" in message):
        message = f"请为 {user_id} 生成消费报告: {message}"

    result = _agent.run(message, session_id=session_id)

    return jsonify({
        "reply": result.reply,
        "agent_trace": result.agent_trace,
        "tool_calls": result.tool_calls,
        "session_id": session_id,
        "total_steps": result.total_steps,
        "total_time_ms": result.total_time_ms,
    })


@app.route("/api/v1/report", methods=["POST"])
def report():
    """消费报告生成接口"""
    data = request.get_json(force=True)
    user_id = data.get("user_id", "").strip()

    if not user_id:
        return jsonify({"error": "user_id 不能为空"}), 400

    if _agent is None:
        init_system()

    result = _agent.run(f"请为 {user_id} 生成消费报告")
    return jsonify({
        "report": result.reply,
        "user_id": user_id,
        "generated_at": datetime.now().isoformat(),
        "total_time_ms": result.total_time_ms,
    })


@app.route("/api/v1/health", methods=["GET"])
def health():
    """健康检查"""
    vs_size = _vector_store.get_collection_size() if _vector_store else 0
    return jsonify({
        "status": "healthy",
        "version": "1.0",
        "vectorstore_size": vs_size,
        "llm_status": "mock (local fallback)",
    })


@app.route("/")
def index():
    return """
    <html>
    <head><title>ShopMind API</title></head>
    <body style="font-family:sans-serif;max-width:600px;margin:40px auto">
      <h1>ShopMind API</h1>
      <p>电商智能客服 Agent 系统 v1.0</p>
      <h3>接口:</h3>
      <ul>
        <li><code>POST /api/v1/chat</code> — 对话接口</li>
        <li><code>POST /api/v1/report</code> — 消费报告生成</li>
        <li><code>GET /api/v1/health</code> — 健康检查</li>
      </ul>
      <p>启动 Streamlit UI: <code>streamlit run ui/app.py</code></p>
    </body>
    </html>
    """


if __name__ == "__main__":
    init_system()
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8002)
    logger.info(f"ShopMind API 启动: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

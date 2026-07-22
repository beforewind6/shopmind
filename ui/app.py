"""ShopMind Streamlit Web UI —— 智能客服聊天界面"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from utils.config import config
from rag.vector_store import VectorStore
from rag.retrieval_chain import RetrievalQA, build_rag_pipeline
from agent.agent_core import ReActAgent, MockLLM
from agent.tools import build_tools

st.set_page_config(
    page_title="ShopMind - 智能客服",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ 初始化 ============

@st.cache_resource
def init_agent():
    vs = VectorStore(
        backend=config.get("vector_store.backend", "numpy"),
        persist_dir=config.get("vector_store.persist_dir", "./data/vector_db"),
    )
    knowledge_dir = Path(__file__).parent.parent / "data" / "knowledge"
    if vs.get_collection_size() == 0 and knowledge_dir.exists():
        rqa = build_rag_pipeline(str(knowledge_dir), vector_store=vs, llm=MockLLM())
    else:
        rqa = RetrievalQA(vector_store=vs, llm=MockLLM())

    db_path = str(Path(__file__).parent.parent / "data" / "db" / "shopmind.db")
    agent_tools = build_tools(rqa, db_path)
    return ReActAgent(tools=agent_tools), vs, agent_tools


agent, vs, tools = init_agent()

# ============ UI ============

st.title("🛍️ ShopMind - 电商智能客服")
st.caption("基于 ReAct Agent + RAG 的智能客服系统 | 知识库 {vs.get_collection_size()} 条文档")

# Sidebar
with st.sidebar:
    st.header("📋 控制面板")
    st.metric("知识库文档", f"{vs.get_collection_size()} 条")
    st.metric("可用工具", "商品知识问答, 消费报告生成")

    st.divider()
    st.subheader("💡 快捷测试")

    quick_qs = [
        "蓝牙耳机的续航时间？",
        "充电宝可以带上飞机吗？",
        "退货后多久退款？",
        "如何申请发票？",
        "帮我看看 U10001 的消费",
    ]
    for q in quick_qs:
        if st.button(q, use_container_width=True):
            st.session_state["quick_q"] = q

    st.divider()
    st.subheader("👤 用户ID")
    user_id = st.text_input("输入用户ID（用于消费报告）", value="U10001", placeholder="U10001")

# Main chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 您好！我是 ShopMind 智能客服。我可以帮您查询商品信息、退换货政策，也可以生成您的消费报告。请问有什么可以帮您？"}
    ]

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框始终渲染（不在条件分支内）
prompt = st.chat_input("输入您的问题...")

# 快捷问题覆盖
if not prompt:
    qq = st.session_state.pop("quick_q", None)
    if qq:
        prompt = qq

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            msg = prompt
            if user_id and any(kw in prompt for kw in ["消费", "报告", "买了", "账单", "订单"]):
                msg = f"请为 {user_id} 生成消费报告: {prompt}"

            result = agent.run(msg)
            st.markdown(result.reply)

            with st.expander("查看推理过程 (ReAct Trace)", expanded=False):
                for step in result.agent_trace:
                    lines = [f"- Thought: {step.get('thought', '')}"]
                    if step.get("action"):
                        lines.append(f"- Action: `{step['action']}` (`{step.get('action_input', '')}`)")
                    if step.get("observation"):
                        lines.append(f"- Observation: {step['observation'][:300]}")
                    if step.get("final_answer"):
                        lines.append(f"- Final Answer: {step['final_answer'][:300]}")
                    st.markdown(f"**Step {step['step']}**\n" + "\n".join(lines))
                st.caption(f"总耗时: {result.total_time_ms}ms | 步数: {result.total_steps}")

    st.session_state.messages.append({"role": "assistant", "content": result.reply})

"""ShopMind 全面测试 —— 覆盖所有接口、边界条件、异常场景"""
import sys
import json
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent))

from api.app import app, init_system

init_system()
client = app.test_client()

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        msg = f"  [FAIL] {name} — {detail}"
        print(msg)
        errors.append(msg)


def call_chat(msg, uid="", sid=None):
    data = {"message": msg}
    if uid:
        data["user_id"] = uid
    if sid:
        data["session_id"] = sid
    r = client.post("/api/v1/chat", json=data)
    return r.status_code, r.get_json()


def call_report(uid):
    r = client.post("/api/v1/report", json={"user_id": uid})
    return r.status_code, r.get_json()


# ════════════════════════════════════════
print("=" * 60)
print("  ShopMind 全面测试")
print("=" * 60)

# ── 1. 健康检查 ──
print("\n--- 1. 基础接口 ---")
r = client.get("/api/v1/health")
test("GET /health 返回200", r.status_code == 200)
data = r.get_json()
test("health 包含 status", data and "status" in data)
test("health 包含 vectorstore_size", data and "vectorstore_size" in data)
test("health vectorstore_size > 0", data and data.get("vectorstore_size", 0) > 0)
test("health 包含 version", data and "version" in data)

r = client.get("/")
test("GET / 返回200", r.status_code == 200)

# ── 2. 知识问答 ──
print("\n--- 2. 知识问答 ---")
knowledge_cases = [
    ("蓝牙耳机续航多久", ["续航", "小时"]),
    ("充电宝可以带上飞机吗", ["飞机", "mAh"]),
    ("退货后多久能收到退款", ["退款", "工作日"]),
    ("如何申请发票", ["发票", "申请"]),
    ("PLUS会员有什么权益", ["PLUS", "会员"]),
    ("支持哪些支付方式", ["支付", "微信", "支付宝"]),
    ("什么时候发货", ["发货", "小时"]),
    ("蓝牙耳机保修多久", ["保修", "质保", "年"]),
    ("手机壳脏了怎么清洁", ["清洁", "擦拭"]),
    ("数据线坏了能换吗", ["数据线", "终身"]),
]

for q, keywords in knowledge_cases:
    code, data = call_chat(q)
    reply = data.get("reply", "") if data else ""
    # 检查是否包含预期关键词
    hits = sum(1 for kw in keywords if kw in reply)
    test(
        f"知识问答: {q[:25]}...",
        code == 200 and hits >= 1,
        f"code={code}, keywords_hit={hits}/{len(keywords)}, reply_preview={reply[:150]}"
    )

# ── 3. 消费报告 ──
print("\n--- 3. 消费报告 ---")
code, data = call_chat("U10001的消费报告")  # 使用 chat 接口
test("Chat 生成消费报告 200", code == 200)
reply = data.get("reply", "") if data else ""
test("消费报告包含用户名", "张明" in reply, f"reply={reply[:100]}")
test("消费报告包含金额", "元" in reply)

# API 报告接口
code, data = call_report("U10001")
test("POST /report 返回200", code == 200)
test("report 接口包含 report 字段", data and "report" in data)
test("report 接口包含 generated_at", data and "generated_at" in data)

code, data = call_report("U10003")
test("U10003 报告200", code == 200)
reply = data.get("report", "") if data else ""
test("U10003 报告包含用户名", "王芳" in reply)

# ── 4. 无效输入 ──
print("\n--- 4. 边界条件 ---")
code, data = call_chat("")
test("空消息返回200", code == 200)  # 应正常处理

code, data = call_report("")
test("空 user_id report 返回400", code == 400)

code, data = call_report("NONEXIST")
test("不存在的用户 report 返回200", code == 200)
reply = data.get("report", "") if data else ""
test("不存在的用户返回提示信息", "未找到" in reply or "不存在" in reply or "暂无" in reply, f"reply={reply[:100]}")

# POST 无 body
r = client.post("/api/v1/chat", data="", content_type="application/json")
test("无 body 返回400", r.status_code == 400)

# 超长文本
long_text = "请问" + "蓝牙耳机" * 200  # ~1600 字
code, data = call_chat(long_text)
test("超长文本不崩溃", code == 200)

# 特殊字符
code, data = call_chat("test & <script>alert(1)</script>")
test("XSS 不崩溃返回200", code == 200)

# ── 5. 闲聊 ──
print("\n--- 5. 闲聊处理 ---")
greetings = ["你好", "您好", "hi", "hello", "在吗", "谢谢"]
for g in greetings:
    code, data = call_chat(g)
    reply = data.get("reply", "") if data else ""
    steps = data.get("total_steps", -1)
    test(
        f"问候 '{g}' → 直接回复不走工具",
        code == 200 and steps == 1 and len(reply) > 5,
        f"steps={steps}, reply={reply[:80]}"
    )

# ── 6. 多轮对话 ──
print("\n--- 6. 多轮对话 ---")
sid = "test_session_001"
# 第一轮
code1, d1 = call_chat("蓝牙耳机续航多久", sid=sid)
test("会话首轮 200", code1 == 200)
# 第二轮（同一 session）
code2, d2 = call_chat("那充电宝呢", sid=sid)
test("会话次轮 200", code2 == 200)

# ── 7. Agent 输出质量 ──
print("\n--- 7. Agent 输出质量 ---")
leak_patterns = [
    "可重复Thought",
    "工具返回结果...",
    "{agent_scratchpad}",
    "{tools}",
]
for q, _ in knowledge_cases[:5]:
    code, data = call_chat(q)
    reply = data.get("reply", "") if data else ""
    for leak in leak_patterns:
        test(f"无模板泄露 [{q[:20]}]: {leak}", leak not in reply, f"reply_contains={leak}")

# ── 8. 向量检索质量 ──
print("\n--- 8. 向量检索覆盖 ---")
from rag.retrieval_chain import RetrievalQA
from rag.vector_store import VectorStore
vs = VectorStore()
rqa = RetrievalQA(vector_store=vs)
test_results = []
for q, keywords in knowledge_cases:
    result = rqa.invoke({"query": q})
    docs = result.get("source_documents", [])
    hits = sum(1 for kw in keywords if any(kw in d["text"] for d in docs))
    passed_case = hits >= 1 and len(docs) > 0
    test_results.append(passed_case)
    test(f"检索召回 [{q[:25]}]: {hits}/{len(keywords)} 关键词, {len(docs)} 文档",
         passed_case, f"keywords={keywords}")
retrieval_ok = all(test_results)
if not retrieval_ok:
    for kw in ["数据线", "保修", "清洁"]:
        r = vs.similarity_search(kw, k=3)
        if not r:
            print(f"    排查: '{kw}' 检索为空，需要检查向量库")

# ════════════════════════════════════════
print()
print("=" * 60)
print(f"  结果: {passed} 通过 / {passed + failed} 总计")
if failed > 0:
    print(f"  {failed} 个失败:")
    for e in errors:
        print(f"    {e}")
else:
    print("  全部通过!")
print("=" * 60)

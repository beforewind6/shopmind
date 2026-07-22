"""ShopMind 三维评估脚本 —— 检索质量 + 生成质量 + 端到端测试"""
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config
from rag.vector_store import VectorStore
from rag.retrieval_chain import RetrievalQA, build_rag_pipeline
from agent.agent_core import ReActAgent, MockLLM
from agent.tools import build_tools


# 测试用例（问题 + 参考答案 + 期望的相关文档关键词）
TEST_CASES = [
    # ===== 知识问答 =====
    {
        "question": "蓝牙耳机的续航时间是多久？",
        "reference": "蓝牙耳机单次充电续航6-8小时，充电盒可额外提供24小时续航。",
        "relevant_keywords": ["续航", "6-8小时", "24小时", "充电"]
    },
    {
        "question": "充电宝可以带上飞机吗？",
        "reference": "额定能量不超过100Wh（约27000mAh）的充电宝可以随身携带，不可托运。",
        "relevant_keywords": ["飞机", "100Wh", "27000", "托运", "随身"]
    },
    {
        "question": "买错了尺码可以换货吗？",
        "reference": "服装类商品支持换货，来回运费买家承担。",
        "relevant_keywords": ["换货", "尺码", "运费", "承担"]
    },
    {
        "question": "退货后多久能收到退款？",
        "reference": "仓库签收后1-2个工作日质检，通过后支付宝/微信3-5个工作日到账。",
        "relevant_keywords": ["退款", "1-2", "3-5", "工作日", "质检"]
    },
    {
        "question": "如何申请发票？",
        "reference": "在订单详情页点击「申请发票」，可选择电子发票或纸质发票。",
        "relevant_keywords": ["发票", "订单详情", "电子", "纸质"]
    },
    {
        "question": "蓝牙耳机支持同时连接几台设备？",
        "reference": "支持同时连接2台设备（手机+电脑），可一键切换音频来源。",
        "relevant_keywords": ["2台", "同时", "连接", "切换"]
    },
    {
        "question": "PLUS会员有什么权益？",
        "reference": "PLUS会员享有包邮、专属折扣、双倍退货时效、优先客服、生日礼包等。",
        "relevant_keywords": ["PLUS", "包邮", "折扣", "退货时效", "优先"]
    },
    {
        "question": "什么时候发货？默认用什么快递？",
        "reference": "现货商品24小时内发货，默认使用顺丰和京东快递。",
        "relevant_keywords": ["24小时", "发货", "顺丰", "京东"]
    },

    # ===== 消费报告 =====
    {
        "question": "帮我看看U10001的消费情况",
        "reference": "生成用户消费报告，包含消费金额、品类偏好、月度趋势",
        "relevant_keywords": ["消费", "报告", "订单", "品类", "元"]
    },
    {
        "question": "U10003最近买了什么？有什么推荐？",
        "reference": "查询用户订单列表并基于消费偏好给出推荐",
        "relevant_keywords": ["订单", "推荐", "购买"]
    },
]


def evaluate_retrieval():
    """评估检索质量：Recall@K, MRR, NDCG@K"""
    print("\n" + "=" * 60)
    print("  1. 检索质量评估")
    print("=" * 60)

    vs = VectorStore(
        backend=config.get("vector_store.backend", "numpy"),
        persist_dir=config.get("vector_store.persist_dir", "./data/vector_db"),
    )

    knowledge_dir = Path(__file__).parent.parent / "data" / "knowledge"
    if vs.get_collection_size() == 0:
        print("  向量库为空，正在构建索引...")
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader(chunk_size=512, chunk_overlap=64)
        docs = loader.load_directory(str(knowledge_dir))
        vs.add_documents(docs, "ecommerce_knowledge")
        vs.save_all()
        print(f"  已加载 {len(docs)} 个文本块")

    knowledge_qa_tests = TEST_CASES[:8]  # 前8个是知识问答
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for i, case in enumerate(knowledge_qa_tests):
        results = vs.similarity_search(case["question"], k=5)
        retrieved_texts = " ".join([r["text"] for r in results])

        # Recall@K: 关键词语料覆盖率
        keywords = case.get("relevant_keywords", [])
        recalled = sum(1 for kw in keywords if kw.lower() in retrieved_texts.lower())
        recall = recalled / len(keywords) if keywords else 0
        recall_scores.append(recall)

        # MRR: 第一个包含关键词的结果排位倒数
        for rank, r in enumerate(results, 1):
            if any(kw.lower() in r["text"].lower() for kw in keywords):
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

        # NDCG@5: 简化版（用关键词命中数作为相关性分数）
        dcg = 0
        for rank, r in enumerate(results, 1):
            hits = sum(1 for kw in keywords if kw.lower() in r["text"].lower())
            rel = hits / len(keywords) if keywords else 0
            dcg += rel / (np_log2(rank + 1))
        idcg = sum(1.0 / np_log2(i + 1) for i in range(1, min(len(keywords), 5) + 1))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

        status = "PASS" if recall >= 0.5 else "WARN"
        print(f"  [{status}] Q{i+1}: {case['question'][:40]}... | Recall@5={recall:.2f} | MRR={mrr_scores[-1]:.2f} | NDCG@5={ndcg:.2f}")

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

    print(f"\n  --- 检索质量汇总 ---")
    print(f"  Recall@5 (avg): {avg_recall:.4f} (目标 >= 0.85)")
    print(f"  MRR (avg):       {avg_mrr:.4f} (目标 >= 0.80)")
    print(f"  NDCG@5 (avg):    {avg_ndcg:.4f} (目标 >= 0.75)")

    return {"recall": avg_recall, "mrr": avg_mrr, "ndcg": avg_ndcg}


def evaluate_generation():
    """评估生成质量：关键词匹配 + 长度合理性"""
    print("\n" + "=" * 60)
    print("  2. 生成质量评估")
    print("=" * 60)

    vs = VectorStore(
        backend=config.get("vector_store.backend", "numpy"),
        persist_dir=config.get("vector_store.persist_dir", "./data/vector_db"),
    )
    rqa = RetrievalQA(vector_store=vs, llm=MockLLM())

    # 构建 Agent 和工具
    agent_tools = build_tools(rqa, str(Path(__file__).parent.parent / "data" / "db" / "shopmind.db"))
    agent = ReActAgent(tools=agent_tools)

    generation_scores = []
    total_time = 0

    for i, case in enumerate(TEST_CASES):
        start = time.time()
        result = agent.run(case["question"])
        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        reply = result.reply
        ref = case["reference"]

        # 简化 ROUGE-L (LCS比例)
        lcs_len = _lcs_length(reply, ref)
        rouge_l = lcs_len / max(len(ref), 1)

        # 关键词重叠率
        ref_words = set(case.get("relevant_keywords", []))
        reply_lower = reply.lower()
        overlap = sum(1 for w in ref_words if w.lower() in reply_lower) / max(len(ref_words), 1)

        # 综合评分
        quality = rouge_l * 0.5 + overlap * 0.3 + (1.0 if 30 < len(reply) < 2000 else 0.5) * 0.2
        generation_scores.append(quality)

        status = "PASS" if quality >= 0.4 else "WARN"
        print(f"  [{status}] Q{i+1}: {case['question'][:40]}... | 生成质量={quality:.3f} | {elapsed:.0f}ms ({result.total_steps}步)")

    avg_quality = sum(generation_scores) / len(generation_scores) if generation_scores else 0
    avg_latency = total_time / len(TEST_CASES)

    print(f"\n  --- 生成质量汇总 ---")
    print(f"  平均质量分: {avg_quality:.4f} (目标 ROUGE-L >= 0.40)")
    print(f"  平均延迟:   {avg_latency:.0f}ms (目标 < 2000ms)")
    print(f"  总步数:     {sum(1 for c in TEST_CASES)} 个测试用例")

    return {"quality": avg_quality, "latency_ms": avg_latency}


def evaluate_end_to_end():
    """端到端系统测试"""
    print("\n" + "=" * 60)
    print("  3. 端到端系统测试")
    print("=" * 60)

    vs = VectorStore(backend=config.get("vector_store.backend", "numpy"), persist_dir=config.get("vector_store.persist_dir"))
    if vs.get_collection_size() == 0:
        knowledge_dir = Path(__file__).parent.parent / "data" / "knowledge"
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader(512, 64)
        docs = loader.load_directory(str(knowledge_dir))
        vs.add_documents(docs, "ecommerce_knowledge")

    rqa = RetrievalQA(vector_store=vs, llm=MockLLM())
    agent_tools = build_tools(rqa, str(Path(__file__).parent.parent / "data" / "db" / "shopmind.db"))
    agent = ReActAgent(tools=agent_tools)

    print(f"  向量库文档数: {vs.get_collection_size()}")
    print(f"  可用工具: {agent_tools.get_tool_names()}")
    print(f"  最大迭代步数: {agent.max_iterations}")

    # 交互式测试
    test_questions = [
        "蓝牙耳机怎么退货？",
        "充电宝的容量有多大？",
        "帮我查一下U10001买了什么",
        "支持花呗付款吗？",
    ]

    for q in test_questions:
        print(f"\n  Q: {q}")
        result = agent.run(q)
        print(f"  A: {result.reply[:200]}...")
        print(f"     (steps={result.total_steps}, {result.total_time_ms}ms)")

    # 并发模拟
    print(f"\n  并发测试 (模拟10用户)...")
    start = time.time()
    for _ in range(10):
        agent.run("蓝牙耳机续航多久")
    elapsed = time.time() - start
    qps = 10 / elapsed if elapsed > 0 else 0
    print(f"  10次请求总耗时: {elapsed:.2f}s, QPS: {qps:.1f} (目标 >= 5)")

    return {"qps": qps}


def _lcs_length(a: str, b: str) -> int:
    """最长公共子序列长度（简化版ROUGE-L）"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def np_log2(x: float) -> float:
    import math
    return math.log2(x) if x > 0 else 0


if __name__ == "__main__":
    print("=" * 60)
    print("  ShopMind 三维评估报告")
    print("=" * 60)

    retrieval_metrics = evaluate_retrieval()
    generation_metrics = evaluate_generation()
    e2e_metrics = evaluate_end_to_end()

    print("\n" + "=" * 60)
    print("  评估总结")
    print("=" * 60)
    print(f"  检索 Recall@5:   {retrieval_metrics['recall']:.4f}  (目标 >= 0.85)")
    print(f"  检索 MRR:         {retrieval_metrics['mrr']:.4f}  (目标 >= 0.80)")
    print(f"  检索 NDCG@5:      {retrieval_metrics['ndcg']:.4f}  (目标 >= 0.75)")
    print(f"  生成质量:         {generation_metrics['quality']:.4f}  (目标 ROUGE-L >= 0.40)")
    print(f"  生成延迟:         {generation_metrics['latency_ms']:.0f}ms  (目标 < 2000ms)")
    print(f"  吞吐量 QPS:       {e2e_metrics['qps']:.1f}  (目标 >= 5)")

    results = {
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "e2e": e2e_metrics,
    }
    output_path = Path(__file__).parent.parent / "logs" / "eval_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  详细结果已保存至: {output_path}")

"""Agent 工具层 —— RAG问答工具 + 消费报告生成工具"""
import sqlite3
import json
from pathlib import Path
from typing import Callable, Optional


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: dict = {}
        self._descriptions: dict = {}

    def register(self, name: str, description: str, func: Callable):
        self._tools[name] = func
        self._descriptions[name] = description

    def get(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def get_all_descriptions(self) -> str:
        lines = []
        for name, desc in self._descriptions.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def get_tool_names(self) -> str:
        return ", ".join(self._tools.keys())


def create_rag_qa_tool(retrieval_qa):
    """创建 RAG 知识问答工具"""
    def tool(query: str) -> str:
        if not query or not query.strip():
            return "请输入您想咨询的问题。"
        try:
            result = retrieval_qa.invoke({"query": query.strip()})
            answer = result.get("result", "抱歉，暂时无法回答该问题。")

            # 清理 MockLLM 可能返回的 Thought/Action 前缀
            import re
            # 去掉 "Final Answer: " 前缀
            fa_match = re.search(r"Final Answer:\s*(.+)", answer, re.DOTALL)
            if fa_match:
                answer = fa_match.group(1).strip()
            # 去掉可能残留的 Thought/Action 行
            answer = re.sub(r"^Thought:.*\n", "", answer)
            answer = re.sub(r"^Action:.*\n", "", answer)
            answer = re.sub(r"^Action Input:.*\n", "", answer)

            return answer.strip() or "抱歉，暂时无法回答该问题。"
        except Exception as e:
            return f"知识库查询出错: {str(e)}，请稍后重试或联系人工客服。"
    return tool


def create_report_tool(db_path: str = None):
    """创建消费报告生成工具"""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "db" / "shopmind.db"

    def tool(user_id: str) -> str:
        if not user_id or not user_id.strip():
            return "请提供您的用户ID（可在个人中心查看）。"

        user_id = user_id.strip()
        db_path_str = str(db_path)

        try:
            conn = sqlite3.connect(db_path_str)
            conn.row_factory = sqlite3.Row

            # 查询用户信息
            user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if not user:
                return f"未找到用户 {user_id} 的消费记录。请确认用户ID是否正确，或联系客服确认。"

            # 查询订单
            orders = conn.execute(
                "SELECT * FROM orders WHERE user_id = ? ORDER BY order_date DESC",
                (user_id,)
            ).fetchall()

            if not orders:
                return f"用户 {user['name']} 暂无消费记录。"

            # 统计分析
            total_spent = sum(o["amount"] for o in orders)
            completed = sum(1 for o in orders if o["status"] == "已完成")
            categories = {}
            for o in orders:
                cat = o["category"]
                categories[cat] = categories.get(cat, 0) + o["amount"]

            top_cat = max(categories, key=categories.get)
            recent = orders[:5]

            # 月度趋势
            monthly = {}
            for o in orders:
                month = o["order_date"][:7]
                monthly[month] = monthly.get(month, 0) + o["amount"]
            sorted_months = sorted(monthly.items())

            conn.close()

            # 生成报告
            report = f"""
## {user['name']} 的消费行为分析报告

### 基本信息
- 会员等级: **{user['level']}**
- 注册时间: {user['register_date']}
- 累计订单: **{user['total_orders']} 笔**
- 总消费金额: **{total_spent:.2f} 元**

### 品类偏好
"""
            for cat, amt in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                pct = amt / total_spent * 100 if total_spent > 0 else 0
                report += f"- {cat}: {amt:.2f} 元 ({pct:.0f}%)\n"

            report += f"""
**最常购买品类**: {top_cat}

### 月度消费趋势
"""
            for month, amt in sorted_months[-6:]:
                bar = "█" * int(amt / 100)
                report += f"- {month}: {bar} {amt:.2f} 元\n"

            report += f"""
### 最近购买
"""
            for o in recent:
                report += f"- [{o['order_date']}] {o['product_name']} - {o['amount']:.2f} 元 ({o['status']})\n"

            report += f"""
### 个性化推荐
基于您的消费习惯，建议关注:
- **{top_cat}** 品类新品（您在该品类消费最多）
- 同价位商品推荐（您的平均消费约 {total_spent / max(len(orders), 1):.0f} 元/单）

如需更详细的消费分析或商品推荐，请随时联系客服！
"""
            return report.strip()

        except Exception as e:
            return f"生成报告时出错: {str(e)}，请稍后重试。"
        finally:
            try:
                conn.close()
            except Exception:
                pass

    return tool


def build_tools(retrieval_qa, db_path: str = None) -> ToolRegistry:
    """构建工具注册中心"""
    registry = ToolRegistry()
    registry.register(
        "商品知识问答",
        "用于查询商品参数、使用说明、规格、退换货政策等知识。输入为自然语言问题，例如：'蓝牙耳机的续航时间'、'如何申请退货'。",
        create_rag_qa_tool(retrieval_qa),
    )
    registry.register(
        "消费报告生成",
        "根据用户消费记录生成消费行为分析报告。输入为用户ID，例如：'U10001'。",
        create_report_tool(db_path),
    )
    return registry

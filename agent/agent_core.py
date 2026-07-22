"""ReAct Agent 推理执行引擎

实现完整的 Thought → Action → Observation 推理循环。
支持 LLM 驱动推理和规则降级两种模式。
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from .tools import ToolRegistry


@dataclass
class AgentStep:
    """单步推理记录"""
    step_num: int
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    final_answer: str = ""


@dataclass
class AgentResult:
    """Agent 执行结果"""
    reply: str
    agent_trace: List[Dict] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    total_steps: int = 0
    total_time_ms: float = 0


class MockLLM:
    """Mock LLM - 模拟 Qwen-Plus 推理"""

    def invoke(self, prompt: str) -> str:
        return self._reason(prompt)

    def _reason(self, prompt: str) -> str:
        # RAG 上下文：提取知识库内容，只返回最相关的段落
        if "## 知识库内容" in prompt and "## 用户问题" in prompt:
            ctx_match = re.search(r"## 知识库内容\n(.+?)(?:\n## |\Z)", prompt, re.DOTALL)
            if ctx_match:
                knowledge = ctx_match.group(1).strip()
                if knowledge and len(knowledge) > 30:
                    # 只取第一个 paragraph（最佳匹配），最多400字
                    first_para = knowledge.split("\n\n")[0] if "\n\n" in knowledge else knowledge
                    # 如果第一段太短，加上第二段
                    if len(first_para) < 100 and len(knowledge.split("\n\n")) > 1:
                        first_para += "\n\n" + knowledge.split("\n\n")[1]
                    return first_para[:500]

        # 检查是否已获得工具观察结果（第2轮推理）
        # 找最后一个 Observation: —— 避免匹配到 prompt 模板中的示例文本
        obs_matches = list(re.finditer(r"^Observation:\s*(.+?)(?:\n\n|\n用户:|\Z)", prompt, re.DOTALL | re.MULTILINE))
        if obs_matches:
            obs_match = obs_matches[-1]  # 取最后一个
            observation = obs_match.group(1).strip()
            # 跳过模板中的示例文本
            if observation and len(observation) > 20 and "错误" not in observation[:20] and "工具返回结果" not in observation[:10]:
                summary = observation[:600]
                return f"Thought: 工具返回了详细信息，我可以基于这些信息回答用户的问题。\nFinal Answer: {summary}"

        # 从 prompt 中提取用户问题（第1轮推理）
        # 兼容多种格式: "用户: xxx" / "用户问题: xxx" / "## 用户问题\nxxx"
        user_input = ""
        for pattern in [r"用户[：:]\s*(.+?)(?:\n|$)", r"用户问题[：:\s]+(.+?)(?:\n|$)", r"##\s*用户问题\s*\n\s*(.+?)(?:\n|$)"]:
            m = re.search(pattern, prompt)
            if m:
                user_input = m.group(1).strip()
                break

        if not user_input:
            return "Final Answer: 您好！请问有什么可以帮助您的吗？"

        question = user_input.lower()

        # 推理：决定使用哪个工具
        report_keywords = ["买了什么", "买了", "消费", "账单", "报告", "花了多少", "购买记录", "订单"]
        should_report = any(kw in question for kw in report_keywords)

        if should_report:
            # 提取 user_id（不设默认值，让工具层处理无效ID）
            uid_match = re.search(r"[Uu]\d{5}", user_input)
            user_id = uid_match.group(0).upper() if uid_match else user_input.strip()
            return f"""Thought: 用户想查看消费记录和消费报告。我需要调用消费报告生成工具来获取用户的消费分析。
Action: 消费报告生成
Action Input: {user_id}"""

        # 其他问题 → RAG 知识问答
        # 判断具体意图
        if any(kw in question for kw in ["退货", "退款", "换货", "退换"]):
            return f"""Thought: 用户咨询退换货相关问题。我需要查询知识库中的退换货政策。
Action: 商品知识问答
Action Input: 退换货政策 退货流程 退款方式"""

        if any(kw in question for kw in ["发货", "物流", "快递", "送货"]):
            return f"""Thought: 用户想知道物流发货相关信息。
Action: 商品知识问答
Action Input: 发货时间 物流 快递"""

        if any(kw in question for kw in ["支付", "付款", "发票"]):
            return f"""Thought: 用户询问支付相关的问题。
Action: 商品知识问答
Action Input: 支付方式 发票申请"""

        if any(kw in question for kw in ["蓝牙", "耳机", "手表", "充电宝", "手机壳", "数据线", "规格", "参数", "保修", "质保", "清洁", "坏了", "能不能换", "质量"]):
            return f"""Thought: 用户想了解具体商品的规格和功能参数。
Action: 商品知识问答
Action Input: {user_input}"""

        if any(kw in question for kw in ["发票", "支付", "付款", "花呗", "银行卡", "积分", "会员", "PLUS"]):
            return f"""Thought: 用户咨询平台服务相关的问题。
Action: 商品知识问答
Action Input: {user_input}"""

        # 闲聊/问候 → 直接回复，不走工具
        chitchat = ["你好", "您好", "嗨", "hi", "hello", "在吗", "谢谢", "多谢", "再见", "拜拜", "你是谁", "你能做什么"]
        if any(user_input.strip().lower().startswith(c) or user_input.strip() == c for c in chitchat):
            return f"Final Answer: 您好！我是 ShopMind 智能客服助手。我可以帮您查询商品信息、退换货政策，也可以生成您的消费报告。请问有什么可以帮您？"

        # 默认 → RAG 知识问答
        return f"""Thought: 用户咨询了一个通用问题，我需要从知识库中检索相关信息来回答。
Action: 商品知识问答
Action Input: {user_input}"""


class ReActAgent:
    """ReAct Agent 推理执行引擎

    核心循环:
    1. LLM 生成 Thought + Action
    2. 执行工具调用
    3. 解析观察结果
    4. 判断是否得到 Final Answer 或需要继续循环
    """

    REACT_PATTERN = re.compile(
        r"Thought:\s*(.+?)\n\s*Action:\s*(.+?)\n\s*Action Input:\s*(.+?)(?:\n|$)",
        re.DOTALL | re.IGNORECASE,
    )
    FINAL_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)

    def __init__(self, tools: ToolRegistry, llm=None, max_iterations: int = 5):
        self.tools = tools
        self.llm = llm or MockLLM()
        self.max_iterations = max_iterations

    def run(self, message: str, session_id: str = None) -> AgentResult:
        """执行 ReAct 推理循环"""
        start_time = time.time()
        trace = []
        tool_calls_record = []
        current_input = message

        # 构建对话 prompt
        prompt = self._build_prompt(message)

        for step_num in range(self.max_iterations):
            agent_step = AgentStep(step_num=step_num + 1)

            # Step 1: LLM 推理
            try:
                llm_output = self.llm.invoke(prompt)
            except Exception as e:
                agent_step.final_answer = "抱歉，系统暂时无法处理您的问题，请稍后重试或联系人工客服。"
                trace.append(agent_step)
                break

            # Step 2: 检查是否为 Final Answer
            final_match = self.FINAL_PATTERN.search(llm_output)
            if final_match:
                agent_step.final_answer = final_match.group(1).strip()
                agent_step.thought = "得到最终答案"
                trace.append(agent_step)
                break

            # Step 3: 解析 Thought + Action
            match = self.REACT_PATTERN.search(llm_output)
            if match:
                thought = match.group(1).strip()
                action_name = match.group(2).strip()
                action_input = match.group(3).strip()

                agent_step.thought = thought
                agent_step.action = action_name
                agent_step.action_input = action_input

                # Step 4: 执行工具
                tool_func = self.tools.get(action_name)
                if tool_func:
                    try:
                        observation = tool_func(action_input)
                        agent_step.observation = observation
                        tool_calls_record.append({
                            "step": step_num + 1,
                            "tool": action_name,
                            "input": action_input,
                            "output": observation[:300],
                        })
                    except Exception as e:
                        observation = f"工具执行出错: {str(e)}"
                        agent_step.observation = observation
                else:
                    observation = f"错误: 未找到工具 '{action_name}'。可用工具: {self.tools.get_tool_names()}"
                    agent_step.observation = observation

                # 将观察结果加入 prompt 继续循环
                prompt += f"\nObservation: {observation}\n"
            else:
                # 无法解析 → 当作最终回答
                agent_step.final_answer = llm_output.strip()
                trace.append(agent_step)
                break

            trace.append(agent_step)

            # 检查是否有足够信息得出答案
            if self._has_sufficient_info(agent_step):
                break

        # 生成最终回复
        if not trace:
            reply = "抱歉，我不太理解您的问题，能换一种方式描述吗？"
        elif trace[-1].final_answer:
            reply = trace[-1].final_answer
        else:
            # 从最后的观察中提取回答
            reply = self._extract_final_reply(trace)

        elapsed = (time.time() - start_time) * 1000

        # 构建 trace 时过滤空字段
        trace_data = []
        for s in trace:
            entry = {"step": s.step_num, "thought": s.thought}
            if s.action:
                entry["action"] = s.action
                entry["action_input"] = s.action_input
            if s.observation:
                entry["observation"] = s.observation[:500]
            if s.final_answer:
                entry["final_answer"] = s.final_answer[:500]
            trace_data.append(entry)

        return AgentResult(
            reply=reply,
            agent_trace=trace_data,
            tool_calls=tool_calls_record,
            total_steps=len(trace),
            total_time_ms=round(elapsed, 2),
        )

    def _build_prompt(self, message: str) -> str:
        return f"""你是一个电商智能客服助手。请使用以下工具来回答用户问题。

可用工具:
{tools.get_all_descriptions()}

工具名称: {tools.get_tool_names()}

严格按照以下格式回复:
Thought: 我需要思考如何回答这个问题...
Action: 工具名称
Action Input: 工具输入参数
Observation: 工具返回结果...
... (可重复Thought/Action/Observation)
Thought: 我现在有足够信息来回答
Final Answer: 最终回复

用户: {message}"""

    def _has_sufficient_info(self, step: AgentStep) -> bool:
        """判断是否有足够信息结束循环"""
        if not step.observation:
            return False
        obs = step.observation
        # 如果工具返回了实质性内容且足够长，认为信息充足
        if len(obs) > 200 and "抱歉" not in obs[:50]:
            return True
        # 如果已经得到报告或具体回答
        if "报告" in step.action and len(obs) > 100:
            return True
        return False

    def _extract_final_reply(self, trace: List[AgentStep]) -> str:
        """从 Agent trace 中提取最终回答"""
        if not trace:
            return "抱歉，我无法处理您的问题。"

        last = trace[-1]
        if last.observation:
            return last.observation

        # 综合所有步骤的信息
        parts = []
        for step in trace:
            if step.observation and len(step.observation) > 20:
                parts.append(step.observation)

        if parts:
            return "\n\n".join(parts[-2:])  # 返回最后2步的观察
        return "抱歉，暂时无法为您提供准确的回答。建议联系人工客服获得帮助。"

    def run_simple(self, message: str) -> str:
        """简化接口 - 直接返回回答文本"""
        result = self.run(message)
        return result.reply


# 全局工具注册实例
tools = ToolRegistry()

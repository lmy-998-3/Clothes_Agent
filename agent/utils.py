"""
工具函数模块 - 添加带上下文的LLM工具
"""
# import os
import json
import re
# import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
# from PIL import Image
# from io import BytesIO
from config import config


def format_results(results: Dict[str, Any]) -> str:
    """格式化结果显示"""
    if not results:
        return "❌ 没有生成结果"

    output = []
    output.append("\n" + "=" * 60)
    output.append("📋 处理结果")
    output.append("=" * 60)

    # 换衣结果
    if "swapped_result" in results:
        output.append("\n👕 换衣结果:")
        output.append(f"   {results['swapped_result']}")

    # 姿势结果
    if "pose_results" in results and results["pose_results"]:
        output.append("\n📸 套图结果:")
        for pose, url in results["pose_results"].items():
            pose_name = get_pose_display_name(pose)
            output.append(f"\n   {pose_name}:")
            output.append(f"     {url}")

    # 统计信息
    pose_count = len(results.get("pose_results", {}))
    output.append("\n" + "-" * 60)
    output.append(f"总计: 1张换衣图 + {pose_count}张姿势图")
    output.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 60)

    return "\n".join(output)


def get_pose_display_name(pose_code: str) -> str:
    """获取姿势的显示名称"""
    display_names = {
        'default': '1. 默认姿势',
        'front_1': '2. 正面全身',
        'front_2': '3. 正面半身',
        'side': '4. 侧面',
        'back': '5. 背面'
    }
    return display_names.get(pose_code, pose_code)

class LLMClient:
    """增强的LLM客户端 - 支持上下文"""

    def __init__(self):
        self.api_key = config.PRO_API_KEY
        self.base_url = config.PRO_BASE_URL
        self.model = config.PRO_MODEL

    def chat_completion(self, messages: List[Dict[str, str]],
                       temperature: float = 0.7,
                       max_tokens: int = 500) -> Optional[str]:
        """
        调用Gemini API进行对话
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # 适配Gemini API格式
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('choices') and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            else:
                print(f"❌ LLM API调用失败: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ LLM调用异常: {str(e)}")

        return None

    def enhanced_analyze_intent_with_context(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        带上下文的意图分析 - 增强版

        Args:
            user_message: 用户当前消息
            context: 完整的对话上下文

        Returns:
            意图分析结果
        """
        # 构建智能上下文系统提示
        system_prompt = """你是一个智能对话理解助手，分析用户意图时要考虑完整的对话上下文。

对话状态分析规则：
1. 如果用户使用"这个"、"那位"等指代词，参考最近提到的实体
2. 如果对话刚上传了图片，用户的问题很可能与之相关
3. 根据对话阶段判断用户意图（咨询、生成、确认等）

请准确识别以下意图类型：
1. clothing_consult - 服装咨询（搭配、适合人群、风格等）
2. generate_request - 生成请求（试穿、展示、效果图等）
3. image_upload - 图片上传
4. pose_selection - 姿势选择（正面、侧面、全部等）
5. confirmation - 确认操作（是、好的、开始等）
6. question - 功能咨询
7. unclear - 需要澄清

关键上下文信息：
{context_info}

请返回JSON格式：
{{
    "intent_type": "意图类型",
    "confidence": 置信度0-1,
    "requires_tool": true/false,
    "requires_clarification": true/false,
    "referenced_entity": "引用的实体类型",
    "extracted_info": {{}},
    "recommended_action": "answer/collect_info/call_tool/clarify/update_poses",
    "response_guide": "建议回复方向"
}}"""

        # 格式化上下文信息
        context_info = self._format_context_for_llm(context)

        # 填充系统提示
        system_prompt = system_prompt.format(context_info=context_info)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self.chat_completion(messages, temperature=0.2, max_tokens=800)

        if response:
            try:
                # 查找JSON部分
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())

                    # 后处理：根据关键词调整置信度
                    result = self._post_process_intent(user_message, result, context)
                    return result

            except Exception as e:
                print(f"❌ 解析意图JSON失败: {e}")

        # 降级到规则分析
        return self._rule_based_intent_with_context(user_message, context)

    def _format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """格式化上下文信息供LLM使用"""
        if not context:
            return "无上下文信息"

        context_parts = []

        # 1. 当前模式
        mode = context.get("mode", "unknown")
        context_parts.append(f"当前模式: {mode}")

        # 2. 已收集的信息
        collected = context.get("collected_info", {})
        if collected.get("has_model"):
            context_parts.append("✓ 已上传模特图片")
        if collected.get("has_clothes"):
            context_parts.append("✓ 已上传服装图片")

        # 3. 对话历史（最近3条）
        history = context.get("conversation_history", [])
        if history:
            context_parts.append("\n最近对话：")
            for i, msg in enumerate(history[-3:]):
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")[:100]  # 截断
                context_parts.append(f"{role}: {content}")

        # 4. 话题链（如果有）
        topic_chain = context.get("topic_chain", [])
        if topic_chain:
            context_parts.append("\n当前话题：")
            for topic in topic_chain[-2:]:
                keywords = topic.get("keywords", [])
                if keywords:
                    context_parts.append(f"- {', '.join(keywords[:3])}")

        return "\n".join(context_parts)

    def _post_process_intent(self, user_message: str, intent_result: Dict, context: Dict) -> Dict:
        """后处理意图结果，提高准确性 - 修复版"""
        user_lower = user_message.lower()

        # 首先检查是否是明确的咨询问题（优先级最高）
        consult_patterns = [
            ("适合什么", "clothing_consult"),
            ("适合人群", "clothing_consult"),
            ("适合季节", "clothing_consult"),
            ("搭配建议", "clothing_consult"),
            ("什么风格", "clothing_consult"),
            ("好不好看", "clothing_consult"),
            ("怎么样", "clothing_consult"),
            ("推荐", "clothing_consult")
        ]

        for pattern, intent_type in consult_patterns:
            if pattern in user_lower:
                intent_result["intent_type"] = intent_type
                intent_result["confidence"] = 0.9
                # 如果上下文有服装图片，标记引用
                if context and context.get("collected_info", {}).get("has_clothes"):
                    intent_result["referenced_entity"] = "clothes_image"
                break

        # 特殊处理指代词 - 只有在不是确认的情况下
        if intent_result.get("intent_type") != "confirmation":
            if any(word in user_lower for word in ["这个", "那位", "刚才", "之前", "那张", "这件"]):
                # 如果有上下文，可能是咨询已上传的图片
                collected = context.get("collected_info", {})

                if "模特" in user_lower and collected.get("has_model"):
                    intent_result["referenced_entity"] = "model_image"
                    intent_result["confidence"] = min(intent_result.get("confidence", 0.5) + 0.2, 1.0)

                elif "衣服" in user_lower or "服装" in user_lower or "裙子" in user_lower or "上衣" in user_lower:
                    if collected.get("has_clothes"):
                        intent_result["referenced_entity"] = "clothes_image"
                        intent_result["confidence"] = min(intent_result.get("confidence", 0.5) + 0.2, 1.0)
                        # 如果是服装相关词，很可能是咨询
                        if intent_result.get("intent_type") not in ["clothing_consult", "generate_request"]:
                            intent_result["intent_type"] = "clothing_consult"

        # 处理确认词 - 更严格的判断
        # 只有当消息很短且是明确的确认词时才判断为确认
        short_confirm_words = ["是", "好的", "可以", "行", "ok", "yes", "y", "确认"]
        start_words = ["开始", "开始生成", "生成", "开始处理", "go", "start"]

        # 只有消息很短（<10字符）且完全是确认词时才判断为确认
        if len(user_lower) < 10:
            if any(word == user_lower.strip() for word in short_confirm_words):
                # 检查上下文模式
                mode = context.get("mode", "consult")
                if mode == "execute":
                    # 在执行模式下，很可能是确认生成
                    intent_result["intent_type"] = "confirmation"
                    intent_result["confidence"] = 0.95
                    intent_result["extracted_info"] = {"action": "start_generation"}
                else:
                    intent_result["intent_type"] = "confirmation"
                    intent_result["confidence"] = 0.9
            elif any(word == user_lower.strip() for word in start_words):
                # "开始"类词汇
                mode = context.get("mode", "consult")
                if mode == "execute":
                    intent_result["intent_type"] = "confirmation"
                    intent_result["confidence"] = 0.95
                    intent_result["extracted_info"] = {"action": "start_generation"}

        # 处理姿势选择词
        pose_words = ["正面", "侧面", "背面", "全身", "半身", "全部", "所有"]
        if any(word in user_lower for word in pose_words):
            if context.get("mode") == "execute":  # 在执行模式下
                if intent_result["intent_type"] not in ["pose_selection", "clothing_consult"]:
                    intent_result["intent_type"] = "pose_selection"
                    intent_result["confidence"] = 0.85

        return intent_result

    def _rule_based_intent_with_context(self, user_message: str, context: Dict) -> Dict:
        """基于规则和上下文的意图分析 - 修复版"""
        user_lower = user_message.lower()
        mode = context.get("mode", "consult") if context else "consult"

        # 首先检查是否是咨询问题（避免误判为确认）
        consult_keywords = [
            '适合', '搭配', '建议', '风格', '好看', '怎么样', '分析', '推荐',
            '休闲', '正式', '场合', '人群', '季节', '年龄', '肤色', '身材'
        ]

        # 如果包含咨询关键词，优先判断为咨询
        for keyword in consult_keywords:
            if keyword in user_lower:
                # 检查是否引用了已上传的图片
                has_model = context.get("collected_info", {}).get("has_model", False) if context else False
                has_clothes = context.get("collected_info", {}).get("has_clothes", False) if context else False

                referenced_entity = None
                if "模特" in user_lower and has_model:
                    referenced_entity = "model_image"
                elif "衣服" in user_lower and has_clothes:
                    referenced_entity = "clothes_image"

                return {
                    "intent_type": "clothing_consult",
                    "confidence": 0.85,
                    "requires_tool": False,
                    "requires_clarification": False,
                    "referenced_entity": referenced_entity,
                    "extracted_info": {},
                    "recommended_action": "answer",
                    "response_guide": "回答服装咨询问题"
                }

        # 检查是否是URL
        if user_message.startswith(('http://', 'https://')):
            return {
                "intent_type": "image_upload",
                "confidence": 0.95,
                "requires_tool": False,
                "requires_clarification": mode == "consult",
                "referenced_entity": None,
                "extracted_info": {"is_url": True},
                "recommended_action": "collect_info",
                "response_guide": "处理图片上传"
            }

        # 根据模式处理
        if mode == "execute":
            # 执行模式下，简短回复很可能是姿势选择或确认
            if len(user_lower) < 10:
                if any(word in user_lower for word in ["全部", "所有", "都要"]):
                    return {
                        "intent_type": "pose_selection",
                        "confidence": 0.9,
                        "requires_tool": False,
                        "requires_clarification": False,
                        "referenced_entity": None,
                        "extracted_info": {"selected_poses": "all"},
                        "recommended_action": "update_poses",
                        "response_guide": "用户选择全部姿势"
                    }
                elif any(word in user_lower for word in ["是", "好的", "开始"]):
                    return {
                        "intent_type": "confirmation",
                        "confidence": 0.85,
                        "requires_tool": False,
                        "requires_clarification": False,
                        "referenced_entity": None,
                        "extracted_info": {},
                        "recommended_action": "call_tool",
                        "response_guide": "用户确认开始生成"
                    }

        # 生成请求
        generate_keywords = ['生成', '制作', '试穿', '展示', '效果', '套图', '穿上', '换衣']
        if any(keyword in user_lower for keyword in generate_keywords):
            return {
                "intent_type": "generate_request",
                "confidence": 0.85,
                "requires_tool": True,
                "requires_clarification": False,
                "referenced_entity": None,
                "extracted_info": {},
                "recommended_action": "collect_info",
                "response_guide": "处理生成请求"
            }

        # 默认：需要澄清
        return {
            "intent_type": "unclear",
            "confidence": 0.5,
            "requires_tool": False,
            "requires_clarification": True,
            "referenced_entity": None,
            "extracted_info": {},
            "recommended_action": "clarify",
            "response_guide": "询问用户具体需求"
        }
        
def print_logo():
    """打印LOGO"""
    logo = """
    ╔══════════════════════════════════════════╗
    ║         👗 服装AI智能体 v2.0            ║
    ║    上下文增强版 - 智能理解对话          ║
    ╚══════════════════════════════════════════╝
    """
    print(logo)


# 创建全局LLM客户端实例
llm_client = LLMClient()
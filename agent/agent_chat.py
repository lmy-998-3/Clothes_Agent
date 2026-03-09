"""
增强的LLM对话管理 - 带完整上下文管理
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from config import config
from utils import llm_client


class LLMAgent:
    """增强的LLM对话智能体 - 带完整上下文管理"""

    def __init__(self):
        # 对话模式栈 - 支持多级状态
        self.mode_stack = ["consult"]  # consult | guide | execute
        self.user_data = {
            "model_image": None,      # 将自动从本地图库选择
            "clothes_image": None,
            "selected_model_type": config.DEFAULT_MODEL_FOLDER,  # 默认模特类型
            "requested_poses": config.SUPPORTED_POSES.copy(),  # 默认所有姿势
            "conversation_history": [],
            "last_intent": None
        }

        # 上下文记忆系统
        self.context_memory = {
            "mentioned_entities": [],  # [{"type": "model", "url": "...", "timestamp": "..."}]
            "topic_chain": [],         # [{"keywords": [], "user_input": "...", "timestamp": "..."}]
            "pending_actions": [],     # 待处理动作
            "user_preferences": {}     # 用户偏好
        }

        # 处理状态
        self.state = {
            "is_processing": False,
            "needs_clarification": False,
            "last_action": None,
            "ready_to_generate": False  # 新增：准备生成标记
        }

    def process_message(self, user_input: str) -> str:
        """处理用户输入 - 带完整上下文"""
        # 1. 更新对话历史
        self.user_data["conversation_history"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        # 2. 更新话题链
        self._update_topic_chain(user_input)

        # 3. 分析意图（带完整上下文）
        context = self._build_complete_context()
        intent_result = llm_client.enhanced_analyze_intent_with_context(user_input, context)

        # 4. 保存意图记录
        self.user_data["last_intent"] = intent_result

        # 5. 更新上下文记忆
        self._update_context_memory(user_input, intent_result)

        # 6. 根据意图生成响应
        response = self._handle_intent_with_context(intent_result, user_input, context)

        # 7. 保存助手回复
        self.user_data["conversation_history"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        # 8. 更新状态
        self._update_agent_state(intent_result, user_input)

        return response

    def _build_complete_context(self) -> Dict[str, Any]:
        """构建完整的上下文"""
        return {
            "mode": self.get_mode(),
            "collected_info": self._get_collected_info(),
            "conversation_history": self.user_data["conversation_history"][-config.CONTEXT_CONFIG['max_history_messages']:],
            "mentioned_entities": self.context_memory["mentioned_entities"][-config.CONTEXT_CONFIG['max_mentioned_entities']:],
            "topic_chain": self.context_memory["topic_chain"][-config.CONTEXT_CONFIG['max_topic_chain']:],
            "last_intent": self.user_data.get("last_intent", {})
        }

    def _update_topic_chain(self, user_input: str):
        """更新话题链"""
        # 提取关键词
        keywords = self._extract_keywords(user_input)

        if keywords:
            self.context_memory["topic_chain"].append({
                "keywords": keywords,
                "user_input": user_input[:50],  # 截断
                "timestamp": datetime.now().isoformat()
            })

            # 保持长度限制
            if len(self.context_memory["topic_chain"]) > config.CONTEXT_CONFIG['max_topic_chain']:
                self.context_memory["topic_chain"] = self.context_memory["topic_chain"][-config.CONTEXT_CONFIG['max_topic_chain']:]

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []

        # 服装相关关键词
        fashion_keywords = ['模特', '衣服', '服装', '搭配', '风格', '试穿', '生成', '展示', '姿势']
        for keyword in fashion_keywords:
            if keyword in text:
                keywords.append(keyword)

        # 指代关键词
        reference_keywords = ['这个', '那位', '刚才', '之前']
        for keyword in reference_keywords:
            if keyword in text:
                keywords.append(keyword)

        return keywords

    def _update_context_memory(self, user_input: str, intent_result: Dict):
        """更新上下文记忆"""
        # 记录提到的实体
        if intent_result.get("referenced_entity"):
            entity_type = intent_result["referenced_entity"]

            if entity_type == "model_image" and self.user_data["model_image"]:
                self.context_memory["mentioned_entities"].append({
                    "type": "model",
                    "url": self.user_data["model_image"],
                    "timestamp": datetime.now().isoformat()
                })
            elif entity_type == "clothes_image" and self.user_data["clothes_image"]:
                self.context_memory["mentioned_entities"].append({
                    "type": "clothes",
                    "url": self.user_data["clothes_image"],
                    "timestamp": datetime.now().isoformat()
                })

        # 保持实体记忆长度
        if len(self.context_memory["mentioned_entities"]) > config.CONTEXT_CONFIG['max_mentioned_entities']:
            self.context_memory["mentioned_entities"] = self.context_memory["mentioned_entities"][-config.CONTEXT_CONFIG['max_mentioned_entities']:]

    def _handle_intent_with_context(self, intent_result: Dict, user_input: str, context: Dict) -> str:
        """基于上下文的意图处理 - 增强版"""
        intent_type = intent_result.get("intent_type", "unclear")
        requires_clarification = intent_result.get("requires_clarification", False)
        referenced_entity = intent_result.get("referenced_entity")

        print(f"[DEBUG] 意图: {intent_type}, 引用实体: {referenced_entity}, 需要澄清: {requires_clarification}")

        # 检查是否是咨询问题的继续
        if intent_type == "clothing_consult":
            # 检查是否在继续之前的咨询对话
            last_intent = self.user_data.get("last_intent", {})
            if last_intent.get("intent_type") == "clothing_consult":
                # 连续咨询，保持上下文
                return self._continue_clothing_consult(user_input, referenced_entity, context)
            else:
                # 新的咨询
                return self._smart_clothing_consult(user_input, referenced_entity)

        # 处理需要澄清的情况
        if requires_clarification:
            return self._smart_clarify(intent_result, user_input, context)

        # 根据意图类型处理
        if intent_type == "generate_request":
            return self._smart_generate_request(user_input)

        elif intent_type == "image_upload":
            return self._smart_handle_image_upload(user_input, context)

        elif intent_type == "pose_selection":
            return self._smart_handle_pose_selection(user_input)

        elif intent_type == "confirmation":
            return self._smart_handle_confirmation(user_input, context)

        elif intent_type == "question":
            return self._answer_general_question(user_input)

        else:  # unclear or other
            return self._smart_clarify(intent_result, user_input, context)
    def _smart_clarify(self, intent_result: Dict, user_input: str, context: Dict) -> str:
        """智能澄清 - 根据上下文提供更精准的澄清"""
        mode = context.get("mode", "consult")

        # 如果是上传图片后的第一次回复
        if user_input.startswith(('http://', 'https://')):
            image_type = "服装" if self._is_likely_clothes_url(user_input) else "模特"

            # 保存图片
            if image_type == "服装":
                self.user_data["clothes_image"] = user_input
            else:
                self.user_data["model_image"] = user_input

            # 检查是否两张图片都有了
            has_both = self.user_data["model_image"] and self.user_data["clothes_image"]

            if has_both:
                return self._get_ready_for_generation_response()
            else:
                # 智能询问下一步
                return self._get_next_step_prompt(image_type)

        # 其他需要澄清的情况
        clarification_options = self._get_clarification_options(context)
        return f"我没有完全理解您的意思。请问您是想：\n{clarification_options}"

    def _get_ready_for_generation_response(self) -> str:
        """准备生成的响应（升级版：模特自动选择）"""
        self.push_mode("execute")

        # 生成姿势选择提示
        pose_options = "\n".join([
            f"- {config.POSE_DESCRIPTIONS[pose]} ({pose})"
            for pose in config.SUPPORTED_POSES
        ])

        return (f"🎉 完美！服装图片已就绪，模特将从本地图库自动选择。\n\n"
               f"请选择要生成的展示姿势：\n{pose_options}\n\n"
               f"您可以回复姿势名称（如'正面全身和侧面'）或'全部'")

    def _get_next_step_prompt(self, image_type: str) -> str:
        """获取下一步提示（升级版：模特自动选择）"""
        if image_type == "服装":
            # 现在只处理服装图片上传，模特自动选择
            self.push_mode("execute")
            clothes_style = self._analyze_clothes_style(self.user_data["clothes_image"])
            self.user_data["clothes_style"] = clothes_style
            return self._get_style_analysis_and_pose_prompt(clothes_style)
        else:
            # 如果意外上传了非服装图片，引导重新上传
            return "请上传服装图片，我会自动从本地图库选择合适的模特进行试穿展示。"

    def _analyze_clothes_style(self, clothes_url: str) -> str:
        """分析服装风格"""
        try:
            # 使用LLM分析服装风格
            system_prompt = """你是一位专业的服装风格分析师，请分析这件服装的风格特点。

请从以下维度分析：
1. 整体风格：休闲、正式、日常、时尚、运动、商务等
2. 适用场合：上班、聚会、约会、休闲、运动等
3. 颜色特点：鲜艳、素雅、经典等
4. 材质推测：棉质、丝绸、牛仔等

请用简洁的语言描述服装风格，给出1-2个风格关键词。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分析这件服装的风格：{clothes_url}"}
            ]

            response = llm_client.chat_completion(messages, temperature=0.3, max_tokens=200)

            if response:
                # 提取风格关键词
                style_keywords = []
                for keyword in ['休闲', '正式', '日常', '时尚', '运动', '商务', '优雅', '简约', '复古', '潮流']:
                    if keyword in response:
                        style_keywords.append(keyword)

                # 如果没找到关键词，使用默认
                if not style_keywords:
                    style_keywords = ['日常']

                return '、'.join(style_keywords[:2])  # 最多2个关键词
            else:
                return '日常'

        except Exception as e:
            print(f"风格分析失败: {e}")
            return '日常'

    def _get_style_analysis_and_pose_prompt(self, clothes_style: str) -> str:
        """返回风格分析结果和姿势选择提示"""
        # 基于服装风格推荐姿势
        recommended_poses = self._recommend_poses_by_style(clothes_style)

        pose_options = "\n".join([
            f"- {config.POSE_DESCRIPTIONS[pose]} ({pose})"
            for pose in config.SUPPORTED_POSES
        ])

        recommended_text = ""
        if recommended_poses:
            recommended_text = f"\n\n💡 根据您服装的'{clothes_style}'风格，我推荐生成：{recommended_poses}"

        return (f"✅ 服装图片已接收！\n\n"
               f"📋 风格分析：这件服装属于 **{clothes_style}** 风格。{recommended_text}\n\n"
               f"🎯 请选择要生成的展示姿势：\n{pose_options}\n\n"
               f"您可以回复姿势名称（如'正面全身'）或'全部'")

    def _recommend_poses_by_style(self, clothes_style: str) -> str:
        """根据服装风格推荐姿势"""
        style_recommendations = {
            '休闲': ['front_all', 'side', 'sit'],
            '日常': ['front_all', 'front_waist', 'side'],
            '时尚': ['front_all', 'back', 'selfie'],
            '正式': ['front_all', 'front_waist', 'side'],
            '商务': ['front_waist', 'side'],
            '运动': ['front_all', 'side', 'sit'],
            '优雅': ['front_all', 'back', 'selfie'],
            '简约': ['front_all', 'front_waist', 'side']
        }

        # 查找匹配的风格
        for style, poses in style_recommendations.items():
            if style in clothes_style:
                pose_names = [config.POSE_DESCRIPTIONS[p].split('（')[0] for p in poses]
                return '、'.join(pose_names)

        # 默认推荐
        return '正面全身、侧面'

    def _get_clarification_options(self, context: Dict) -> str:
        """根据上下文生成澄清选项"""
        mode = context.get("mode", "consult")
        collected = context.get("collected_info", {})

        options = []

        if mode == "consult":
            options.append("1. 咨询服装相关问题（搭配、风格等）")
            options.append("2. 生成模特试穿展示图")
            options.append("3. 了解我的功能和使用方法")

        elif mode == "guide":
            if collected.get("has_model") and not collected.get("has_clothes"):
                options.append("1. 上传服装图片")
                options.append("2. 咨询模特相关的问题")
            elif collected.get("has_clothes") and not collected.get("has_model"):
                options.append("1. 上传模特图片")
                options.append("2. 咨询服装相关的问题")
            else:
                options.append("1. 继续完成套图生成")
                options.append("2. 咨询相关问题")

        elif mode == "execute":
            options.append("1. 选择要生成的姿势")
            options.append("2. 开始生成套图")
            options.append("3. 修改图片")

        return "\n".join(options)

    def _smart_clothing_consult(self, question: str, referenced_entity: str = None) -> str:
        """智能服装咨询 - 关联已有图片 - 增强版"""
        # 构建上下文信息
        context_info = ""
        image_references = []

        if referenced_entity == "model_image" and self.user_data["model_image"]:
            context_info = f"\n\n（基于您上传的模特图片进行分析）"
            image_references.append(f"模特图片: {self.user_data['model_image']}")
        elif referenced_entity == "clothes_image" and self.user_data["clothes_image"]:
            context_info = f"\n\n（基于您上传的服装图片进行分析）"
            image_references.append(f"服装图片: {self.user_data['clothes_image']}")
        else:
            # 如果没有明确引用，检查是否有相关图片
            if "模特" in question and self.user_data["model_image"]:
                context_info = f"\n\n（基于您之前上传的模特图片进行分析）"
                image_references.append(f"模特图片: {self.user_data['model_image']}")
            elif "衣服" in question or "服装" in question:
                if self.user_data["clothes_image"]:
                    context_info = f"\n\n（基于您之前上传的服装图片进行分析）"
                    image_references.append(f"服装图片: {self.user_data['clothes_image']}")

        # 获取相关历史对话
        relevant_history = self._get_relevant_history(question)
        history_context = ""
        if relevant_history:
            history_context = "\n\n之前的对话参考："
            for msg in relevant_history:
                role = "用户" if msg["role"] == "user" else "我"
                history_context += f"\n{role}: {msg['content'][:100]}"

        # 构建提示词
        system_prompt = f"""你是一位专业的服装顾问，请基于用户的问题提供专业建议。{context_info}{history_context}

    回答要求：
    1. 提供具体、实用的建议
    2. 如果涉及已上传的图片，基于图片内容分析
    3. 保持友好自然的语气
    4. 可以适当询问更多细节以获得更精准的建议
    5. 保持对话的连贯性

    如果需要更多信息才能给出精准建议，请友好地询问。"""

        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]

        # 添加当前问题
        messages.append({"role": "user", "content": question})

        response = llm_client.chat_completion(messages, temperature=0.7, max_tokens=800)

        if response:
            # 添加自然的过渡建议
            transition = self._get_consultation_transition(question)
            return response + transition

        return "请提供更多细节，这样我可以给出更精准的服装建议。"

    def _continue_clothing_consult(self, user_input: str, referenced_entity: str = None, context: Dict = None) -> str:
        """继续服装咨询 - 保持上下文"""
        # 获取对话历史
        conversation_history = self.user_data.get("conversation_history", [])

        # 构建带完整上下文的提示
        system_prompt = """你是一位专业的服装顾问，正在继续之前的服装咨询对话。

    请基于之前的对话内容和用户的新问题，提供连贯、专业的建议。

    如果用户使用了"这件衣服"、"这个模特"等指代词，请基于之前提到的图片或描述进行分析。

    保持对话的自然流畅，提供实用的建议。"""

        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 添加最近的对话历史（最多6条）
        recent_history = conversation_history[-6:]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 添加当前问题
        messages.append({"role": "user", "content": user_input})

        response = llm_client.chat_completion(messages, temperature=0.7, max_tokens=800)

        if response:
            # 添加自然的过渡
            transition = self._get_consultation_transition(user_input)
            return response + transition

        # 降级到普通咨询
        return self._smart_clothing_consult(user_input, referenced_entity)

    def _get_relevant_history(self, current_question: str, max_messages: int = 4) -> List[Dict]:
        """获取相关历史对话 - 增强版"""
        relevant = []
        conversation_history = self.user_data.get("conversation_history", [])

        # 关键词匹配
        keywords = ["模特", "衣服", "服装", "搭配", "风格", "适合", "咨询", "建议"]
        question_keywords = []

        for keyword in keywords:
            if keyword in current_question:
                question_keywords.append(keyword)

        # 如果没有明显关键词，检查是否为连续咨询
        if not question_keywords and len(conversation_history) > 2:
            # 检查最后几条是否是咨询对话
            last_messages = conversation_history[-3:]
            is_consult_continuation = False
            for msg in last_messages:
                if msg["role"] == "assistant" and any(kw in msg["content"] for kw in ["建议", "适合", "搭配", "风格"]):
                    is_consult_continuation = True
                    break

            if is_consult_continuation:
                # 返回最近的咨询相关对话
                return last_messages

        # 基于关键词匹配
        for msg in conversation_history[-max_messages * 2:]:  # 查看更多消息
            content = msg.get("content", "").lower()

            # 检查是否相关
            is_relevant = False
            for keyword in question_keywords:
                if keyword in content:
                    is_relevant = True
                    break

            # 或者检查是否是连续的问答对
            if not is_relevant and len(relevant) > 0:
                # 如果是连续的对话，也包含进来
                last_msg = relevant[-1]
                time_diff = abs(
                    datetime.fromisoformat(msg["timestamp"]).timestamp() -
                    datetime.fromisoformat(last_msg["timestamp"]).timestamp()
                )
                if time_diff < 300:  # 5分钟内的对话认为是连续的
                    is_relevant = True

            if is_relevant:
                relevant.append(msg)
                if len(relevant) >= max_messages:
                    break

        return relevant

    def _get_consultation_transition(self, question: str) -> str:
        """获取咨询后的自然过渡"""
        transition = ""

        has_model = self.user_data["model_image"] is not None
        has_clothes = self.user_data["clothes_image"] is not None

        # 根据情况提供过渡建议
        if "模特" in question and has_model and not has_clothes:
            transition = "\n\n💡 如果您有想试穿的服装图片，我可以为您生成试穿效果图。"
        elif "衣服" in question and has_clothes and not has_model:
            transition = "\n\n💡 如果您有模特图片，我可以为您生成试穿效果图。"
        elif has_model and has_clothes and self.get_mode() != "execute":
            transition = "\n\n💡 您已经上传了模特和服装图片，可以开始生成试穿展示图了。"

        return transition

    def _smart_generate_request(self, request: str) -> str:
        """智能处理生成请求（升级版：模特自动选择）"""
        # 切换到引导模式
        self.push_mode("guide")

        # 检查已有信息
        has_clothes = self.user_data["clothes_image"] is not None

        if has_clothes:
            # 已有机服图片，直接进入姿势选择
            clothes_style = self.user_data.get("clothes_style", "日常")
            return self._get_style_analysis_and_pose_prompt(clothes_style)
        else:
            # 需要上传服装图片
            return "好的！我可以帮您生成模特试穿展示图。现在请提供您想要试穿的服装图片（模特将从本地图库自动选择）。"

    def _smart_handle_image_upload(self, url: str, context: Dict) -> str:
        """智能处理图片上传（升级版：只接受衣服图片）"""
        # 现在只接受服装图片，模特从本地图库自动选择
        self.user_data["clothes_image"] = url

        # 切换到执行模式，准备生成
        self.push_mode("execute")

        # 分析服装风格
        clothes_style = self._analyze_clothes_style(url)

        # 保存风格信息
        self.user_data["clothes_style"] = clothes_style

        # 返回风格分析结果，并询问姿势选择
        return self._get_style_analysis_and_pose_prompt(clothes_style)

    def _smart_handle_pose_selection(self, user_input: str) -> str:
        """智能处理姿势选择"""
        # 解析姿势选择
        self._parse_pose_selection(user_input)

        poses = self.user_data["requested_poses"]
        if not poses:
            return "请告诉我具体要生成哪些姿势，例如：'正面全身和侧面' 或 '全部'"

        # 显示选择结果
        poses_count = len(poses)
        pose_names = [config.POSE_DESCRIPTIONS.get(p, p) for p in poses]
        pose_list = "、".join(pose_names)

        return f"✅ 已选择 {poses_count} 种姿势：{pose_list}\n\n回复'开始'开始生成套图。"

    def _smart_handle_confirmation(self, user_input: str, context: Dict) -> str:
        """智能处理确认 - 修复版"""
        mode = context.get("mode", "consult")

        if mode == "execute":
            # 在执行模式下，"开始"表示开始生成套图
            user_lower = user_input.lower()
            if any(word in user_lower for word in ["开始", "开始生成", "生成", "开始处理", "是", "好的"]):
                # 检查是否有选择的姿势
                if self.user_data["requested_poses"]:
                    # 设置标记，让主程序知道可以开始生成
                    self.state["ready_to_generate"] = True
                    poses_count = len(self.user_data["requested_poses"])
                    return f"✅ 确认开始生成 {poses_count} 种姿势的套图！"
                else:
                    return "请先选择要生成的姿势。"
            else:
                # 其他确认词，但不明确
                return "确认什么？请具体说明，例如：'开始生成'或'是'。"
        elif mode == "guide":
            # 在引导模式下，确认可能是确认上传图片
            return "已确认收到。请继续下一步。"
        else:
            # 咨询模式下
            return "确认什么？请具体说明。"

    def _parse_pose_selection(self, user_input: str):
        """解析姿势选择"""
        user_lower = user_input.lower()

        # 清空之前的姿势选择
        self.user_data["requested_poses"] = []

        # 检查是否选择了全部姿势
        if any(word in user_lower for word in ["全部", "所有", "都要", "全要", "all", "全部姿势"]):
            self.user_data["requested_poses"] = config.SUPPORTED_POSES.copy()
            print(f"[DEBUG] 选择了全部姿势: {self.user_data['requested_poses']}")
            return

        # 检查具体姿势
        pose_mapping = {
            "默认": "default",
            "默认姿势": "default",
            "正面": "front_1",
            "正面全身": "front_1",
            "全身": "front_1",
            "半身": "front_2",
            "正面半身": "front_2",
            "侧面": "side",
            "侧身": "side",
            "背面": "back",
            "背后": "back"
        }

        selected_poses = []
        for chinese_pose, pose_code in pose_mapping.items():
            if chinese_pose in user_lower:
                if pose_code not in selected_poses:
                    selected_poses.append(pose_code)

        # 如果没有明确选择，检查英文代码
        if not selected_poses:
            for pose_code in config.SUPPORTED_POSES:
                if pose_code in user_lower:
                    selected_poses.append(pose_code)

        # 设置选择的姿势
        self.user_data["requested_poses"] = selected_poses if selected_poses else ["default", "front_1", "side"]
        print(f"[DEBUG] 解析后的姿势: {self.user_data['requested_poses']}")

    def _is_likely_clothes_url(self, url: str) -> bool:
        """判断是否是服装图片"""
        # 简单的启发式判断
        clothes_keywords = ['clothes', 'dress', 'shirt', 'clothing', 'garment', 'apparel']
        url_lower = url.lower()

        # 检查URL中是否包含服装关键词
        for keyword in clothes_keywords:
            if keyword in url_lower:
                return True

        # 检查是否是平铺图、挂牌图的常见模式
        if any(pattern in url_lower for pattern in ['flat', 'lay', 'hang', 'model_']):
            return True

        # 默认假设是模特图（因为更常见）
        return False

    def _answer_general_question(self, question: str) -> str:
        """回答一般问题"""
        system_prompt = """你是服装AI助手，回答用户关于功能、操作方法等一般问题。

常见问题解答：
1. 你能做什么？ - 可以分析服装、生成模特试穿展示图
2. 需要什么图片？ - 需要模特正面全身照和服装挂牌图/平铺图
3. 能生成哪些姿势？ - 正面全身、正面半身、侧面、背面
4. 处理需要多久？ - 通常1-2分钟
5. 图片要求？ - 建议纯色背景，图片清晰

请友好、清晰地回答问题。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        response = llm_client.chat_completion(messages)
        return response or "我可以帮您分析服装或生成试穿展示图。有什么具体问题吗？"

    def _get_collected_info(self) -> Dict[str, Any]:
        """获取已收集的信息（升级版：模特自动选择）"""
        return {
            "has_model": True,  # 模特将自动从图库选择
            "has_clothes": self.user_data["clothes_image"] is not None,
            "has_poses": len(self.user_data.get("requested_poses", [])) > 0,
            "selected_model_type": self.user_data.get("selected_model_type", config.DEFAULT_MODEL_FOLDER)
        }

    def _update_agent_state(self, intent_result: Dict, user_input: str = ""):
        """更新智能体状态"""
        intent_type = intent_result.get("intent_type")

        if intent_type == "generate_request":
            self.state["needs_clarification"] = False
        elif intent_type == "unclear" or intent_result.get("requires_clarification"):
            self.state["needs_clarification"] = True
        else:
            self.state["needs_clarification"] = False

        self.state["last_action"] = intent_type

        # 如果是确认开始，重置标记
        if intent_type == "confirmation" and "开始" in user_input.lower():
            self.state["ready_to_generate"] = True

    # 模式管理方法
    def push_mode(self, new_mode: str):
        """推入新模式"""
        if len(self.mode_stack) < config.CONTEXT_CONFIG['state_stack_limit']:
            self.mode_stack.append(new_mode)

    def pop_mode(self) -> str:
        """弹出模式"""
        if len(self.mode_stack) > 1:
            return self.mode_stack.pop()
        return self.mode_stack[0]

    def get_mode(self) -> str:
        """获取当前模式"""
        return self.mode_stack[-1]

    # 状态检查方法
    def is_ready_to_generate(self) -> bool:
        """检查是否可以开始生成（升级版：模特自动选择）"""
        return (self.user_data["clothes_image"] is not None and
                self.get_mode() == "execute" and
                len(self.user_data["requested_poses"]) > 0)

    def set_processing_complete(self, results: Dict):
        """设置处理完成"""
        self.state["is_processing"] = False
        self.state["ready_to_generate"] = False  # 重置准备生成标记
        self.user_data["results"] = results
        self.mode_stack = ["consult"]  # 回到咨询模式

    def get_user_data(self) -> Dict:
        return self.user_data

    def reset(self):
        """重置对话"""
        self.__init__()
        # 确保状态标记也被重置
        self.state["ready_to_generate"] = False
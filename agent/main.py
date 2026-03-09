"""
LLM服装AI智能体 - 主程序
上下文增强版
"""
import time
import threading
import json
from typing import Dict, List, Optional

from config import config
from image_processor import processor
from agent_chat import LLMAgent
from utils import format_results, llm_client, print_logo


class FashionAgent:
    """增强的服装AI智能体"""

    def __init__(self):
        print_logo()
        print("\n🤖 智能体已启动！")
        print("📱 支持：服装咨询 + 套图生成\n")

    def run_cli(self):
        """运行增强的CLI交互"""
        agent = LLMAgent()

        print("=" * 50)
        print("💬 增强交互模式")
        print("=" * 50)
        print("\n输入 '退出' 结束对话，'重置' 开始新任务，'模式' 查看当前模式\n")

        # 初始问候
        welcome_msg = """您好！我是服装AI助手，我可以：
                    1. 回答服装相关问题（适合人群、搭配建议等）
                    2. 生成模特试穿展示套图

                   请告诉我您需要什么帮助？"""
        print(f"🤖 AI: {welcome_msg}")

        while True:
            # 获取用户输入
            user_input = input("\n👉 你: ").strip()

            if user_input.lower() in ["退出", "exit", "quit"]:
                print("\n👋 再见！")
                break

            if user_input.lower() in ["重置", "reset", "重新开始"]:
                agent.reset()
                print("\n🔄 对话已重置")
                print("🤖 AI: 已重置，请问您需要什么帮助？")
                continue

            if user_input.lower() in ["模式", "mode", "状态"]:
                current_mode = agent.get_mode()
                mode_desc = {
                    "consult": "咨询模式（回答服装问题）",
                    "guide": "引导模式（收集信息生成套图）",
                    "execute": "执行模式（准备生成套图）"
                }
                print(f"当前模式: {mode_desc.get(current_mode, current_mode)}")
                continue

            # 检查是否在处理中
            if agent.state.get("is_processing", False):
                print("⏳ 正在处理中，请稍候...")
                time.sleep(3)
                continue

            # 处理用户输入
            response = agent.process_message(user_input)
            print(f"\n🤖 AI: {response}")

            # 检查是否可以开始生成
            if agent.is_ready_to_generate() or agent.state.get("ready_to_generate", False):
                user_data = agent.get_user_data()
                poses = user_data.get("requested_poses", [])

                if poses:
                    # 显示将要生成的姿势
                    print(f"\n📋 将要生成的姿势：")
                    for pose in poses:
                        desc = config.POSE_DESCRIPTIONS.get(pose, pose)
                        print(f"  - {desc}")

                    # 检查是否已经确认过
                    if agent.state.get("ready_to_generate", False):
                        print("\n🔄 检测到确认指令，自动开始生成套图...")
                        self._start_generation(agent)
                        # 重置标记
                        agent.state["ready_to_generate"] = False
                    else:
                        print("\n🔄 准备开始生成套图...")
                        confirm = input("是否开始生成？(是/否): ").strip().lower()

                        if confirm in ["是", "yes", "y", ""]:
                            self._start_generation(agent)
                        else:
                            print("🤖 AI: 好的，随时告诉我何时开始生成。")
                else:
                    print("🤖 AI: 请先选择要生成的姿势类型。")

    def _start_generation(self, agent):
        """开始生成套图"""
        user_data = agent.get_user_data()

        print("\n" + "=" * 50)
        print("🚀 开始生成套图")
        print("=" * 50)

        # 在新线程中处理
        thread = threading.Thread(
            target=self._process_generation,
            args=(agent, user_data)
        )
        thread.daemon = True
        thread.start()

    def _process_generation(self, agent, user_data):
        """处理生成任务"""
        try:
            agent.state["is_processing"] = True
            agent.state["ready_to_generate"] = False  # 重置准备生成标记

            # 自动从本地图库选择模特图片
            model_type = user_data.get("selected_model_type", config.DEFAULT_MODEL_FOLDER)
            model_url = config.get_first_model_image(model_type)
            if not model_url:
                print("❌ 无法从本地模特图库获取图片")
                return

            clothes_url = user_data["clothes_image"]
            poses = user_data.get("requested_poses", config.SUPPORTED_POSES)

            print(f"\n📊 任务信息:")
            print(f"   模特类型: {model_type}")
            print(f"   模特图片: {model_url}")
            print(f"   服装图片: {clothes_url[:50]}...")
            print(f"   生成姿势: {len(poses)}种")

            # 1. 换衣处理
            print("\n1️⃣ 第一步：换衣处理...")
            swapped_result = processor.change_clothes(model_url, clothes_url)

            if not swapped_result:
                print("❌ 换衣处理失败")
                return

            # 2. 生成套图
            print(f"\n2️⃣ 第二步：生成{len(poses)}种姿势...")
            set_results = processor.generate_set(swapped_result, poses)

            # 3. 完成
            result = {
                "swapped": swapped_result,
                "poses": set_results,
                "success": True
            }

            # 4. 完成后，重置准备生成标记，回到咨询模式
            # self.state["is_processing"] = False
            # self.state["ready_to_generate"] = False
            # self.user_data["results"] = results
            # self.mode_stack = ["consult"]

            agent.set_processing_complete(result)

            # 显示结果
            print("\n" + "=" * 50)
            print("🎉 生成完成！")
            print("=" * 50)

            formatted = format_results({
                "swapped_result": swapped_result,
                "pose_results": set_results
            })
            print(formatted)

            # 保存结果
            # self._save_results(result)

            # 询问下一步
            print("\n🤖 AI: 套图生成完成！您还想咨询其他服装问题吗？")

        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            agent.state["is_processing"] = False

    def run_auto_demo(self, model_url: str = None, clothes_url: str = None):
        """演示模式 - 展示两种能力"""
        print("=" * 50)
        print("🎬 增强演示模式")
        print("=" * 50)

        # 演示咨询能力
        print("\n💬 演示1：服装咨询")
        print("-" * 30)
        consult_question = "https://img4.yishouapp.com/new_clothes1.png这件衣服适合什么场合？"
        print(f"👉 用户: {consult_question}")

        # 模拟咨询回答
        system_msg = "你是一位服装顾问，分析这件连衣裙适合的场合。"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": consult_question}
        ]
        consult_response = llm_client.chat_completion(messages) or "这件连衣裙适合休闲聚会和约会场合。"
        print(f"🤖 AI: {consult_response}")

        # 演示生成能力
        print("\n💬 演示2：套图生成")
        print("-" * 30)
        print("👉 用户: 我想生成模特试穿这件衣服的展示图")
        print("🤖 AI: 好的！请提供模特照片。")
        print("👉 用户: https://img4.yishouapp.com/model1.jpg")
        print("🤖 AI: ✅ 收到！现在可以生成展示图。您想要哪些角度？")
        print("👉 用户: 正面和侧面")
        print("🤖 AI: 明白！开始处理...")

        # 使用示例图片或传入的图片
        if not model_url:
            model_url = "https://img4.yishouapp.com/ini_model.png"
        if not clothes_url:
            clothes_url = "https://img4.yishouapp.com/new_clothes.png"

        # 实际处理
        result = self._process_images(
            model_url,
            clothes_url,
            ['front_1', 'side']
        )

        if result:
            print("🎉 生成完成！")
        else:
            print("❌ 演示生成失败")

    def _process_images(self, model_url: str, clothes_url: str, poses: List[str] = None) -> Optional[Dict]:
        """处理图片的核心逻辑"""
        if poses is None:
            poses = config.SUPPORTED_POSES

        try:
            # 第一步：换衣
            print("\n1️⃣ 第一步：换衣处理...")
            swapped_result = processor.change_clothes(model_url, clothes_url)

            if not swapped_result:
                print("❌ 换衣处理失败")
                return None

            # 第二步：生成套图
            print(f"\n2️⃣ 第二步：生成{len(poses)}种姿势的套图...")
            set_results = processor.generate_set(swapped_result, poses)

            return {
                "swapped": swapped_result,
                "poses": set_results,
                "success": True
            }

        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            return None

    # def _save_results(self, results: Dict, filename: str = "output_results.json"):
    #     """保存结果到文件"""
    #     try:
    #         # 保存JSON数据
    #         with open(filename, "w", encoding="utf-8") as f:
    #             json.dump(results, f, ensure_ascii=False, indent=2)
    #
    #         # 保存可读格式
    #         txt_filename = filename.replace(".json", ".txt")
    #         formatted = format_results({
    #             "swapped_result": results.get("swapped"),
    #             "pose_results": results.get("poses", {})
    #         })
    #
    #         with open(txt_filename, "w", encoding="utf-8") as f:
    #             f.write(formatted)
    #
    #         print(f"\n📄 结果已保存到: {filename} 和 {txt_filename}")
    #
    #     except Exception as e:
    #         print(f"❌ 保存文件失败: {str(e)}")


def main():
    """主函数"""
    agent = FashionAgent()  # 创建实例

    print("请选择运行模式：")
    print("1. 🗣️  交互模式 (LLM智能对话)")
    print("2. 🚀 自动演示模式 (使用示例图片)")
    print("3. ⚡ 直接处理模式 (输入图片URL)")

    choice = input("\n请输入选项 (1/2/3): ").strip()

    if choice == "1":
        # 交互模式
        agent.run_cli()

    elif choice == "2":
        # 自动演示模式
        agent.run_auto_demo()

    elif choice == "3":
        # 直接处理模式
        model_url = input("请输入模特图片URL: ").strip()
        clothes_url = input("请输入衣服图片URL: ").strip()

        if not model_url or not clothes_url:
            print("❌ 请输入有效的URL")
            return

        agent.run_auto_demo(model_url, clothes_url)

    else:
        print("❌ 无效选项")


if __name__ == "__main__":
    main()
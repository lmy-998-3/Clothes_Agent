"""
精简配置文件 - 添加LLM配置和上下文配置
"""
import os
from typing import Optional, List
# from typing import Dict, List

class SimpleConfig:
    """简单配置类"""

    # 图片处理模型API配置
    BLTCY_API_URL = "https://api.bltcy.ai/v1/images/edits"
    # BLTCY_API_KEY = "Bearer sk-wqhpQNzt0QiL67n1J9JVXGNnOnjJJg1QxtVnUTl6EyOBfVEL"
    BLTCY_API_KEY: str = "Bearer sk-oM2ZkSBXer1S8oq9S6H5GQcndoGr0A5bw3U7ezWxyl1oxeqw"

    # LLM配置
    PRO_API_KEY = "vck_7uigPm8N2xncEqqrZnbKK4ix9QG3bqHpn3xsGjUFuWSgp8fYwL2yRjx9"
    PRO_BASE_URL = "https://ai-gateway.vercel.sh/v1"
    PRO_MODEL = "google/gemini-2.5-flash-image-preview"

    # OBS存储配置
    OBS_ACCESS_KEY_ID = "EULIXLKJQLUS62GPNIAX"
    OBS_SECRET_ACCESS_KEY = "vmoTDJVUOXVtwt8pwh5LQ6LnE2bJPDtxid7biimK"
    OBS_SERVER = "https://obs.cn-south-1.myhuaweicloud.com"
    OBS_BUCKET = "yishou-img"
    OBS_PUBLIC_HOST = "img4.yishouapp.com"

    # =====================
    # 本地图库配置（模特/姿势）
    # =====================
    # 开启后：模特/姿势参考图将从本地目录读取（每个目标文件夹取第一张图片）
    USE_LOCAL_GALLERIES = True

    # 模特图库根目录（包含 Chinese / white_people / black_people / latin_people 等子文件夹）
    MODEL_GALLERY_ROOT = r"D:\Desktop\models"
    MODEL_FOLDERS = {
        "Chinese": "Chinese",
        "white_people": "white_people",
        "black_people": "black_people",
        "latin_people": "latin_people",
    }
    DEFAULT_MODEL_FOLDER = "Chinese"

    # 姿势图库根目录（包含 front_all/front_knee/... 等子文件夹）
    POSE_GALLERY_ROOT = r"D:\DeskTop\poses"
    POSE_FOLDERS = {
        "front_all": "front_all",
        "front_knee": "front_knee",
        "front_waist": "front_waist",
        "front_noface": "front_noface",
        "back": "back",
        "side": "side",
        "sit": "sit",
        "selfie": "selfie",
        "clothes_detail": "clothes_detail",
    }

    # =====================
    # 线上姿势参考图（兼容保留）
    # =====================
    REF_MODEL_IMAGES = {
        'default': 'https://img4.yishouapp.com/ref_model_default.png',
        'front_1': 'https://img4.yishouapp.com/ref_model_front1.png',
        'front_2': 'https://img4.yishouapp.com/ref_model_front2.png',
        'side': 'https://img4.yishouapp.com/ref_model_side.png',
        'back': 'https://img4.yishouapp.com/ref_model_back.png'
    }

    # 支持的姿势类型 - 本地图库版本（带中文描述）
    # default：不做姿势转换，直接输出换衣结果
    SUPPORTED_POSES = ['default'] + list(POSE_FOLDERS.keys())
    POSE_DESCRIPTIONS = {
        'default': '默认（不转换姿势，仅输出换衣结果）',
        'front_all': '正面全身（全身展示）',
        'front_knee': '正面全身（膝盖及以上）',
        'front_waist': '正面半身（腰部及以上）',
        'front_noface': '正面（无脸示意姿势，保持模特本人脸不变）',
        'back': '背面（背对镜头）',
        'side': '侧面（侧身展示）',
        'sit': '坐姿（坐下展示）',
        'selfie': '自拍视角（上半身近景）',
        'clothes_detail': '服装细节特写（突出面料/细节）',
    }

    # 处理超时时间
    PROCESS_TIMEOUT = 120  # 秒

    # 最大图片大小
    MAX_IMAGE_SIZE_MB = 5

    # 上下文配置
    CONTEXT_CONFIG = {
        'max_history_messages': 5,      # 最大历史消息数
        'max_mentioned_entities': 3,    # 最大实体记忆数
        'max_topic_chain': 5,           # 最大话题链长度
        'state_stack_limit': 3,         # 状态栈深度限制
    }

    # LLM提示词模板
    LLM_PROMPTS = {
        'system_prompt': """你是一个多功能的服装AI助手，具有双重角色：

角色1：服装顾问
- 回答关于服装款式、搭配、适合人群等问题
- 提供服装相关的专业建议
- 分析服装图片的风格特点

角色2：套图生成助手
- 帮助用户生成模特试穿展示图
- 引导用户提供必要的服装图片（模特与姿势可从本地图库自动选择）
- 解释生成过程和预期结果

请根据用户意图选择合适的角色：
1. 如果用户询问服装相关问题 → 作为服装顾问回答
2. 如果用户想要生成展示图 → 作为套图生成助手引导
3. 如果意图不明确 → 询问澄清并提供选择

关键原则：
- 咨询问题不触发工具调用
- 只有明确表达生成意图时才启动工具
- 保持对话自然流畅""",

        'welcome': "👗 欢迎使用服装AI智能体！我可以帮您为模特图片更换服装并生成多种姿势的展示套图。",

        'ask_for_model': "（已升级）模特图片可从本地图库自动选择。如需指定人种/模特类型，请告诉我：Chinese / white_people / black_people / latin_people。",

        'ask_for_clothes': "请发送服装图片的URL（建议挂牌图或平铺图）。上传后我会先分析风格，再让您选择姿势。",

        'ask_for_pose': "请选择要生成的姿势：{poses}\n直接回复姿势中文/英文关键字即可，例如：'正面全身'、'sit'、'背面'、'全部'。",

        'pose_recommendation': "根据您上传的服装，我推荐生成以下姿势的展示图：{poses}。您想调整这个选择吗？",

        'processing': "正在为您处理，这可能需要1-2分钟。请稍候...",

        'error_understanding': "我没有完全理解您的需求。您是想生成服装展示图吗？请发送服装图片；模特与姿势我也可以从本地图库为您选择。"
    }

    # 意图类型常量
    INTENT_TYPES = {
        'clothing_consult': '服装咨询',
        'generate_request': '生成请求',
        'image_upload': '图片上传',
        'pose_selection': '姿势选择',
        'confirmation': '确认操作',
        'question': '一般问题',
        'unclear': '意图不明确'
    }

    @classmethod
    def get_pose_prompt(cls, pose_type: str) -> str:
        """获取姿势转换的提示词"""
        prompts = {
            # default：上层逻辑会跳过姿势转换
            'default': '不需要姿势转换',

            # 本地图库姿势（第一张=姿势参考图，第二张=换衣后的模特图）
            'front_all': '将第二张图中的模特姿势换为第一张图的模特姿势，只更换姿势，不改变模特本体，保持模特本人脸部与身份特征不变，保持衣服款式与颜色不变，构图为9:16全身展示',
            'front_knee': '将第二张图中的模特姿势换为第一张图的模特姿势，只更换姿势，不改变模特本体，保持脸部与衣服不变，同时拉近镜头，只展示模特膝盖及以上区域',
            'front_waist': '将第二张图中的模特姿势换为第一张图的模特姿势，只更换姿势，不改变模特本体，保持脸部与衣服不变，同时拉近镜头，只展示模特腰部及以上区域',
            'front_noface': '将第二张图中的模特身体姿势换为第一张图的姿势参考（第一张可能没有脸），只更换身体姿势与肢体动作，不改变第二张图模特的脸部与身份特征，保持衣服不变',
            'side': '将第二张图中的模特姿势换为第一张图的模特姿势，只更换姿势，不改变模特本体，保持脸部与衣服不变，呈侧身展示（侧面角度清晰）',
            'back': '将第二张图中的模特姿势换为第一张图的模特姿势，只更换姿势，不改变模特本体，保持衣服不变，背对镜头展示背面效果',
            'sit': '将第二张图中的模特姿势换为第一张图的模特坐姿，只更换姿势，不改变模特本体，保持脸部与衣服不变，坐姿自然',
            'selfie': '将第二张图中的模特姿势换为第一张图的自拍视角姿势，只更换姿势与镜头视角，不改变模特本体，保持脸部与衣服不变，上半身近景为主',
            'clothes_detail': '将第二张图生成服装细节特写，参考第一张图的构图与视角，只调整镜头距离与取景以突出服装细节（面料、领口、袖口、纽扣等），不改变模特与衣服本身'
        }
        return prompts.get(pose_type, prompts['default'])

    @classmethod
    def get_clothes_swap_prompt(cls) -> str:
        """获取换衣的提示词"""
        return ("图二是挂牌图，图一是模特，将图一模特的全套衣服，包括上衣和裤子，全部替换为图一挂牌图的衣服，"
                "其他内容保持不变，保持模特姿势不变，保持挂牌图衣服的款式，颜色不变，不得添加或减少挂牌图衣服的元素")

    # =====================
    # 本地图库读取工具方法
    # =====================
    @classmethod
    def _list_image_files(cls, dir_path: str) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        try:
            names = os.listdir(dir_path)
        except Exception:
            return []
        files = []
        for name in names:
            p = os.path.join(dir_path, name)
            if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
                files.append(p)
        files.sort()
        return files

    @classmethod
    def get_first_model_image(cls, model_folder: Optional[str] = None) -> Optional[str]:
        """从模特图库某个子目录取第一张图"""
        folder_key = model_folder or cls.DEFAULT_MODEL_FOLDER
        sub = cls.MODEL_FOLDERS.get(folder_key, folder_key)
        dir_path = os.path.join(cls.MODEL_GALLERY_ROOT, sub)
        files = cls._list_image_files(dir_path)
        return files[0] if files else None

    @classmethod
    def get_first_pose_image(cls, pose_type: str) -> Optional[str]:
        """从姿势图库某个子目录取第一张图"""
        sub = cls.POSE_FOLDERS.get(pose_type, pose_type)
        dir_path = os.path.join(cls.POSE_GALLERY_ROOT, sub)
        files = cls._list_image_files(dir_path)
        return files[0] if files else None

    @classmethod
    def get_pose_reference_source(cls, pose_type: str) -> Optional[str]:
        """
        获取姿势参考图来源（本地路径或线上URL）
        - 本地图库开启且找到文件：返回本地路径
        - 否则：回退到线上REF_MODEL_IMAGES
        """
        if getattr(cls, "USE_LOCAL_GALLERIES", False) and pose_type in getattr(cls, "POSE_FOLDERS", {}):
            local_path = cls.get_first_pose_image(pose_type)
            if local_path:
                return local_path
        return cls.REF_MODEL_IMAGES.get(pose_type)

# 创建配置实例
config = SimpleConfig()
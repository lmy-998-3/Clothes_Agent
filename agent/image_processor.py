"""
图片处理器 - 基于原代码，添加类型提示
"""
import base64
import uuid
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, Optional, List, Tuple, Any
import requests
from PIL import Image
from obs import ObsClient

from config import config

class SimpleImageProcessor:
    """简化图片处理器"""

    def __init__(self):
        """初始化OBS客户端"""
        self.obs_client = ObsClient(
            access_key_id=config.OBS_ACCESS_KEY_ID,
            secret_access_key=config.OBS_SECRET_ACCESS_KEY,
            server=config.OBS_SERVER
        )

    def change_clothes(self, model_image_url: str, clothes_image_url: str) -> Optional[str]:
        """
        更换衣服 - 主要功能
        返回处理后的图片URL
        """
        print("🔄 开始换衣处理...")

        # 下载图片
        model_image = self._download_image(model_image_url)
        clothes_image = self._download_image(clothes_image_url)

        if not model_image or not clothes_image:
            print("❌ 图片下载失败")
            return None

        # 调用BLTCY API换衣
        prompt = config.get_clothes_swap_prompt()

        try:
            # 准备API请求
            headers = {"Authorization": config.BLTCY_API_KEY}
            files = [
                ("image", ("model.jpg", model_image, "image/jpeg")),
                ("image", ("clothes.png", clothes_image, "image/png"))
            ]
            data = {
                "prompt": prompt,
                "model": "nano-banana-2",
                "n": 1,
                "aspect_ratio": "9:16"
            }

            # 发送请求
            print("📡 调用AI换衣API...")
            response = requests.post(
                config.BLTCY_API_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=config.PROCESS_TIMEOUT
            )

            if response.status_code != 200:
                print(f"❌ API调用失败: {response.status_code}")
                return None

            # 处理结果
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                result_url = self._process_api_result(result['data'][0])
                if result_url:
                    print(f"✅ 换衣完成: {result_url}")
                    return result_url

            return None

        except Exception as e:
            print(f"❌ 换衣处理异常: {str(e)}")
            return None

    def generate_pose(self, model_image_url: str, pose_type: str = 'default') -> Optional[str]:
        """
        生成特定姿势
        返回处理后的图片URL
        """
        print(f"🔄 生成姿势: {pose_type}")

        # 如果是default姿势（不转换），直接返回原图
        if pose_type == 'default':
            print("✅ 默认姿势：不进行姿势转换，直接返回原图")
            return model_image_url

        # 下载模特图片
        model_image = self._download_image(model_image_url)
        if not model_image:
            return None

        # 获取参考模特图片（优先本地图库）
        ref_model_url = config.get_pose_reference_source(pose_type)
        if not ref_model_url:
            print(f"❌ 不支持的姿势类型或图库中无对应图片: {pose_type}")
            return None

        # 下载参考模特图片
        ref_image = self._download_image(ref_model_url)
        if not ref_image:
            return None

        # 调用API转换姿势
        prompt = config.get_pose_prompt(pose_type)

        try:
            headers = {"Authorization": config.BLTCY_API_KEY}
            files = [
                ("image", ("ref_model.jpg", ref_image, "image/jpeg")),
                ("image", ("model.jpg", model_image, "image/jpeg"))
            ]
            data = {
                "prompt": prompt,
                "model": "nano-banana-2",
                "n": 1,
                "aspect_ratio": "9:16"
            }

            print("📡 调用AI姿势转换API...")
            response = requests.post(
                config.BLTCY_API_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=config.PROCESS_TIMEOUT
            )

            if response.status_code != 200:
                print(f"❌ API调用失败: {response.status_code}")
                return None

            # 处理结果
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                result_url = self._process_api_result(result['data'][0])
                if result_url:
                    print(f"✅ 姿势生成完成: {result_url}")
                    return result_url

            return None

        except Exception as e:
            print(f"❌ 姿势生成异常: {str(e)}")
            return None

    def generate_set(self, model_image_url: str, poses: List[str] = None) -> Dict[str, str]:
        """
        生成套图
        返回姿势类型到图片URL的字典
        """
        if poses is None:
            poses = config.SUPPORTED_POSES

        results = {}

        for pose in poses:
            print(f"\n📸 处理姿势: {pose}")
            result_url = self.generate_pose(model_image_url, pose)
            if result_url:
                results[pose] = result_url
                print(f"✅ {pose}: {result_url}")
            else:
                print(f"❌ {pose} 生成失败")

        return results

    def _download_image(self, url_or_path: str) -> Optional[bytes]:
        """下载图片或读取本地文件"""
        try:
            # 检查是否是本地文件路径
            if url_or_path.startswith(('http://', 'https://')):
                # 网络图片：下载
                response = requests.get(url_or_path, timeout=30)
                if response.status_code == 200:
                    return response.content
                else:
                    print(f"❌ 图片下载失败: {url_or_path} - 状态码: {response.status_code}")
                    return None
            else:
                # 本地文件：直接读取
                try:
                    with open(url_or_path, 'rb') as f:
                        return f.read()
                except Exception as e:
                    print(f"❌ 读取本地图片失败: {url_or_path} - {str(e)}")
                    return None
        except Exception as e:
            print(f"❌ 获取图片异常: {str(e)}")
            return None

    def _process_api_result(self, result_item: Dict[str, Any]) -> Optional[str]:
        """处理API返回结果"""
        try:
            result_url = None

            if 'b64_json' in result_item and result_item['b64_json']:
                # 处理base64图片
                img = Image.open(BytesIO(base64.b64decode(result_item['b64_json'])))
                result_url = self._upload_to_obs(img)
            elif 'url' in result_item:
                # 下载图片并上传
                img_response = requests.get(result_item['url'], timeout=30)
                if img_response.status_code == 200:
                    img = Image.open(BytesIO(img_response.content))
                    result_url = self._upload_to_obs(img)

            return result_url

        except Exception as e:
            print(f"❌ 处理API结果异常: {str(e)}")
            return None

    def _upload_to_obs(self, image: Image.Image) -> Optional[str]:
        """上传图片到OBS"""
        try:
            # 生成文件名
            dt = datetime.now().strftime('%Y%m%d')
            filename = f"fashion_agent_{uuid.uuid4().hex[:8]}.png"
            object_key = f"yishou_tqsx_img/agent_output/{dt}/{filename}"

            # 保存图片到内存
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # 上传到OBS
            resp = self.obs_client.putContent(config.OBS_BUCKET, object_key, buffer)
            if resp.status < 300:
                return f"https://{config.OBS_PUBLIC_HOST}/{object_key}"
            else:
                print(f"❌ OBS上传失败: {resp.status}")
                return None

        except Exception as e:
            print(f"❌ 上传到OBS异常: {str(e)}")
            return None

# 创建全局处理器实例
processor = SimpleImageProcessor()
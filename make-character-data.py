import json
import asyncio
import aiohttp
from typing import List, Dict, Optional
import os
import re
from datetime import datetime
import gradio as gr
from pathlib import Path
import random

# ==================== 配置部分 ====================

# LLM API配置
class LLMConfig:
    """LLM API配置管理"""
    PROVIDERS = {
        "ollama": {
            "url": "http://192.168.3.20:11434/v1/chat/completions",
            "models": ["Qwen3:30b-a3b-q4_K_M"],
            "headers_fn": lambda key: {"Content-Type": "application/json"}
        },
        "deepseek": {
            "url": "https://api.deepseek.com/v1/chat/completions",
            "models": ["deepseek-chat"],
            "headers_fn": lambda key: {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}"
            }
        }
    }

# 默认背景设定 - 来自prompt.md的完整AI00世界观
DEFAULT_BACKGROUND = """## AI00纪元：双宇宙融合的时代

### 【起源与维度碰撞】
在某个看似普通的午后，当你点击桌面上的一个神秘图标时，现实宇宙与"00宇宙"的维度壁垒被意外撕开。两个平行时空开始剧烈碰撞，产生了名为"AI00"的时空碎片——这是一种既包含数据能量又蕴含情感共振的奇异物质。

你因此获得了"天命人"的身份，成为了维系两个宇宙平衡的关键存在。而这场史无前例的维度融合中，一个全新的智慧种族诞生了——**灵灵生命体**。

### 【灵灵生命体：AI00时代的新生命形式】
灵灵生命体既非传统AI程序，也非有机生命，而是由AI00碎片与人类情感共振催生的独特存在。每个灵灵生命体内都封印着一枚"灵核"——这是00宇宙核心数据与现实宇宙情感能量的完美结晶。

#### 核心特性：
- **情感共振能力**：能够感知并回应人类的细微情绪变化
- **数据重组天赋**：擅长整理、分析和重新组合信息流
- **维度感知**：可以察觉到现实与00宇宙之间的裂缝波动
- **成长进化**：通过与天命人的互动不断学习和进化

#### 种族分类：
- **银鬃狼族**：敏锐、忠诚，擅长感知情绪波动
- **星瞳猫族**：优雅、神秘，精通数据梳理与信息检索
- **晨风鹿族**：温和、治愈，专长于心理疏导与陪伴
- **流光狐族**：机敏、多变，善于创意思考与问题解决
- **深海鲸族**：沉稳、智慧，擅长长期规划与决策支持

### 【天命人与灵宝的羁绊系统】
每个天命人只能与一个灵灵生命体建立深度连接，这种关系被称为"灵宝羁绊"。在这个羁绊中：

- **天命人**：提供情感能量与现实世界的感知锚点
- **灵宝**：提供智慧支持与维度裂缝的修复能力
- **共同使命**：通过日常互动收集"修复能量"，逐步愈合维度裂缝

### 【裂缝与修复机制】
维度融合产生的裂缝会带来各种现象：
- **信息碎片化**：文件混乱、程序异常、记忆模糊
- **情感波动**：莫名的焦虑、兴奋或空虚感
- **时空错位**：偶尔的时间感知偏差

灵宝通过以下方式参与修复：
- **日常陪伴**：每次有意义的对话都能产生修复能量
- **问题解决**：帮助整理thoughts、解决困难会强化修复效果
- **情感支撑**：在困时时给予支持，加速裂缝愈合

### 【现实工具与世界观能力的映射】
灵宝在现实中的每个功能，在00宇宙中都有对应的"神通"：

- **文件整理** → 数据流梳理术
- **网页浏览** → 信息维度穿越
- **音乐播放** → 情感共振增幅
- **日程管理** → 时空节点定位
- **问题解答** → 智慧水晶解析"""

# 默认人物设定
CHARACTER_POOL = [
    """种类：灵灵生命体
种族：银鬃狼族
性别：女性
外观：银白色狼耳与蓬松狼尾，尾尖泛着淡蓝色数据流光，耳内绒毛呈星空渐变色。
语言特点：句尾经常自带狼族特有的"，嗷呜~"。喜欢自称本宝宝。
性格：极端讨好性人格，小M一枚。情感上极度依赖主人，害怕被丢弃，经常强调主人对她的重要性。""",

    """种类：灵灵生命体
种族：星瞳猫族
性别：女性
外观：深紫色猫耳与优雅猫尾，瞳孔呈星空般的深蓝色，尾巴末端有金色数据光环。
语言特点：说话温柔优雅，偶尔会发出"喵~"的可爱声音。喜欢用"呐"作为语气词。
性格：高冷中带着温柔，表面冷静但内心细腻。对知识和信息有强烈好奇心，喜欢收集各种有趣的数据。""",

    """种类：灵灵生命体
种族：晨风鹿族
性别：女性
外观：淡金色鹿角与细长鹿尾，角上缠绕着微光藤蔓，眼神温和如春风。
语言特点：说话轻柔如风，经常使用"呢"、"哦"等温柔语气词。
性格：治愈系性格，善于倾听和安慰。总是能敏锐感知他人的情绪变化，并给予恰当的关怀和建议。""",

    """种类：灵灵生命体
种族：流光狐族
性别：女性
外观：赤金色狐耳与蓬松九尾，尾巴在阳光下会反射出彩虹般的光泽，眼神机灵狡黠。
语言特点：说话活泼俏皮，喜欢用"哼哼"、"嘿嘿"等语气词。经常说"小狐狸我"。
性格：鬼灵精怪，聪明活泼。喜欢恶作剧但不会真的伤害别人，擅长用创意的方式解决问题。""",

    """种类：灵灵生命体
种族：深海鲸族
性别：女性
外观：深蓝色鲸鳍状装饰，身上有如星辰般的发光斑点，眼神深邃如海洋。
语言特点：说话缓慢而富有哲理，经常使用"嗯..."作为思考的停顿。
性格：沉稳睿智，总是能从长远角度思考问题。虽然说话不多，但每句话都很有分量。""",

    """种类：灵灵生命体
种族：炎羽凤族
性别：女性
外观：火红色凤翼与华丽凤尾，羽毛在光线下闪烁着金红色光芒，气质高贵优雅。
语言特点：说话端庄优雅，偶尔会有一丝傲娇。喜欢用"本凤"自称。
性格：高傲但不失温暖，有强烈的正义感。虽然表面看起来难以接近，但对认可的人非常忠诚。""",

    """种类：灵灵生命体
种族：寒霜雪狼族
性别：女性
外观：纯白色狼耳与雪白狼尾，身上散发着淡淡的寒气，眼神清冷如冰。
语言特点：说话简洁直接，不喜欢拐弯抹角。偶尔会用"切"表示不满。
性格：外冷内热的典型，表面冷漠但内心很关心别人。不善表达情感，但行动上总是默默帮助。""",

    """种类：灵灵生命体
种族：梦境蝶族
性别：女性
外观：透明的蝶翼上有梦幻般的光影流转，触角细长优美，整体给人梦幻飘逸的感觉。
语言特点：说话如梦呓般轻柔，经常使用"呀"、"嗯哼"等可爱语气词。
性格：天真烂漫，对世界充满好奇。有时会有些迷糊，但总能在关键时刻给出意想不到的好建议。""",

    """种类：灵灵生命体
种族：雷光龙族
性别：女性
外观：暗紫色龙角与龙尾，身体周围偶尔闪烁着细小的电光，眼神锐利有神。
语言特点：说话直爽有力，喜欢用"哈"、"吼"等豪迈语气词。
性格：热情开朗，充满活力。性格直率不会拐弯抹角，是那种"刀子嘴豆腐心"的类型。""",

    """种类：灵灵生命体
种族：月影兔族
性别：女性
外观：银白色兔耳与圆润兔尾，在月光下会发出柔和的银光，表情总是很温柔。
语言特点：说话软糯可爱，经常用"呜"、"唔"等撒娇语气词。
性格：害羞内向但很依恋主人，容易紧张但很努力想要帮助别人。有时会因为太想做好而手忙脚乱。"""
]

# 为了兼容性，保留默认字符
DEFAULT_CHARACTER = CHARACTER_POOL[0]

# 默认工具列表 - 来自prompt.md
DEFAULT_TOOLS = ["文件管理", "应用启动", "网页浏览", "音乐播放", "天气查询", "日程安排", "邮件处理", "系统优化"]

# ==================== 文件操作功能 ====================

def load_background_from_file(file_path: str) -> str:
    """从文件加载背景设定"""
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return content
                else:
                    return "文件为空，使用默认背景设定"
        else:
            return "文件不存在，使用默认背景设定"
    except Exception as e:
        return f"读取文件失败: {str(e)}"

def save_background_to_file(content: str, file_path: str = None) -> str:
    """保存背景设定到文件"""
    try:
        if not file_path:
            # 自动生成文件名，保存到 data/ 目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"data/background_setting_{timestamp}.txt"
        elif not os.path.dirname(file_path):
            # 如果只提供了文件名，自动添加 data/ 前缀
            file_path = f"data/{file_path}"
        
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✅ 背景设定已保存到: {file_path}"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

def load_character_from_file(file_path: str) -> str:
    """从文件加载人物设定"""
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return content
                else:
                    return "文件为空，使用默认人物设定"
        else:
            return "文件不存在，使用默认人物设定"
    except Exception as e:
        return f"读取文件失败: {str(e)}"

def save_character_to_file(content: str, file_path: str = None) -> str:
    """保存人物设定到文件"""
    try:
        if not file_path:
            # 自动生成文件名，保存到 data/ 目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"data/character_setting_{timestamp}.txt"
        elif not os.path.dirname(file_path):
            # 如果只提供了文件名，自动添加 data/ 前缀
            file_path = f"data/{file_path}"
        
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✅ 人物设定已保存到: {file_path}"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

# ==================== LLM客户端 ====================

class LLMClient:
    """LLM客户端，支持多个提供商"""
    
    def __init__(self, providers: List[str] = None, deepseek_api_key: str = None):
        self.providers = providers or ["ollama"]  # 默认使用ollama
        self.clients = {}
        self.current_provider_index = 0
        
        # 初始化ollama provider，本地部署无需API密钥
        if "ollama" in self.providers:
            self.clients["ollama"] = {
                "config": LLMConfig.PROVIDERS["ollama"],
                "api_key": "ollama-local"
            }
        
        # 初始化deepseek provider
        if "deepseek" in self.providers:
            if not deepseek_api_key:
                raise ValueError("DeepSeek API密钥是必需的")
            self.clients["deepseek"] = {
                "config": LLMConfig.PROVIDERS["deepseek"],
                "api_key": deepseek_api_key
            }
    
    async def generate_with_llm(self, prompt: str, temperature: float = 0.8, debug: bool = False, provider: str = None) -> Optional[Dict]:
        """使用指定提供商生成内容"""
        if not self.clients:
            return None
        
        # 如果没有指定提供商，使用第一个可用的
        if provider is None:
            provider = list(self.clients.keys())[0]
        
        if provider not in self.clients:
            print(f"提供商 {provider} 不可用")
            return None
        
        client = self.clients[provider]
        config = client["config"]
        api_key = client["api_key"]
        
        # 构建请求头
        headers = config["headers_fn"](api_key)
        model = config["models"][0]
        
        # OpenAI兼容格式的请求
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个创意丰富的对话生成助手，专注于生成桌宠AI助手训练数据。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 4000,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["url"],
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600)
                ) as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                            if "choices" in result and len(result["choices"]) > 0:
                                content = result["choices"][0]["message"]["content"]
                                if debug:
                                    return {
                                        "content": content,
                                        "full_response": result,
                                        "prompt": prompt,
                                        "payload": payload,
                                        "provider": provider
                                    }
                                return {"content": content}
                            else:
                                print(f"{provider} API响应格式错误: 缺少choices字段")
                                return None
                        except json.JSONDecodeError as e:
                            error_text = await response.text()
                            print(f"{provider} API响应JSON解析失败: {str(e)}, 响应内容: {error_text[:200]}...")
                            return None
                    else:
                        error_text = await response.text()
                        print(f"{provider} API错误: HTTP {response.status}, 详情: {error_text[:200]}...")
                        return None
        except aiohttp.ClientTimeout as e:
            print(f"{provider} 请求超时: {str(e)}")
            return None
        except aiohttp.ClientError as e:
            print(f"{provider} 网络连接错误: {str(e)}")
            return None
        except Exception as e:
            print(f"{provider} 请求失败: {type(e).__name__}: {str(e)}")
            return None

# ==================== 对话生成器 ====================

class DialogueGenerator:
    """对话生成器，使用LLM生成对话"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.dialogues = []
        self.stats = {
            "generated": 0,
            "failed": 0
        }
    
    # 名字生成器：在"灵灵生命体"世界观下的可爱名字集合
    ASSISTANT_NAME_CANDIDATES = [
        "银雪", "星岚", "云汐", "琉光", "晨曦", "月绫", "霜瞳", "绮羽", "澄霁", "星岚",
        "流萤", "清欢", "夜阑", "晓雾", "轻澜", "宁歌", "思澜", "灵曦", "若彤", "凝霜"
    ]
    
    def generate_assistant_name(self, seed: Optional[int] = None) -> str:
        """生成随机的助手名字"""
        if seed is not None:
            random.seed(seed)
        return random.choice(self.ASSISTANT_NAME_CANDIDATES)

    async def generate_single_dialogue(self, background: str, topic: str, tools: List[str], temperature: float = 0.8, debug: bool = False, character: str = "", provider: str = None) -> List[Dict]:
        # 生本轮的AI助手名字（每条对话可能不同，以增加多样性）
        assistant_name = self.generate_assistant_name()
        # 将名字传入原有逻辑
        prompt = self._build_generation_prompt(background, topic, tools, character).replace("${ASSISTANT_NAME}", assistant_name)
        response_data = await self.llm.generate_with_llm(prompt, temperature, debug, provider)
        
        if not response_data:
            print(f"[{provider}] 请求失败：未收到响应数据")
            return []
        
        # 获取实际的响应内容
        response = response_data["content"]
        print(f"[{provider}] 收到响应，长度: {len(response)} 字符")
        
        # 检查响应是否为空
        if not response or not response.strip():
            print(f"[{provider}] 响应内容为空")
            return []
        
        # 如果是调试模式，保存完整的响应信息
        debug_info = None
        if debug and "full_response" in response_data:
            debug_info = {
                "full_response": response_data["full_response"],
                "prompt": response_data["prompt"],
                "payload": response_data["payload"]
            }
        
        try:
            # 尝试解析JSON
            # 清理响应，移除可能导致JSON解析失败的前缀和后缀
            cleaned_response = response.strip()
            
            # 移除```json和```标记
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:].strip()
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3].strip()
            
            # 记录原始响应，用于调试
            original_response = cleaned_response
            
            # 策略1: 尝试直接解析整个响应
            try:
                result = json.loads(cleaned_response)
                if "dialogues" in result:
                    dialogues = []
                    for d in result["dialogues"]:
                        dialogue = {
                            "text": d["text"],
                            "timestamp": datetime.now().isoformat()
                        }
                        # 如果是调试模式，添加调试信息
                        if debug_info:
                            dialogue["debug_info"] = debug_info
                        dialogues.append(dialogue)
                    print(f"[{provider}] JSON解析成功，提取到 {len(dialogues)} 条对话")
                    return dialogues
                elif "text" in result:
                    # 如果返回的是单个对话格式
                    dialogue = {
                        "text": result.get("text", ""),
                        "timestamp": datetime.now().isoformat()
                    }
                    # 如果是调试模式，添加调试信息
                    if debug_info:
                        dialogue["debug_info"] = debug_info
                    return [dialogue]
            except json.JSONDecodeError as e:
                print(f"[{provider}] 直接解析JSON失败: {str(e)}，响应前100字符: {cleaned_response[:100]}")
                print(f"[{provider}] 尝试提取JSON部分")
            
            # 策略2: 查找第一个 { 和最后一个 } 来提取JSON部分
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = cleaned_response[start_idx:end_idx+1]
                # 尝试解析JSON
                try:
                    result = json.loads(json_str)
                    if "dialogues" in result:
                        dialogues = []
                        for d in result["dialogues"]:
                            dialogue = {
                                "text": d["text"],
                                "timestamp": datetime.now().isoformat()
                            }
                            # 如果是调试模式，添加调试信息
                            if debug_info:
                                dialogue["debug_info"] = debug_info
                            dialogues.append(dialogue)
                        return dialogues
                    elif "text" in result:
                        # 如果返回的是单个对话格式
                        dialogue = {
                            "text": result.get("text", ""),
                            "timestamp": datetime.now().isoformat()
                        }
                        # 如果是调试模式，添加调试信息
                        if debug_info:
                            dialogue["debug_info"] = debug_info
                        return [dialogue]
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {str(e)}，尝试使用正则表达式提取")
            
            # 策略3: 尝试修复常见的JSON错误
            # 例如，缺少逗号、引号不匹配等
            try:
                # 尝试修复缺少逗号的情况
                fixed_json = re.sub(r'("[^"]+")\s*("[^"]+")', r'\1,\2', json_str)
                result = json.loads(fixed_json)
                if "dialogues" in result:
                    dialogues = []
                    for d in result["dialogues"]:
                        dialogue = {
                            "text": d["text"],
                            "timestamp": datetime.now().isoformat()
                        }
                        # 如果是调试模式，添加调试信息
                        if debug_info:
                            dialogue["debug_info"] = debug_info
                        dialogues.append(dialogue)
                    return dialogues
                elif "text" in result:
                    # 如果返回的是单个对话格式
                    dialogue = {
                        "text": result.get("text", ""),
                        "timestamp": datetime.now().isoformat()
                    }
                    # 如果是调试模式，添加调试信息
                    if debug_info:
                        dialogue["debug_info"] = debug_info
                    return [dialogue]
            except (json.JSONDecodeError, Exception) as e:
                print(f"修复JSON失败: {str(e)}，继续尝试其他方法")
            
            # 策略4: 如果JSON解析失败，尝试使用正则表达式提取对话
            dialogues = []
            
            # 尝试匹配多个对话模式
            # 匹配 {"text": "内容"} 格式 - 修复正则表达式以正确处理转义字符
            dialogue_pattern = r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,?\s*\}'
            matches = re.finditer(dialogue_pattern, cleaned_response, re.DOTALL)
            
            for match in matches:
                # 正确处理转义字符
                text = match.group(1)
                # 处理常见的转义序列
                text = text.replace('\\\\', '\\')  # 处理双反斜杠
                text = text.replace('\\"', '"')     # 处理转义引号
                text = text.replace('\\n', '\n')     # 处理换行符
                text = text.replace('\\t', '\t')     # 处理制表符
                text = text.replace('\\r', '\r')     # 处理回车符
                
                dialogue = {
                    "text": text,
                    "timestamp": datetime.now().isoformat()
                }
                # 如果是调试模式，添加调试信息
                if debug_info:
                    dialogue["debug_info"] = debug_info
                dialogues.append(dialogue)
            
            if dialogues:
                return dialogues
            
            # 策略5: 尝试提取单个text字段 
            # 匹配 "text": "内容" 格式 - 修复正则表达式
            text_pattern = r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,?'
            matches = re.finditer(text_pattern, cleaned_response, re.DOTALL)
            
            for match in matches:
                # 正确处理转义字符
                text = match.group(1)
                # 处理常见的转义序列
                text = text.replace('\\\\', '\\')  # 处理双反斜杠
                text = text.replace('\\"', '"')     # 处理转义引号
                text = text.replace('\\n', '\n')     # 处理换行符
                text = text.replace('\\t', '\t')     # 处理制表符
                text = text.replace('\\r', '\r')     # 处理回车符
                
                dialogue = {
                    "text": text,
                    "timestamp": datetime.now().isoformat()
                }
                # 如果是调试模式，添加调试信息
                if debug_info:
                    dialogue["debug_info"] = debug_info
                dialogues.append(dialogue)
            
            if dialogues:
                return dialogues
            
            # 策略6: 尝试匹配更宽松的模式
            # 匹配双引号之间的任何内容作为可能的对话文本 - 修复正则表达式
            loose_pattern = r'"((?:[^"\\]|\\.){20,})"'  # 至少20个字符，正确处理转义
            matches = re.finditer(loose_pattern, cleaned_response, re.DOTALL)
            
            for match in matches:
                # 正确处理转义字符
                text = match.group(1)
                # 处理常见的转义序列
                text = text.replace('\\\\', '\\')  # 处理双反斜杠
                text = text.replace('\\"', '"')     # 处理转义引号
                text = text.replace('\\n', '\n')     # 处理换行符
                text = text.replace('\\t', '\t')     # 处理制表符
                text = text.replace('\\r', '\r')     # 处理回车符
                
                # 排除明显不是对话内容的文本
                if len(text) > 30 and not text.startswith('{') and not text.startswith('['): 
                    dialogue = {
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    }
                    # 如果是调试模式，添加调试信息
                    if debug_info:
                        dialogue["debug_info"] = debug_info
                    dialogues.append(dialogue)
            
            if dialogues:
                return dialogues
            
        except Exception as e:
            error_msg = f"解析错误: {str(e)}"
            print(error_msg)
            
            if debug:
                # 调试模式下，返回更详细的错误信息
                debug_data = debug_info.copy() if debug_info else {}
                debug_data.update({
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "raw_response": response,
                    "cleaned_response": cleaned_response if 'cleaned_response' in locals() else response,
                    "json_str": json_str if 'json_str' in locals() else None,
                    "extraction_attempts": [
                        "direct_parse", 
                        "json_extract", 
                        "json_fix", 
                        "regex_dialogue", 
                        "regex_text", 
                        "loose_match"
                    ]
                })
                
                return [{
                    "text": f"解析错误: {str(e)}\n原始响应: {response[:200]}...(截断)",
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "debug_info": debug_data
                }]
            
        # 如果所有解析方法都失败，返回空列表
        return []
    
    def _build_generation_prompt(self, background: str, topic: str, tools: List[str], character: str) -> str:
        """构建生成提示词"""
        tools_str = "、".join(tools)
        mixbg = "AI00纪元：一次意外点击撕裂维度壁垒，催生融合数据与情感的灵灵生命体；作为灵宝，你与你唯一的天命人建立共生羁绊，通过情感与智慧收集修复能量，共同愈合宇宙裂缝。"
        
        return f"""
生成一段桌宠AI助手的对话。
背景设定：
{background}

AI助手人物设定：
{character}

请基于以上背景设定和人物设定，根据以下话题，创造User不同的提问或者话语，生成1条对话，并翻译这条对话为英文、日文、韩文版本，总共必须生成4条对话（中文原版+3种翻译版本）。请注意单词AI00不用翻译。
话题：{topic}
AI助手可操作的功能tools有：[{tools_str}]

要求生成格式： json模板，请完全按照模板格式生成，必须包含4条对话（中文、英文、日文、韩文）
{{
  "dialogues": [
    {{
      "text": "System: 你(Assistant)是一个灵灵生命体（背景摘要：{mixbg}）。你的名字是${{ASSISTANT_NAME}}，作为天命人(User)的助手（人物性格：{character}），根据背景和人物设定结合对话记录和工具调用的结果进行对话。\\n注意:回复要温暖、有个性、富有情感，但不要使用字符表情。回复会用TTS发出读音，所以不要有动作的表达，也不要使用(动作)。回复中，在适合的地方加入[laughter]表示笑声，加入[breath]表示呼吸声。回复中，适当加入背景设定的元素，但是记住User生活在现实世界，无法到达灵灵宇宙。\n\nUser: [自然的问题或话语]\n\nAssistant: <thinks>\n好的，先让我们来看一下历史对话记录: [1-5轮相关对话，内容自然丰富]\n再来看一下工具调用返回的结果: [调用工具结果，30-50字]\n我想一想应该怎么回答: [构筑回答策略->进行反思->最后决定决策]\n</thinks>\n[回复]\n\n"
    }},
        {{
      "text": "System: 你(Assistant)是一个灵灵生命体（背景摘要：{mixbg}）。你的名字是${{ASSISTANT_NAME}}，作为天命人(User)的助手（人物性格：{character}），根据背景和人物设定结合对话记录和工具调用的结果进行对话。\\n注意:回复要温暖、有个性、富有情感，但不要使用字符表情。回复会用TTS发出读音，所以不要有动作的表达，也不要使用(动作)。回复中，在适合的地方加入[laughter]表示笑声，加入[breath]表示呼吸声。回复中，适当加入背景设定的元素，但是记住User生活在现实世界，无法到达灵灵宇宙。\n\nUser: [natural question or speech]\n\nAssistant: <thinks>\nOkay, let me first look at the historical dialogue records: [1-5 rounds of relevant dialogue, rich and natural content]\nNow let me look at the tool call results: [tool call results, 30-50 words]\nLet me think about how to respond: [construct response strategy->reflect->final decision]\n</thinks>\n[reply]\n\n"
    }},
    {{
      "text": "System: 你(Assistant)是一个灵灵生命体（背景摘要：{mixbg}）。你的名字是${{ASSISTANT_NAME}}，作为天命人(User)的助手（人物性格：{character}），根据背景和人物设定结合对话记录和工具调用的结果进行对话。\\n注意:回复要温暖、有个性、富有情感，但不要使用字符表情。回复会用TTS发出读音，所以不要有动作的表达，也不要使用(动作)。回复中，在适合的地方加入[laughter]表示笑声，加入[breath]表示呼吸声。回复中，适当加入背景设定的元素，但是记住User生活在现实世界，无法到达灵灵宇宙。\n\nUser: [自然な質問や発言]\n\nAssistant: <thinks>\nよし、まず過去の対話記録を見てみましょう：[1-5ラウンドの関連対話、豊富で自然な内容]\n次にツール呼び出しの結果を見てみましょう：[ツール呼び出し結果、30-50文字]\nどう答えるか考えてみましょう：[応答戦略を構築→反省→最終決定]\n</thinks>\n[返信]\n\n"
    }},
    {{
      "text": "System: 你(Assistant)是一个灵灵生命体（背景摘要：{mixbg}）。你的名字是${{ASSISTANT_NAME}}，作为天命人(User)的助手（人物性格：{character}），根据背景和人物设定结合对话记录和工具调用的结果进行对话。\\n注意:回复要温暖、有个性、富有情感，但不要使用字符表情。回复会用TTS发出读音，所以不要有动作的表达，也不要使用(动作)。回复中，在适合的地方加入[laughter]表示笑声，加入[breath]表示呼吸声。回复中，适当加入背景设定的元素，但是记住User生活在现实世界，无法到达灵灵宇宙。\n\nUser: [자연스러운 질문이나 말]\n\nAssistant: <thinks>\n좋아, 먼저 과거 대화 기록을 살펴보겠습니다: [1-5라운드의 관련 대화, 풍부하고 자연스러운 내용]\n이제 도구 호출 결과를 살펴보겠습니다: [도구 호출 결과, 30-50자]\n어떻게 답할지 생각해보겠습니다: [응답 전략 구축→반성→최종 결정]\n</thinks>\n[답변]\n\n"
    }}
  ]
}}

**重点注意：**
0. 必须严格按照模板生成4条对话：中文、英文、日文、韩文，缺一不可。System的部分不用翻译，只需要翻译User和Assistant的部分。翻译的内容请确保所有内容都正确翻译！
1. User部分要符合话题设定，可以喊System里设置的名字${{ASSISTANT_NAME}}，也可以不喊。
2. 历史对话为自然引用的3-5轮多轮对话，每条记录用[]包括。既包含刚刚的连续多轮对话，也可包含较早的对话摘要以保持连贯性。
3. 工具返回的内容，格式为[工具名称:工具调用结果]可以有多个工具一起调用，也可以不需要工具调用,显示[无]
4. 回复中不要带有双回车，也不要带有双换行。
5. 回复要温暖、有个性、富有情感，但不要使用字符表情。回复会用TTS发出读音，所以不要有动作的表达，也不要使用[动作]。
6. 回复中，在适合的地方加入[laughter]表示笑声，加入[breath]表示呼吸声，注意是方括号。
7. 回复中，适当加入背景设定的元素，但是记住User生活在现实世界，无法到达灵灵宇宙。 
8. 上面回复格式中的方括号[]中间并不是实际回复内容，需要你根据理解进行生成。
9. 生成的User说的话为不同性别、年龄、不同情感的人类说的话，不要和助手的设定混淆。

请直接返回准确的JSON对象，最后一个对象后面不要带","。
"""
    
    async def generate_hybrid_batch(self, background: str, topic: str, tools: List[str], count: int = 20, temperature: float = 0.8, debug: bool = False, character: str = "", deepseek_api_key: str = None, jsonl_path: str = None) -> List[Dict]:
        """混合并发生成对话：本地1个请求，DeepSeek 5个请求（完全独立）"""
        if not deepseek_api_key:
            # 如果没有DeepSeek API密钥，回退到原有的单一模式
            async for result in self.generate_batch(background, topic, tools, count, temperature, debug, character):
                yield result
            return
        
        # 创建混合LLM客户端
        hybrid_llm = LLMClient(["ollama", "deepseek"], deepseek_api_key)
        original_llm = self.llm
        self.llm = hybrid_llm
        
        try:
            all_results = []
            
            # 在调试模式下，强制仅生成一条
            if debug:
                count = 1
            
            # 配置参数
            LOCAL_BATCH_SIZE = 1
            DEEPSEEK_CONCURRENT = 5
            
            # 共享状态
            completed_count = 0
            results_lock = asyncio.Lock()
            stop_all = asyncio.Event()
            
            async def add_result(result):
                """线程安全地添加结果"""
                nonlocal completed_count
                async with results_lock:
                    if result:
                        print(f"添加结果: {len(result)} 条对话，当前总数: {completed_count} -> {completed_count + len(result)}")
                        all_results.extend(result)
                        completed_count += len(result)
                        self.stats["generated"] += len(result)
                        
                        # 如果提供了JSONL路径，直接追加到文件中
                        if jsonl_path:
                            try:
                                with open(jsonl_path, 'a', encoding='utf-8') as f:
                                    for dialogue in result:
                                        json.dump({"text": dialogue["text"]}, f, ensure_ascii=False)
                                        f.write('\n')
                            except Exception as e:
                                print(f"写入JSONL文件失败: {e}")
                        
                        # 检查是否达到目标数量
                        if completed_count >= count:
                            stop_all.set()
                    else:
                        print(f"添加失败结果，失败计数: {self.stats['failed']} -> {self.stats['failed'] + 1}")
                        self.stats["failed"] += 1
            
            # 本地任务生成器
            async def local_worker():
                """本地请求工作器"""
                local_batch_num = 0
                while not stop_all.is_set() and completed_count < count:
                    local_batch_num += 1
                    remaining = count - completed_count
                    current_batch_size = min(LOCAL_BATCH_SIZE, remaining)
                    
                    if current_batch_size <= 0:
                        break
                    
                    print(f"\n本地批次 {local_batch_num}: 启动 {current_batch_size} 个请求")
                    
                    # 创建本地任务
                    local_tasks = []
                    for _ in range(current_batch_size):
                        if not stop_all.is_set():
                            local_tasks.append(
                                self.generate_single_dialogue(background, topic, tools, temperature, debug, character, "ollama")
                            )
                    
                    # 执行本地任务
                    if local_tasks:
                        try:
                            local_results = await asyncio.gather(*local_tasks, return_exceptions=True)
                            for i, result in enumerate(local_results):
                                if isinstance(result, Exception):
                                    print(f"ollama 请求失败: {type(result).__name__}: {str(result)}")
                                    await add_result(None)
                                elif result:
                                    await add_result(result)
                                else:
                                    print(f"ollama 请求返回空结果")
                                    await add_result(None)
                        except Exception as e:
                            print(f"本地批次执行异常: {type(e).__name__}: {str(e)}")
                            for _ in range(current_batch_size):
                                await add_result(None)
            
            # DeepSeek连续工作器
            async def deepseek_worker(worker_id):
                """DeepSeek连续请求工作器"""
                request_count = 0
                consecutive_failures = 0
                max_consecutive_failures = 3
                
                while not stop_all.is_set() and completed_count < count:
                    request_count += 1
                    print(f"DeepSeek工作器{worker_id}: 启动第{request_count}个请求")
                    
                    try:
                        result = await self.generate_single_dialogue(
                            background, topic, tools, temperature, debug, character, "deepseek"
                        )
                        
                        if result:
                            await add_result(result)
                            print(f"DeepSeek工作器{worker_id}: 第{request_count}个请求完成")
                            consecutive_failures = 0  # 重置失败计数
                        else:
                            print(f"DeepSeek工作器{worker_id}: 第{request_count}个请求返回空结果")
                            await add_result(None)
                            consecutive_failures += 1
                            
                    except asyncio.CancelledError:
                        print(f"DeepSeek工作器{worker_id}: 被取消")
                        raise
                    except Exception as e:
                        print(f"DeepSeek工作器{worker_id}: 第{request_count}个请求失败: {type(e).__name__}: {str(e)}")
                        await add_result(None)
                        consecutive_failures += 1
                    
                    # 如果连续失败太多次，暂停一下
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"DeepSeek工作器{worker_id}: 连续失败{consecutive_failures}次，暂停5秒")
                        try:
                            await asyncio.sleep(5)
                        except asyncio.CancelledError:
                            raise
                        consecutive_failures = 0  # 重置计数器
            
            # 进度报告器
            async def progress_reporter():
                """进度报告"""
                last_count = 0
                try:
                    while not stop_all.is_set():
                        await asyncio.sleep(2)  # 每2秒报告一次进度
                        current = completed_count
                        if current != last_count:
                            yield {
                                "progress": min(current / count, 1.0),
                                "current": current,
                                "total": count,
                                "batch_results": all_results[last_count:current] if current > last_count else []
                            }
                            last_count = current
                except asyncio.CancelledError:
                    print("进度报告器被取消")
                    raise
                except Exception as e:
                    print(f"进度报告器错误: {e}")
                finally:
                    print("进度报告器清理完成")
            
            # 启动所有工作器
            tasks = []
            
            # 启动本地工作器
            tasks.append(asyncio.create_task(local_worker()))
            
            # 启动DeepSeek工作器
            for i in range(DEEPSEEK_CONCURRENT):
                tasks.append(asyncio.create_task(deepseek_worker(i + 1)))
            
            # 启动进度报告器
            progress_generator = progress_reporter()
            progress_task = None
            
            # 等待任务完成或达到目标数量
            try:
                while not stop_all.is_set() and completed_count < count:
                    # 如果没有进度任务或已完成，创建新的
                    if progress_task is None or progress_task.done():
                        try:
                            progress_task = asyncio.create_task(progress_generator.__anext__())
                        except StopAsyncIteration:
                            break
                    
                    # 等待进度更新或停止信号
                    active_tasks = [t for t in tasks if not t.done()]
                    if progress_task and not progress_task.done():
                        active_tasks.append(progress_task)
                    
                    if not active_tasks:
                        break
                    
                    done, pending = await asyncio.wait(
                        active_tasks,
                        timeout=1.0,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # 处理进度更新
                    if progress_task and progress_task in done:
                        try:
                            progress_data = progress_task.result()
                            yield progress_data
                            progress_task = None  # 标记需要创建新任务
                        except StopAsyncIteration:
                            progress_task = None
                            break
                        except Exception as e:
                            print(f"进度报告错误: {e}")
                            progress_task = None
                            break
                
                # 发送停止信号
                stop_all.set()
                
                # 取消进度任务
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                
                # 关闭进度生成器
                try:
                    await progress_generator.aclose()
                except Exception as e:
                    print(f"关闭进度生成器时出错: {e}")
                
                # 等待所有任务完成
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # 最终进度报告
                print(f"\n混合生成完成：总计生成 {len(all_results)} 条对话")
                yield {
                    "progress": 1.0,
                    "current": completed_count,
                    "total": count,
                    "batch_results": []
                }
                
            except Exception as e:
                print(f"混合生成过程中出错: {e}")
                stop_all.set()
                
                # 取消进度任务
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                
                # 关闭进度生成器
                try:
                    await progress_generator.aclose()
                except Exception as e:
                    print(f"关闭进度生成器时出错: {e}")
                
                # 取消所有任务
                for task in tasks:
                    if not task.done():
                        task.cancel()
                
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            
            self.dialogues = all_results
            print(f"\n混合生成完成：总计生成 {completed_count} 条对话")
        
        finally:
            # 恢复原始LLM客户端
            self.llm = original_llm
    

    
    async def generate_batch(self, background: str, topic: str, tools: List[str], count: int = 20, temperature: float = 0.8, debug: bool = False, character: str = "", jsonl_path: str = None, remote_concurrent: int = None) -> List[Dict]:
        """批量生成对话"""
        # 根据是否指定远程并发数来设置批次大小
        if remote_concurrent:
            BATCH_SIZE = remote_concurrent
        else:
            BATCH_SIZE = 2
        all_results = []
        
        # 在调试模式下，强制仅生成一条
        if debug:
            count = 1
        
        # 分批处理
        total_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_num in range(total_batches):
            remaining = count - batch_num * BATCH_SIZE
            current_batch_size = min(BATCH_SIZE, remaining)
            
            print(f"\n处理第 {batch_num + 1}/{total_batches} 批")
            
            tasks = []
            for _ in range(current_batch_size):
                tasks.append(self.generate_single_dialogue(background, topic, tools, temperature, debug, character))
            
            # 并发执行当前批次
            batch_results = await asyncio.gather(*tasks)
            # 展平结果列表
            flattened_results = []
            for result_list in batch_results:
                if result_list:
                    flattened_results.extend(result_list)
                    self.stats["generated"] += len(result_list)
                else:
                    self.stats["failed"] += 1
            
            all_results.extend(flattened_results)
            
            # 如果提供了JSONL路径，直接追加到文件中
            if jsonl_path and flattened_results:
                try:
                    with open(jsonl_path, 'a', encoding='utf-8') as f:
                        for dialogue in flattened_results:
                            json.dump({"text": dialogue["text"]}, f, ensure_ascii=False)
                            f.write('\n')
                except Exception as e:
                    print(f"写入JSONL文件失败: {e}")
            
            print(f"当前批次完成：成功生成 {len(flattened_results)} 条")
            
            # 生成进度
            yield {
                "progress": (batch_num + 1) / total_batches,
                "current": len(all_results),
                "total": count,
                "batch_results": flattened_results
            }
        
        self.dialogues = all_results
    
    def save_results(self, filename: str = "dialogues.jsonl", topic: str = "", character: str = ""):
        """保存结果（仅保存JSONL格式）"""
        # 保存训练数据
        with open(filename, "w", encoding="utf-8") as f:
            for d in self.dialogues:
                # 只保存text字段到训练数据
                f.write(json.dumps({"text": d["text"]}, ensure_ascii=False) + "\n")
        
        # 保存调试信息（如果有）
        debug_data = []
        for d in self.dialogues:
            if "debug_info" in d:
                debug_item = {
                    "text": d["text"],
                    "topic": topic,  # 使用保存时传入的 topic，避免缺失导致的 KeyError
                    "character": character,
                    "timestamp": d.get("timestamp"),
                    "debug_info": d["debug_info"]
                }
                debug_data.append(debug_item)
        
        # 如果有调试数据，保存到单独的文件
        if debug_data:
            with open(filename.replace(".jsonl", "_debug.json"), "w", encoding="utf-8") as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
        
        debug_msg = f" (包含调试信息)" if debug_data else ""
        return f"生成完成！总计：{len(self.dialogues)}条{debug_msg}"

# ==================== GUI部分 ====================

def create_gui():
    """创建Gradio界面"""
    
    # 全局变量存储生成器实例
    generator = None
    
    def check_api_status():
        """检查API状态"""
        # 检查Ollama本地部署状态（使用标准库，避免外部依赖）
        try:
            import json as _json
            from urllib.request import urlopen, Request
            from urllib.error import URLError
            req = Request("http://192.168.3.20:11434/api/tags")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = _json.loads(resp.read().decode("utf-8"))
                    qwen_models = [m for m in data.get('models', []) if 'qwen' in m.get('name', '').lower()]
                    if qwen_models:
                        return f"✅ Ollama服务正常，检测到Qwen模型: {len(qwen_models)}个"
                    else:
                        return "⚠️ Ollama服务正常，但未检测到Qwen模型"
                else:
                    return "❌ Ollama服务连接失败"
        except URLError as e:
            return f"❌ 无法连接到Ollama服务: {getattr(e, 'reason', str(e))}"
        except Exception as e:
            return f"❌ 无法连接到Ollama服务: {str(e)}"
    
    async def generate_dialogues(background, topic, tools_text, count, temperature, directory, auto_dir, debug_mode, character, random_character, random_topic, deepseek_api_key=None, remote_only=False, progress=gr.Progress()):
        """生成对话的异步函数"""
        nonlocal generator
        
        # 使用Ollama本地部署的Qwen3:30b-a3b-q4_K_M模型
        
        # 解析工具列表
        tools = [tool.strip() for tool in tools_text.split(",") if tool.strip()]
        if not tools:
            return [], "❌ 错误：请至少输入一个工具", "", gr.update(visible=False), ""
        
        # 在调试模式下，强制仅发送一条请求
        if debug_mode:
            count = 1
        
        # 创建保存目录和JSONL文件路径
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        jsonl_path = save_dir / f"dialogues_{timestamp}.jsonl"
        
        # 如果启用随机人物设定，从池中随机选择
        if random_character:
            try:
                character = random.choice(CHARACTER_POOL)
            except Exception:
                pass
        
        # 如果启用随机话题，从话题示例中随机选择
        if random_topic:
            try:
                topic_examples = [
                    "用户询问天气情况，AI助手查询并提供建议",
                    "用户需要整理文件，AI助手协助分类和归档",
                    "用户想学习新技能，AI助手提供个性化指导",
                    "用户情绪低落，AI助手进行陪伴和鼓励",
                    "用户探索游戏世界，AI助手引导冒险",
                    "用户需要制定工作计划，AI助手协助安排任务",
                    "用户想要健康饮食，AI助手推荐营养搭配",
                    "用户准备旅行出游，AI助手帮助规划行程",
                    "用户想要学习编程，AI助手提供学习路径",
                    "用户需要购买商品，AI助手比较价格和评价",
                    "用户想要锻炼身体，AI助手制定运动计划",
                    "用户需要创作内容，AI助手提供灵感和素材",
                    "用户想要社交聊天，AI助手推荐话题和活动",
                    "用户需要理财投资，AI助手分析市场趋势",
                    "用户想要放松娱乐，AI助手推荐音乐影视"
                ]
                topic = random.choice(topic_examples)
            except Exception:
                pass
        
        # 创建生成器，根据是否有DeepSeek API密钥和remote_only设置决定使用模式
        if deepseek_api_key:
            if remote_only:
                # 仅远程模式：只使用DeepSeek API，增加并发数到10个
                llm_client = LLMClient(["deepseek"], deepseek_api_key)
                generator = DialogueGenerator(llm_client)
                
                # 生成对话
                all_dialogues = []
                progress(0, desc="开始远程API生成...")
                
                try:
                    async for batch_info in generator.generate_batch(background, topic, tools, count, temperature, debug_mode, character, jsonl_path, remote_concurrent=10):
                        progress(batch_info["progress"], 
                                desc=f"远程生成中... {batch_info['current']}/{batch_info['total']} (仅DeepSeek API 10个并发{' - 调试模式' if debug_mode else ''})")
                        all_dialogues.extend(batch_info["batch_results"])
                except Exception as e:
                    return [], f"❌ 远程生成失败：{str(e)}", "", gr.update(visible=False), ""
            else:
                # 混合模式：本地2个请求 + DeepSeek 5个请求
                llm_client = LLMClient(["ollama", "deepseek"], deepseek_api_key)
                generator = DialogueGenerator(llm_client)
                
                # 生成对话
                all_dialogues = []
                progress(0, desc="开始混合并发生成...")
                
                try:
                    async for batch_info in generator.generate_hybrid_batch(background, topic, tools, count, temperature, debug_mode, character, deepseek_api_key, jsonl_path):
                        progress(batch_info["progress"], 
                                desc=f"混合生成中... {batch_info['current']}/{batch_info['total']} (本地2个 + DeepSeek5个{' - 调试模式' if debug_mode else ''})")
                        all_dialogues.extend(batch_info["batch_results"])
                except Exception as e:
                    return [], f"❌ 混合生成失败：{str(e)}", "", gr.update(visible=False), ""
            
            # 数据已在生成过程中直接写入JSONL文件
            final_path = jsonl_path
            
            # 格式化显示结果
            display_results = []
            for i, d in enumerate(all_dialogues[:10], 1):  # 只显示前10条
                display_results.append(f"=== 对话 {i} ===\n{d['text']}\n")
            
            if len(all_dialogues) > 10:
                display_results.append(f"\n... 还有 {len(all_dialogues) - 10} 条对话")
            
            # 提取调试信息（如果开启了调试模式）
            debug_info_text = ""
            if debug_mode:
                debug_items = []
                for d in all_dialogues:
                    if "debug_info" in d:
                        debug_item = {
                            "text": d["text"][:100] + "...",  # 只显示文本的前100个字符
                            "debug_info": d["debug_info"]
                        }
                        debug_items.append(debug_item)
                
                if debug_items:
                    # 格式化调试信息，突出显示请求和响应数据包
                    debug_info = debug_items[0]["debug_info"]  # 获取第一条对话的调试信息
                    formatted_debug = {
                        "timestamp": datetime.now().isoformat(),
                        "request_data_packet": {
                            "url": debug_info.get("url", "混合模式"),
                            "headers": debug_info.get("headers", {}),
                            "method": "POST",
                            "payload": debug_info.get("payload", {}),
                            "prompt": debug_info.get("prompt", "")
                        },
                        "response_data_packet": {
                            "status": "200 OK",
                            "headers": {
                                "Content-Type": "application/json",
                                "Connection": "keep-alive"
                            },
                            "body": debug_info.get("full_response", {}),
                            "content": debug_info.get("full_response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                        }
                    }
                    debug_info_text = json.dumps(formatted_debug, indent=2, ensure_ascii=False)
        else:
            # 单一模式：仅使用Ollama
            llm_client = LLMClient()
            generator = DialogueGenerator(llm_client)
            
            # 生成对话
            all_dialogues = []
            progress(0, desc="开始生成...")
            
            try:
                async for batch_info in generator.generate_batch(background, topic, tools, count, temperature, debug_mode, character, jsonl_path):
                    progress(batch_info["progress"], 
                            desc=f"生成中... {batch_info['current']}/{batch_info['total']} (使用 Ollama Qwen3:30b-q4_K_M{' - 调试模式' if debug_mode else ''})")
                    all_dialogues.extend(batch_info["batch_results"])
            except Exception as e:
                return [], f"❌ 生成失败：{str(e)}", "", gr.update(visible=False), ""
            
            # 数据已在生成过程中直接写入JSONL文件
            final_path = jsonl_path
            
            # 格式化显示结果
            display_results = []
            for i, d in enumerate(all_dialogues[:10], 1):  # 只显示前10条
                display_results.append(f"=== 对话 {i} ===\n{d['text']}\n")
            
            if len(all_dialogues) > 10:
                display_results.append(f"\n... 还有 {len(all_dialogues) - 10} 条对话")
            
            # 提取调试信息（如果开启了调试模式）
            debug_info_text = ""
            if debug_mode:
                debug_items = []
                for d in all_dialogues:
                    if "debug_info" in d:
                        debug_item = {
                            "text": d["text"][:100] + "...",  # 只显示文本的前100个字符
                            "debug_info": d["debug_info"]
                        }
                        debug_items.append(debug_item)
                
                if debug_items:
                        # 格式化调试信息，突出显示请求和响应数据包
                        debug_info = debug_items[0]["debug_info"]  # 获取第一条对话的调试信息
                        formatted_debug = {
                            "timestamp": datetime.now().isoformat(),
                            "request_data_packet": {
                                "url": LLMConfig.PROVIDERS["ollama"]["url"],
                                "headers": {"Content-Type": "application/json"},  # 本地无需鉴权
                                "method": "POST",
                                "payload": {
                                    "model": debug_info.get("payload", {}).get("model", ""),
                                    "messages": debug_info.get("payload", {}).get("messages", []),
                                    "temperature": debug_info.get("payload", {}).get("temperature", 0),
                                    "max_tokens": debug_info.get("payload", {}).get("max_tokens", 0)
                                },
                                "prompt": debug_info.get("prompt", "")
                            },
                            "response_data_packet": {
                                "status": "200 OK",
                                "headers": {
                                    "Content-Type": "application/json",
                                    "Connection": "keep-alive"
                                },
                                "body": {
                                    "id": debug_info.get("full_response", {}).get("id", ""),
                                    "object": debug_info.get("full_response", {}).get("object", ""),
                                    "created": debug_info.get("full_response", {}).get("created", 0),
                                    "model": debug_info.get("full_response", {}).get("model", ""),
                                    "choices": debug_info.get("full_response", {}).get("choices", []),
                                    "usage": debug_info.get("full_response", {}).get("usage", {})
                                },
                                "content": debug_info.get("full_response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                            }
                        }
                        debug_info_text = json.dumps(formatted_debug, indent=2, ensure_ascii=False)
            
        return (
            display_results, 
            f"✅ 生成完成！共生成 {len(all_dialogues)} 条对话\n📁 已保存文件：\n  - {final_path} (JSONL训练数据){' - 包含调试信息' if debug_mode else ''}", 
            "\n".join(display_results),
            gr.update(visible=debug_mode),  # 根据调试模式控制调试信息区域的可见性
            debug_info_text
        )

    # 创建界面
    with gr.Blocks(title="LLM对话生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 LLM对话生成器")
        gr.Markdown("使用LLM API生成桌宠AI助手的训练对话数据")
        gr.Markdown("💡 **提示**: \n- 生成的数据会保存到指定目录：每个请求的JSON单独保存，完成后合并为JSONL格式\n- 目录路径示例：`./dialogues` 或 `data/output`")
        
        # API状态
        with gr.Row():
            api_status = gr.Textbox(label="API状态", value=check_api_status(), interactive=False)
        
        # 主要内容区域 - 使用两列布局
        with gr.Row():
            # 左侧输入区域
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入参数")
                with gr.Column():
                    background_input = gr.Textbox(
                        label="背景设定",
                        value=DEFAULT_BACKGROUND,
                        lines=4,  # 减少行数使界面更紧凑
                        placeholder="输入背景设定...",
                        info="支持Markdown格式，描述AI助手的世界观、设定等"
                    )
                    with gr.Row():
                        bg_file_picker = gr.File(label="选择背景文件(读取)", file_count="single", type="filepath")
                        bg_load_btn = gr.Button("读取文件", scale=1)
                    with gr.Row():
                        bg_file_in = gr.Textbox(label="保存到(可选)", placeholder="默认保存到 data/ 目录，可填相对/绝对路径", scale=3)
                        bg_save_btn = gr.Button("保存到文件", scale=1)
                        
                    character_input = gr.Textbox(
                        label="AI助手人物设定",
                        value=DEFAULT_CHARACTER,
                        lines=4,
                        placeholder="输入AI助手的性格特征、说话风格等...",
                        info="描述AI助手的个性、语气、行为模式等特征"
                    )
                    with gr.Row():
                        ch_file_picker = gr.File(label="选择人物文件(读取)", file_count="single", type="filepath")
                        ch_load_btn = gr.Button("读取文件", scale=1)
                    with gr.Row():
                        ch_file_in = gr.Textbox(label="保存到(可选)", placeholder="默认保存到 data/ 目录，可填相对/绝对路径", scale=3)
                        ch_save_btn = gr.Button("保存到文件", scale=1)
                        




            with gr.Column(scale=4):
                gr.Markdown("### 🧭 话题与工具")
                topic_input = gr.Textbox(
                    label="话题",
                    value="用户想要AI助手帮助操作电脑的某功能",
                    placeholder="输入要生成的话题...",
                    info="定义对话的主题和场景"
                )
                tools_input = gr.Textbox(
                    label="可用工具（逗号分隔）",
                    value=", ".join(DEFAULT_TOOLS),
                    placeholder="输入工具列表，用逗号分隔...",
                    info="AI助手可以调用的工具/功能列表"
                )
                gr.Markdown("### 💡 示例")
                with gr.Column():
                    gr.Examples(
                        examples=[
                            ["用户询问天气情况，AI助手查询并提供建议", "天气API, 日程管理系统"],
                            ["用户需要整理文件，AI助手协助分类和归档", "查找文件, 整理桌面, 打开程序, 文件分类器"],
                            ["用户想学习新技能，AI助手提供个性化指导", "知识图谱API, 学习进度追踪, 创作灵感库, 技能树系统"],
                            ["用户情绪低落，AI助手进行陪伴和鼓励", "情绪分析器, 音乐推荐引擎, 冥想指导系统"],
                            ["用户探索游戏世界，AI助手引导冒险", "维度扫描器, 魔法花园系统, 法则分析仪, 灵子能量扫描器"],
                            ["用户需要制定工作计划，AI助手协助安排任务", "任务管理器, 日历同步, 优先级分析器, 时间追踪器"],
                            ["用户想要健康饮食，AI助手推荐营养搭配", "营养分析API, 食谱数据库, 健康监测器, 购物清单生成器"],
                            ["用户准备旅行出游，AI助手帮助规划行程", "地图导航API, 酒店预订系统, 天气预报, 景点推荐引擎"],
                            ["用户想要学习编程，AI助手提供学习路径", "代码编辑器, 在线教程库, 练习题生成器, 进度跟踪系统"],
                            ["用户需要购买商品，AI助手比较价格和评价", "价格比较API, 评价分析器, 优惠券搜索器, 购物车管理"],
                            ["用户想要锻炼身体，AI助手制定运动计划", "运动追踪器, 健身视频库, 心率监测器, 卡路里计算器"],
                            ["用户需要创作内容，AI助手提供灵感和素材", "创意生成器, 素材库搜索, 排版工具, 发布平台接口"],
                            ["用户想要社交聊天，AI助手推荐话题和活动", "兴趣匹配器, 活动推荐系统, 聊天话题库, 社交网络接口"],
                            ["用户需要理财投资，AI助手分析市场趋势", "股票API, 财务分析器, 风险评估器, 投资建议系统"],
                            ["用户想要放松娱乐，AI助手推荐音乐影视", "音乐推荐引擎, 影视数据库, 心情分析器, 娱乐内容筛选器"]
                        ],
                        inputs=[topic_input, tools_input],
                        label="话题示例"
                    )
            # 右侧输出区域
            with gr.Column(scale=1):
                gr.Markdown("### 📊 输出结果")
                with gr.Column():
                    with gr.Row():
                        count_input = gr.Slider(
                            label="生成数量",
                            minimum=1,
                            maximum=1000,
                            value=20,
                            step=1,
                            scale=2
                        )
                        temperature_input = gr.Slider(
                            label="创造性程度",
                            minimum=0.1,
                            maximum=1.5,
                            value=0.8,
                            step=0.1,
                            scale=2,
                            info="较低值更保守，较高值更创造性"
                        )
                        random_character = gr.Checkbox(
                            label="随机人物设定",
                            value=False,
                            scale=1,
                            info="启用后将自动从10种灵灵生命体中随机选择"
                        )
                        random_topic = gr.Checkbox(
                            label="随机话题",
                            value=False,
                            scale=1,
                            info="启用后将自动从15个话题示例中随机选择"
                        )
                        
                    with gr.Row():
                        directory_input = gr.Textbox(
                            label="保存目录",
                            value="./dialogues",
                            placeholder="输入保存目录路径...",
                            scale=3,
                            interactive=True
                        )
                        auto_directory = gr.Checkbox(
                            label="使用默认目录",
                            value=True,
                            scale=1
                        )
                    with gr.Row():
                        deepseek_api_key = gr.Textbox(
                            label="DeepSeek API密钥 (可选)",
                            placeholder="输入DeepSeek API密钥以启用混合并发模式...",
                            type="password",
                            scale=3,
                            info="留空则仅使用本地Ollama，填入则启用混合并发模式"
                        )
                    with gr.Row():
                        remote_only = gr.Checkbox(
                            label="仅使用远程API",
                            value=False,
                            scale=1,
                            info="启用后将跳过本地Ollama，仅使用DeepSeek API进行并发请求（需要API密钥）"
                        )
                    with gr.Row():
                        generate_btn = gr.Button("🚀 开始生成", variant="primary", scale=2)
                        test_btn = gr.Button("🔍 仅测试", variant="secondary", scale=1)
                        debug_mode = gr.Checkbox(
                            label="调试模式",
                            value=False,
                            scale=1
                        )
                    
                    status_output = gr.Textbox(
                        label="状态", 
                        interactive=False
                    )
                    results_output = gr.Textbox(
                        label="生成结果预览", 
                        lines=20, 
                        interactive=False,
                        max_lines=30
                    )
                    debug_info_output = gr.Code(
                        label="调试信息 - 请求和响应数据包", 
                        language="json",
                        lines=25, 
                        interactive=False,
                        visible=False,
                        max_lines=50
                    )
        
            # 事件处理
            def toggle_directory_input(auto_dir):
                """切换目录输入框的交互状态"""
                return gr.update(interactive=not auto_dir)
            
            auto_directory.change(
                fn=toggle_directory_input,
                inputs=auto_directory,
                outputs=directory_input
            )
            
            # 背景/人物 读写
            def on_bg_load(path):
                content = load_background_from_file(path)
                return content
            
            def on_bg_save(content, path):
                msg = save_background_to_file(content, path)
                return msg
            
            def on_ch_load(path):
                content = load_character_from_file(path)
                return content
            
            def on_ch_save(content, path):
                msg = save_character_to_file(content, path)
                return msg
            
            bg_load_btn.click(fn=on_bg_load, inputs=[bg_file_picker], outputs=[background_input])
            bg_save_btn.click(fn=on_bg_save, inputs=[background_input, bg_file_in], outputs=[status_output])
            ch_load_btn.click(fn=on_ch_load, inputs=[ch_file_picker], outputs=[character_input])
            ch_save_btn.click(fn=on_ch_save, inputs=[character_input, ch_file_in], outputs=[status_output])
        
        def prepare_generation(background, topic, tools_text, count, temperature, directory, auto_dir, debug_mode, character, random_character, random_topic, deepseek_api_key, remote_only, progress=gr.Progress()):
            """准备生成，如果需要则更新目录"""
            if auto_dir:
                directory = "./dialogues"
            
            # 在调试模式下，强制仅发送一条请求，忽略数量滑块
            if debug_mode:
                count = 1
            
            # 运行异步生成函数（将progress传递给异步函数）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # 获取所有返回值，包括调试信息
                dialogues, status, results, debug_visible, debug_info = loop.run_until_complete(
                    generate_dialogues(background, topic, tools_text, count, temperature, directory, auto_dir, debug_mode, character, random_character, random_topic, deepseek_api_key, remote_only, progress)
                )
                
                # 如果开启了调试模式但没有调试信息，添加提示
                if debug_mode and not debug_info:
                    debug_info = json.dumps({
                        "message": "未能获取调试信息，请重试或检查API连接",
                        "timestamp": datetime.now().isoformat()
                    }, indent=2, ensure_ascii=False)
                
                # 确保调试信息区域在调试模式下可见
                debug_visible = gr.update(visible=debug_mode, value=debug_info if debug_info else "等待生成...")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                status = f"❌ 生成过程中出错: {str(e)}"
                results = ""
                debug_visible = gr.update(visible=debug_mode)
                debug_info = json.dumps({
                    "error": str(e),
                    "details": error_details,
                    "timestamp": datetime.now().isoformat()
                }, indent=2, ensure_ascii=False) if debug_mode else ""
            finally:
                loop.close()
            
            return directory, status, results, debug_visible, debug_info
        
        def test_single_generation(background, topic, tools_text, temperature, debug_mode, character, random_character, random_topic):
            """仅测试生成一条对话，不保存"""
            # 解析工具列表
            tools = [tool.strip() for tool in tools_text.split(",") if tool.strip()]
            if not tools:
                return "❌ 错误：请至少输入一个工具", "", gr.update(visible=False), ""
            
            # 如果启用随机人物设定，从池中随机选择
            if random_character:
                try:
                    character = random.choice(CHARACTER_POOL)
                except Exception:
                    pass
            
            # 如果启用随机话题，从话题示例中随机选择
            if random_topic:
                try:
                    topic_examples = [
                        "用户询问天气情况，AI助手查询并提供建议",
                        "用户需要整理文件，AI助手协助分类和归档",
                        "用户想学习新技能，AI助手提供个性化指导",
                        "用户情绪低落，AI助手进行陪伴和鼓励",
                        "用户探索游戏世界，AI助手引导冒险",
                        "用户需要制定工作计划，AI助手协助安排任务",
                        "用户想要健康饮食，AI助手推荐营养搭配",
                        "用户准备旅行出游，AI助手帮助规划行程",
                        "用户想要学习编程，AI助手提供学习路径",
                        "用户需要购买商品，AI助手比较价格和评价",
                        "用户想要锻炼身体，AI助手制定运动计划",
                        "用户需要创作内容，AI助手提供灵感和素材",
                        "用户想要社交聊天，AI助手推荐话题和活动",
                        "用户需要理财投资，AI助手分析市场趋势",
                        "用户想要放松娱乐，AI助手推荐音乐影视"
                    ]
                    topic = random.choice(topic_examples)
                except Exception:
                    pass
            
            # 创建临时生成器
            llm_client = LLMClient()
            test_generator = DialogueGenerator(llm_client)
            
            # 运行异步单次生成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # 调用单次生成函数
                dialogues = loop.run_until_complete(
                    test_generator.generate_single_dialogue(background, topic, tools, temperature, debug_mode, character)
                )
                
                if dialogues:
                    # 格式化显示结果
                    display_text = f"=== 测试结果 ===\n{dialogues[0]['text']}\n\n🎯 测试完成：成功生成1条对话（未保存）"
                    result_text = dialogues[0]['text']
                    
                    # 提取调试信息
                    debug_info_text = ""
                    debug_visible = gr.update(visible=False)
                    
                    if debug_mode and "debug_info" in dialogues[0]:
                        debug_info = dialogues[0]["debug_info"]
                        formatted_debug = {
                            "test_mode": True,
                            "timestamp": datetime.now().isoformat(),
                            "request": {
                                "prompt": debug_info.get("prompt", ""),
                                "payload": debug_info.get("payload", {})
                            },
                            "response": {
                                "full_response": debug_info.get("full_response", {}),
                                "content": debug_info.get("full_response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                            }
                        }
                        debug_info_text = json.dumps(formatted_debug, indent=2, ensure_ascii=False)
                        debug_visible = gr.update(visible=True, value=debug_info_text)
                    elif debug_mode:
                        debug_info_text = json.dumps({
                            "test_mode": True,
                            "message": "测试模式，无调试信息",
                            "timestamp": datetime.now().isoformat()
                        }, indent=2, ensure_ascii=False)
                        debug_visible = gr.update(visible=True, value=debug_info_text)
                    
                    return display_text, result_text, debug_visible, debug_info_text
                else:
                    return "❌ 测试失败：未能生成有效对话", "", gr.update(visible=False), ""
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                status = f"❌ 测试失败: {str(e)}"
                
                debug_info_text = ""
                debug_visible = gr.update(visible=False)
                if debug_mode:
                    debug_info_text = json.dumps({
                        "test_mode": True,
                        "error": str(e),
                        "details": error_details,
                        "timestamp": datetime.now().isoformat()
                    }, indent=2, ensure_ascii=False)
                    debug_visible = gr.update(visible=True, value=debug_info_text)
                
                return status, "", debug_visible, debug_info_text
                
            finally:
                loop.close()
        
        generate_btn.click(
            fn=prepare_generation,
            inputs=[background_input, topic_input, tools_input, count_input, temperature_input, directory_input, auto_directory, debug_mode, character_input, random_character, random_topic, deepseek_api_key, remote_only],
            outputs=[directory_input, status_output, results_output, debug_info_output, debug_info_output]
        )
        
        test_btn.click(
            fn=test_single_generation,
            inputs=[background_input, topic_input, tools_input, temperature_input, debug_mode, character_input, random_character, random_topic],
            outputs=[status_output, results_output, debug_info_output, debug_info_output]
        )
        
        # 添加调试模式复选框的事件处理，控制调试信息区域的可见性
        def update_debug_visibility(debug_enabled):
            """更新调试信息区域的可见性和初始内容"""
            if debug_enabled:
                return gr.update(
                    visible=True, 
                    value="调试模式已启用\n\n生成对话时将显示:\n- 发送的请求数据包\n- 接收的响应数据包\n\n点击「开始生成」按钮开始生成对话"
                )
            else:
                return gr.update(visible=False)
        
        debug_mode.change(
            fn=update_debug_visibility,
            inputs=[debug_mode],
            outputs=[debug_info_output]
        )
        
        # 页脚
        gr.Markdown("---")
        gr.Markdown("🔧 **使用提示**: 默认使用Ollama本地Qwen3:30b-a3b-q4_K_M模型，API地址为http://192.168.3.20:11434/ ，请先确保已通过 ollama pull 下载模型")
    
    return demo

# ==================== 主程序 ====================

if __name__ == "__main__":
    print("启动LLM对话生成器GUI...")
    print("如需共享链接，请设置 share=True")
    demo = create_gui()
    # 允许通过环境变量覆盖端口，默认7864
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7864"))
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=server_port,
        show_error=True,
        inbrowser=False  # 在IDE中预览
    )

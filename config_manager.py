# config_manager.py
import json
import os
import httpx
import streamlit as st
from typing import Dict, Any, Tuple
import traceback

CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "user_settings.json")

# 默认配置（首次使用时自动创建）
DEFAULT_CONFIG = {
    "model_provider": "deepseek",
    "providers": {
        "deepseek": {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "model_name": "deepseek-chat",
            "test_endpoint": "/models"
        },
        "openai": {
            "name": "OpenAI ChatGPT",
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model_name": "gpt-4o-mini",
            "test_endpoint": "/models"
        },
        "groq": {
            "name": "Groq（超快）",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "",
            "model_name": "llama-3.1-70b-versatile",
            "test_endpoint": "/models"
        },
        "claude": {
            "name": "Anthropic Claude",
            "base_url": "https://api.anthropic.com/v1",
            "api_key": "",
            "model_name": "claude-3-5-sonnet-20241022",
            "test_endpoint": "/messages"
        },
        "gemini": {
            "name": "Google Gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1",
            "api_key": "",
            "model_name": "gemini-1.5-pro",
            "test_endpoint": "/models/gemini-1.5-pro:generateContent?key={api_key}"
        }
    },
    "proxy": {
        "enabled": False,
        "protocol": "http",
        "host": "",
        "port": "",
        "username": "",
        "password": ""
    },
    "rag_settings": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retriever_k": 6
    }
}

def load_config() -> Dict[Any, Any]:
    """加载持久化配置"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 合并默认值（防止新增字段缺失）
                return merge_config(DEFAULT_CONFIG, data)
        except Exception as e:
            st.warning(f"配置加载失败，将使用默认配置: {e}")
    # 首次使用或文件损坏时创建默认配置
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[Any, Any]):
    """保存配置到本地文件"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def merge_config(default: dict, user: dict) -> dict:
    """深度合并配置，确保新增字段不会丢失"""
    result = default.copy()
    for k, v in user.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result

def get_proxy_url(config: dict) -> str | None:
    """返回单个代理字符串（httpx 推荐方式）"""
    p = config["proxy"]
    if not p["enabled"] or not p["host"] or not p["port"]:
        return None

    protocol = p["protocol"]
    if protocol.startswith("socks"):
        st.warning("SOCKS 代理需安装 pysocks：pip install pysocks")

    url = f"{protocol}://"
    if p["username"]:
        url += f"{p['username']}:{p['password']}@"
    url += f"{p['host']}:{p['port']}"
    return url

def get_proxy_for_httpx(config: dict):
    """返回 httpx 推荐的 proxies 格式：字符串或 None"""
    url = get_proxy_url(config)
    return url

def get_proxy_dict(config: dict) -> Dict[str, str] | None:
    """返回 httpx 所需的 proxies 字典"""
    url = get_proxy_url(config)
    if not url:
        return None
    return {"http://": url, "https://": url}

def test_proxy_connection(config: dict, show_traceback: bool = False) -> Tuple[bool, str]:
    """
    测试代理 + 当前模型 API 连接（完全修复 'dict' object has no attribute 'url'）
    """
    provider_key = config["model_provider"]
    provider = config["providers"][provider_key]
    base_url = provider["base_url"].rstrip("/")
    test_endpoint = provider.get("test_endpoint", "/models")  # 防止缺失
    api_key = provider["api_key"].strip() if provider["api_key"] else ""

    proxies = get_proxy_dict(config)

    headers = {}
    if api_key and provider_key != "gemini":
        if provider_key == "claude":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
            headers["content-type"] = "application/json"
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    try:
        timeout = httpx.Timeout(20.0, connect=10.0)
        client_kwargs = {
            "proxy": get_proxy_for_httpx(config),
            "timeout": timeout,
            "follow_redirects": True,
            "headers": headers
        }

        with httpx.Client(**client_kwargs) as client:
            # 构建完整 URL
            if "{api_key}" in test_endpoint:  # Gemini 特殊
                if not api_key:
                    return False, "Gemini 需要填写 API Key"
                url = base_url + test_endpoint.format(api_key=api_key)
                response = client.get(url)
            elif provider_key == "claude":
                url = base_url + test_endpoint
                data = {
                    "model": provider["model_name"],
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "ping"}]
                }
                response = client.post(url, json=data)
            else:
                url = base_url + test_endpoint
                response = client.get(url)

            # 响应状态判断
            if response.status_code == 200:
                return True, f"成功连接 {provider['name']}！网络和 API 均正常"
            elif response.status_code == 401:
                return False, "API Key 无效或未填写"
            elif response.status_code == 403:
                return False, "API Key 权限不足或地区限制"
            elif response.status_code == 429:
                return False, "请求超限（Rate Limit）"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:300]}"

    except httpx.ConnectTimeout:
        return False, "连接超时（代理或网络无法到达服务器）"
    except httpx.ProxyError as e:
        return False, "代理配置错误或代理服务器不可用"
    except httpx.NetworkError as e:
        return False, "网络错误（DNS 解析失败或无网络）"
    except httpx.RequestError as e:
        # 关键修复：安全提取 URL，避免 'dict' object has no attribute 'url'
        error_detail = str(e)
        request_obj = getattr(e, "request", None)
        if request_obj is not None:
            try:
                # 安全访问 url（可能是 dict 或 Request 对象）
                req_url = request_obj.url if hasattr(request_obj, "url") else str(request_obj.get("url", "未知"))
                error_detail += f" (目标: {req_url})"
            except:
                pass
        return False, f"请求失败: {error_detail}"
    except Exception as e:
        # ====== 关键：捕获未知错误并输出完整堆栈 ======
        full_traceback = traceback.format_exc()
        simple_msg = f"未知错误: {str(e)}"

        if show_traceback:
            return False, f"{simple_msg}\n\n**完整错误堆栈：**\n```\n{full_traceback}\n```"
        else:
            return False, f"{simple_msg}（点击“显示详细错误”查看堆栈）"
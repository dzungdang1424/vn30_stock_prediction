"""
VN30 Stock Prediction — LLM Client Abstraction (v1.1)

Abstraction layer cho LLM providers.
Mặc định dùng Google Gemini 2.5 Flash (Google AI Studio — Free API) theo CLAUDE.md §3.

Supported providers:
    - "gemini": Google Gemini REST API (default) — không cần SDK
    - "ollama": Local Ollama server
    - "anthropic": Claude API
    - "openai": OpenAI API

"""

import os
import json
import requests
from loguru import logger


class LLMClient:
    """
    Unified LLM client — switch provider bằng constructor parameter.

    Default: Gemini 2.5 Flash qua REST API (Google AI Studio free tier).
    API key đọc từ os.environ["GEMINI_API_KEY"].

    Usage:
        client = LLMClient()  # default: gemini/gemini-2.5-flash
        response = client.chat("Phân tích FPT.VN", system="Bạn là trợ lý...")
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        max_tokens: int = 1024,
    ):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens

        # Fail-fast: validate model prefix khớp provider
        _GEMINI_PREFIX    = ("gemini-",)
        _OLLAMA_NOTE      = None   # Ollama model names tự do — không validate
        _ANTHROPIC_PREFIX = ("claude-",)
        _OPENAI_PREFIX    = ("gpt-", "o1", "o3", "text-")

        if self.provider == "gemini" and not any(model.startswith(p) for p in _GEMINI_PREFIX):
            raise ValueError(
                f"Model '{model}' không hợp lệ cho provider 'gemini'. "
                f"Model Gemini phải bắt đầu bằng 'gemini-' (ví dụ: 'gemini-2.5-flash')."
            )
        if self.provider == "anthropic" and not any(model.startswith(p) for p in _ANTHROPIC_PREFIX):
            raise ValueError(
                f"Model '{model}' không hợp lệ cho provider 'anthropic'. "
                f"Model Claude phải bắt đầu bằng 'claude-' (ví dụ: 'claude-3-5-sonnet-20241022')."
            )
        if self.provider == "openai" and not any(model.startswith(p) for p in _OPENAI_PREFIX):
            raise ValueError(
                f"Model '{model}' không hợp lệ cho provider 'openai'. "
                f"Model OpenAI phải bắt đầu bằng 'gpt-', 'o1', 'o3'... (ví dụ: 'gpt-4o')."
            )
        self.max_tokens = max_tokens

    def chat(self, prompt: str, system: str = "") -> str:
        """
        Gửi prompt và nhận response text.

        Args:
            prompt: User message
            system: System prompt (optional)

        Returns:
            Response text từ LLM

        Raises:
            TimeoutError: Khi LLM không phản hồi trong timeout
            ConnectionError: Khi không kết nối được server
            RuntimeError: Lỗi khác từ LLM
        """
        if self.provider == "gemini":
            return self._call_gemini(prompt, system)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, system)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system)
        elif self.provider == "openai":
            return self._call_openai(prompt, system)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ── Gemini (Primary — CLAUDE.md §6) ─────────────────────────────────────

    def _call_gemini(self, prompt: str, system: str) -> str:
        """
        Gọi Google Gemini REST API trực tiếp (không cần SDK).
        Theo CLAUDE.md §6: generativelanguage.googleapis.com
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Thiếu GEMINI_API_KEY. "
                "Thêm vào file .env: GEMINI_API_KEY=your_key_here\n"
                "Lấy free API key tại: https://aistudio.google.com/app/apikey"
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )

        # Gemini dùng role "user"/"model", không có "system" role chuẩn
        # Workaround: gộp system prompt vào đầu message đầu tiên
        contents = []
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"{system}\n\n{prompt}"}]
            })
        else:
            contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,  # thấp để output nhất quán
                "maxOutputTokens": self.max_tokens
            }
        }

        import time
        max_retries = 3
        base_delay = 5  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()  # raise HTTPError nếu 4xx/5xx
                break  # Thành công thì thoát loop
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    logger.warning(f"Gemini API timeout (lần {attempt + 1}/{max_retries}), thử lại sau {base_delay}s...")
                    time.sleep(base_delay)
                    base_delay *= 2
                    continue
                raise TimeoutError(
                    f"Gemini API không phản hồi sau {self.timeout}s (đã thử {max_retries + 1} lần). "
                    "Kiểm tra kết nối mạng."
                )
            except requests.exceptions.HTTPError as e:
                # Nếu là lỗi 503 (Overloaded) hoặc 429 (Too Many Requests) -> thử lại
                if resp.status_code in (503, 429) and attempt < max_retries:
                    logger.warning(f"Gemini API báo lỗi {resp.status_code} (lần {attempt + 1}/{max_retries}), server đang quá tải. Thử lại sau {base_delay}s...")
                    time.sleep(base_delay)
                    base_delay *= 2
                    continue
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}") from e
            except requests.exceptions.ConnectionError:
                if attempt < max_retries:
                    logger.warning(f"Lỗi kết nối Gemini API (lần {attempt + 1}/{max_retries}). Thử lại sau {base_delay}s...")
                    time.sleep(base_delay)
                    base_delay *= 2
                    continue
                raise ConnectionError("Không kết nối được Gemini API. Kiểm tra kết nối mạng.")

        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected Gemini response format: {data}") from e

    # ── Ollama ───────────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str, system: str) -> str:
        """Gọi Ollama local server."""
        url = f"{self.base_url}/api/chat"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.Timeout:
            # Retry 1 lần
            logger.warning("Ollama timeout, retrying...")
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
            except requests.exceptions.Timeout:
                raise TimeoutError(
                    f"Ollama không phản hồi sau {self.timeout}s (2 lần thử). "
                    "Kiểm tra: ollama serve đang chạy?"
                )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Không kết nối được Ollama server. "
                "Chạy: ollama serve & ollama pull <model>"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    # ── Anthropic ────────────────────────────────────────────────────────────

    def _call_anthropic(self, prompt: str, system: str) -> str:
        """Gọi Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu ANTHROPIC_API_KEY")

        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        return response.content[0].text

    # ── OpenAI ───────────────────────────────────────────────────────────────

    def _call_openai(self, prompt: str, system: str) -> str:
        """Gọi OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu OPENAI_API_KEY")

        client = openai.OpenAI(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
"""LLM API 封装：聊天、重试、超时、流式输出（Day 2 版本）"""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=30.0,      # 超时：30 秒没响应就抛 TimeoutError
    max_retries=0,     # SDK 自带重试关掉，我们用手写的（教学目的，看得见每一步）
)

# 可重试的错误：限流、服务器错误、超时、连接问题
RETRYABLE_ERRORS = ("Rate limit", "Timeout", "Connection", "Overloaded", "429", "500", "502", "503")


def _should_retry(error: Exception) -> bool:
    """判断这个错误值不值得重试"""
    msg = str(error)
    return any(keyword in msg for keyword in RETRYABLE_ERRORS)


def chat(
    messages: list,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
) -> str:
    """
    聊天接口（带手写重试 + 指数退避）。

    messages: OpenAI 格式的对话列表，例如
        [{"role": "system", "content": "你是机械工程师助手"},
         {"role": "user", "content": "45号钢的常用热处理方式？"}]
    temperature: 0~2，越低越稳定，知识问答场景用低值
    """
    last_error = None
    for attempt in range(3):                    # 最多尝试 3 次
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if not _should_retry(e):
                raise                           # 不可重试的错误（如 key 错），直接抛出
            wait = 2 ** attempt                 # 指数退避：1s → 2s → 4s
            print(f"[llm] 第{attempt + 1}次调用失败({e})，{wait}秒后重试...")
            time.sleep(wait)
    raise last_error                             # 重试全失败，抛出最后的错误


def chat_stream(messages: list, model: str = "deepseek-chat", temperature: float = 0.3):
    """
    流式聊天：逐段返回内容（生成器）。

    为什么需要流式：普通调用要等模型写完整个回答才返回，长回答要等 10-30 秒；
    流式是模型每生成几个字就推过来一段，前端可以实时显示，体验完全不同。
    Day 5 的 Streamlit 界面就用它。
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,        # 关键参数：开启流式
    )
    for chunk in response:  # 每收到一段就 yield 出去
        delta = chunk.choices[0].delta.content
        if delta:           # 有的 chunk 是空的（只有角色信息），跳过
            yield delta


if __name__ == "__main__":
    # 测试 1：普通聊天（多轮 + system 提示词）
    msgs = [
        {"role": "system", "content": "你是一个严谨的机械工程助手，回答控制在100字内。"},
        {"role": "user", "content": "45号钢常用的热处理方式有哪些？"},
    ]
    print("=== 普通调用（等待完整回答）===")
    print(chat(msgs))

    # 测试 2：流式输出
    print("\n=== 流式调用（逐字打印）===")
    for piece in chat_stream(msgs):
        print(piece, end="", flush=True)
    print("\n\n=== Day 2 测试通过 ===")
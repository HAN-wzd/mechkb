"""LLM API 封装（Day 2 会加入重试和超时，今天先跑通）"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取 .env 文件，把里面的变量装进环境变量

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量取 key，不写死在代码里
    base_url="https://api.deepseek.com",    # DeepSeek 的接口地址（OpenAI 兼容格式）
)


def chat(prompt: str) -> str:
    """调用 DeepSeek 聊天接口，返回模型回复文本"""
    response = client.chat.completions.create(
        model="deepseek-chat",                      # 模型名
        messages=[{"role": "user", "content": prompt}],  # 对话历史，这里只有一条用户提问
    )
    return response.choices[0].message.content      # 从返回结构里取出回复文本


if __name__ == "__main__":
    # 直接运行本文件时执行这里；被别的文件 import 时不执行
    print(chat("用一句话说明什么是检索增强生成（RAG）"))
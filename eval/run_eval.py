"""检索命中率评测：遍历评测集，检查期望文档是否出现在检索结果 top-k 中"""
import json
import sys

from core.vectorstore import search


def load_questions(path: str = "eval/questions.json") -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate(k: int = 3) -> None:
    questions = load_questions()
    retrieval_qs = [q for q in questions if q["expect"] is not None]
    hits = 0
    misses = []

    for item in retrieval_qs:
        docs = search(item["q"], k=k)
        sources = [d.metadata["source"] for d in docs]
        if item["expect"] in sources:
            hits += 1
        else:
            misses.append((item["q"], item["expect"], sources))

    total = len(retrieval_qs)
    rate = hits / total * 100
    print(f"\n=== 检索命中率: {hits}/{total} = {rate:.1f}% (top-{k}) ===")
    if misses:
        print("未命中的题：")
        for q, expect, got in misses:
            print(f"  ✗ {q}")
            print(f"    期望: {expect}  实际检出: {got}")


if __name__ == "__main__":
    evaluate()
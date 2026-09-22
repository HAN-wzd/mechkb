"""chunk_size 对比实验：三档参数分别重建向量库并评测命中率"""
from core.vectorstore import build_vectorstore, search
from eval.run_eval import load_questions

CONFIGS = [
    {"chunk_size": 300, "chunk_overlap": 30},
    {"chunk_size": 500, "chunk_overlap": 50},
    {"chunk_size": 800, "chunk_overlap": 100},
]


def hit_rate(k: int = 3) -> float:
    questions = [q for q in load_questions() if q["expect"] is not None]
    hits = sum(
        1 for item in questions
        if item["expect"] in [d.metadata["source"] for d in search(item["q"], k=k)]
    )
    return hits / len(questions) * 100


if __name__ == "__main__":
    results = []
    for cfg in CONFIGS:
        print(f"\n>>> 构建 chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}")
        build_vectorstore(**cfg)  # 重建向量库（删集合再入库）
        rate = hit_rate()
        results.append((cfg["chunk_size"], rate))
        print(f">>> 命中率: {rate:.1f}%")

    print("\n===== 对比实验结果 =====")
    print(f"{'chunk_size':<12}{'命中率':<10}")
    for size, rate in results:
        print(f"{size:<12}{rate:.1f}%")
        
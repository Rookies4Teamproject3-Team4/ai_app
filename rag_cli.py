# rag_cli.py
import argparse, json
from rag_pipeline import answer_question

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="faiss_index")
    p.add_argument("--q", required=True)
    p.add_argument("--search", default="mmr", choices=["mmr", "similarity"])
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--fetch_k", type=int, default=20)
    p.add_argument("--provider", default="groq", choices=["groq", "openai"])
    p.add_argument("--model", default="moonshotai/kimi-k2-instruct-0905")
    args = p.parse_args()

    result = answer_question(
        index_dir=args.index,
        question=args.q,
        search_type=args.search,
        k=args.k,
        fetch_k=args.fetch_k,
        provider=args.provider,
        model=args.model,
    )
    print("\n=== 답변 ===")
    print(result["answer"])
    print("\n=== 소스 ===")
    print(json.dumps(result["sources"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

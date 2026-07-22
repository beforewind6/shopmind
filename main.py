"""ShopMind 主入口

用法:
  python main.py --init         初始化系统（生成数据 + 向量化）
  python main.py --api          启动 Flask API 服务
  python main.py --ui           启动 Streamlit 聊天界面
  python main.py --eval         运行三维评估
"""

import sys
import argparse
from pathlib import Path


def init_system():
    print("=" * 50)
    print("  ShopMind 系统初始化")
    print("=" * 50)

    # 1. 生成数据
    print("\n[1/3] 生成演示数据...")
    from data.generate_demo_data import generate_all
    generate_all()

    # 2. 向量化知识库
    print("\n[2/3] 向量化知识库...")
    from rag.vector_store import VectorStore
    from rag.document_loader import DocumentLoader

    vs = VectorStore()
    loader = DocumentLoader(chunk_size=512, chunk_overlap=64)
    kb_dir = Path(__file__).parent / "data" / "knowledge"
    docs = loader.load_directory(str(kb_dir))
    vs.add_documents(docs, "ecommerce_knowledge")
    vs.save_all()
    print(f"  已向量化 {len(docs)} 个文本块 -> {vs.get_collection_size()} 条")

    print("\n[3/3] 初始化完成！")
    print("  python main.py --api  → 启动 API 服务")
    print("  python main.py --ui   → 启动聊天界面")
    print("  python main.py --eval → 运行评估")


def run_api():
    from api.app import app, init_system
    init_system()
    from utils.config import config
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8002)
    print(f"\n  ShopMind API: http://{host}:{port}")
    print(f"  接口文档: http://{host}:{port}/")
    app.run(host=host, port=port, debug=True)


def run_ui():
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run(["streamlit", "run", str(ui_path), "--server.port", "8501"])


def run_eval():
    from evaluation.evaluate import evaluate_retrieval, evaluate_generation, evaluate_end_to_end
    evaluate_retrieval()
    evaluate_generation()
    evaluate_end_to_end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopMind - 电商智能客服 Agent 系统")
    parser.add_argument("--init", action="store_true", help="初始化系统")
    parser.add_argument("--api", action="store_true", help="启动 API 服务")
    parser.add_argument("--ui", action="store_true", help="启动 Streamlit UI")
    parser.add_argument("--eval", action="store_true", help="运行评估")
    args = parser.parse_args()

    if args.init:
        init_system()
    elif args.api:
        run_api()
    elif args.ui:
        run_ui()
    elif args.eval:
        run_eval()
    else:
        print("ShopMind v1.0 - 请选择运行模式:")
        print("  --init  初始化系统")
        print("  --api   启动 API 服务")
        print("  --ui    启动聊天界面")
        print("  --eval  运行评估")

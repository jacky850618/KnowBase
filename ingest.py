import os
from knowledge_manager import add_document
from config import CHROMA_DB_PATH


def main():
    if not os.path.exists("./data"):
        print("data/ 目录不存在，跳过初始加载")
        return

    for file_name in os.listdir("./data"):
        file_path = os.path.join("./data", file_name)
        if os.path.isfile(file_path):
            print(f"添加文档: {file_name}")
            add_document(file_path)

    print("初始知识库构建完成！")


if __name__ == "__main__":
    main()
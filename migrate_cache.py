import json
import os

def migrate_cache_data():
    """
    将旧的、单一的 cache.json 文件，迁移到新的、分离的缓存结构中。
    - AI释义 -> cache/word_details.json
    - 词向量 -> cache/embeddings.json
    """
    print("--- 开始缓存数据迁移 ---")

    # 1. 加载主配置文件以获取新的缓存路径
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        word_details_cache_file = config.get("word_details_cache_file")
        embeddings_cache_file = config.get("embeddings_cache_file")
        if not word_details_cache_file or not embeddings_cache_file:
            print("错误: 'config.json' 中缺少 'word_details_cache_file' 或 'embeddings_cache_file' 配置。")
            return
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载或解析 'config.json': {e}")
        return

    # 2. 加载旧的缓存文件
    old_cache_file = 'cache.json'
    if not os.path.exists(old_cache_file):
        print(f"信息: 未找到旧的缓存文件 '{old_cache_file}'，无需迁移。")
        return

    try:
        with open(old_cache_file, 'r', encoding='utf-8') as f:
            old_cache_data = json.load(f)
        print(f"成功加载旧的缓存文件 '{old_cache_file}'，包含 {len(old_cache_data)} 条记录。")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"错误: 无法加载或解析旧的缓存文件 '{old_cache_file}': {e}")
        return

    # 3. 分离数据
    word_details_cache = {}
    embeddings_cache = {}

    for key, value in old_cache_data.items():
        if key.startswith('embedding_'):
            # 这是词向量数据
            word = key.replace('embedding_', '', 1)
            embeddings_cache[word] = value
        else:
            # 这是AI释义数据
            word_details_cache[key] = value
    
    print(f"数据分离完成: {len(word_details_cache)} 条单词释义, {len(embeddings_cache)} 条词向量。")

    # 4. 写入新的缓存文件
    try:
        # 确保 cache 目录存在
        cache_dir = os.path.dirname(word_details_cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"已创建缓存目录: '{cache_dir}'")

        # 写入单词释义缓存
        with open(word_details_cache_file, 'w', encoding='utf-8') as f:
            json.dump(word_details_cache, f, ensure_ascii=False, indent=4)
        print(f"已成功写入单词释义缓存到: '{word_details_cache_file}'")

        # 写入词向量缓存
        with open(embeddings_cache_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_cache, f, ensure_ascii=False, indent=4)
        print(f"已成功写入词向量缓存到: '{embeddings_cache_file}'")

    except IOError as e:
        print(f"错误: 写入新缓存文件时出错: {e}")
        return

    # 5. 备份旧的缓存文件
    try:
        backup_file_name = 'cache.json.bak'
        os.rename(old_cache_file, backup_file_name)
        print(f"迁移完成！已将旧的缓存文件重命名为 '{backup_file_name}'。")
    except OSError as e:
        print(f"警告: 重命名旧缓存文件时出错: {e}")
        print(f"请手动删除或备份 '{old_cache_file}' 以免下次运行主程序时出现数据混淆。")

    print("--- 缓存数据迁移成功 ---")

if __name__ == "__main__":
    migrate_cache_data()

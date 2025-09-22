import json
import shutil
import os

def clean_chinese_entries(file_path="example.json"):
    """
    从指定的JSON文件中，移除 "lanFrom" 字段为 "zh-CN" 的单词条目。
    会创建一个原始文件的备份 .bak。
    """
    print(f"--- 开始清洗数据文件: {file_path} ---")

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 未找到。")
        return

    # 2. 创建备份
    backup_path = file_path + ".bak"
    try:
        shutil.copy2(file_path, backup_path)
        print(f"已成功创建备份文件: {backup_path}")
    except Exception as e:
        print(f"错误: 创建备份文件失败: {e}")
        return

    # 3. 加载JSON数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"错误: 加载或解析JSON文件 '{file_path}' 失败: {e}")
        return

    # 4. 过滤数据
    original_count = 0
    cleaned_count = 0
    
    # 确保数据结构符合预期
    if "data" in data and "itemList" in data["data"]:
        for item_group in data["data"]["itemList"]:
            if "list" in item_group:
                entries_to_keep = []
                for entry in item_group["list"]:
                    original_count += 1
                    # 保留 lanFrom 不为 "zh-CN" 的条目
                    if entry.get("lanFrom") != "zh-CHS":
                        entries_to_keep.append(entry)
                    else:
                        cleaned_count += 1
                
                item_group["list"] = entries_to_keep
    
    print(f"数据过滤完成。总条目数: {original_count}, 移除的中文条目数: {cleaned_count}")

    # 5. 写回清洗后的数据
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"已成功将清洗后的数据写回到: {file_path}")
    except IOError as e:
        print(f"错误: 写入文件 '{file_path}' 时出错: {e}")

    print("--- 数据清洗成功 ---")

if __name__ == "__main__":
    clean_chinese_entries()

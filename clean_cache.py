import json
import os

def clean_cache_phonetics():
    """
    一个一次性脚本，用于清理 cache.json 文件中的音标数据。
    它会移除 'phonetic' 字段下 'uk' 和 'us' 值中的所有 '/' 字符。
    """
    cache_file = 'cache.json'
    
    if not os.path.exists(cache_file):
        print(f"缓存文件 '{cache_file}' 未找到，无需清理。")
        return

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"无法读取或解析缓存文件 '{cache_file}'。")
        return
        
    print("开始清理缓存文件中的音标数据...")
    cleaned_count = 0
    
    for word, details in cache_data.items():
        if 'phonetic' in details and isinstance(details.get('phonetic'), dict):
            phonetic_changed = False
            phonetic_data = details['phonetic']
            
            if 'uk' in phonetic_data and isinstance(phonetic_data['uk'], str):
                original_uk = phonetic_data['uk']
                cleaned_uk = original_uk.strip(
                    '/').replace('·', '').replace('-', '').replace('.', '')
                if original_uk != cleaned_uk:
                    phonetic_data['uk'] = cleaned_uk
                    phonetic_changed = True
                
            if 'us' in phonetic_data and isinstance(phonetic_data['us'], str):
                original_us = phonetic_data['us']
                cleaned_us = original_us.strip(
                    '/').replace('·', '').replace('-', '').replace('.', '')
                if original_us != cleaned_us:
                    phonetic_data['us'] = cleaned_us
                    phonetic_changed = True
            
            if phonetic_changed:
                cleaned_count += 1
                # print(f"  - 清理了单词 '{word}' 的音标。")

    if cleaned_count > 0:
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=4)
            print(f"\n清理完成！共处理了 {cleaned_count} 个单词的音标。")
            print(f"缓存文件 '{cache_file}' 已更新。")
        except IOError as e:
            print(f"写入更新后的缓存文件时出错: {e}")
    else:
        print("\n缓存文件中的音标数据已是最新格式，无需清理。")

if __name__ == "__main__":
    clean_cache_phonetics()

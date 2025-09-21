import json
import os
import requests
import time
import re

# --- 全局配置 ---
CONFIG = {}

def load_config():
    """加载配置文件 config.json。"""
    global CONFIG
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            CONFIG = json.load(f)
    except FileNotFoundError:
        print("错误: 配置文件 'config.json' 未找到。")
        print("请将 'config.example.json' 复制为 'config.json' 并填入您的配置。")
        exit(1)
    except json.JSONDecodeError:
        print("错误: 配置文件 'config.json' 格式不正确。")
        exit(1)

def load_cache():
    """加载缓存文件。如果文件不存在或为空，则返回一个空字典。"""
    cache_file = CONFIG.get("cache_file", "cache.json")
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_cache(cache_data):
    """将缓存数据保存到文件。"""
    cache_file = CONFIG.get("cache_file", "cache.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)

def parse_word_data(file_path):
    """从输入的JSON文件中解析单词数据。"""
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data.get("data", {}).get("itemList", []):
            for entry in item.get("list", []):
                words.append(entry.get("word"))
    return words

def get_word_details_from_ai(word):
    """调用AI API获取单词的详细信息，并增加了防死循环机制。"""
    print(f"正在查询单词: {word}")
    
    prompt = f"""
    请为英语单词 "{word}" 提供一份**纯英汉词典式的简洁分析**，目标用户是**高中生**。
    请以结构化的JSON格式返回信息。JSON对象应包含以下键：
    - "phonetic": 包含 "uk" 和 "us" 键的英美音标对象。
    - "synonyms": 近义词列表，每个对象包含 "synonym" 和 "translation"。
    - "phrases": 短语列表，每个对象包含 "phrase" 和 "translation"。
    - "collocations": 固定搭配列表，每个对象包含 "collocation" 和 "translation"。
    - "meanings": 一个**释义列表**。每个对象必须包含：
        - "part_of_speech": 词性 (例如: "n.", "v.", "adj.", "adv.")。
        - "translation": **中文释义**。
        - "sentence": 一个例句。
        - "sentence_translation": 例句的中文翻译。
    - "special_tips": (可选) 一段**极其简短**的中文提示，如无则返回空字符串 ""。
    """

    headers = {
        "Authorization": f"Bearer {CONFIG.get('api_key')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": CONFIG.get("ai_model"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "response_format": { "type": "json_object" },
        "stream": True
    }

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(CONFIG.get("api_endpoint"), headers=headers, json=data, timeout=120, stream=True)
            response.raise_for_status()

            json_string = ""
            print(f"--- [调试] AI流式输出 '{word}' ---")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[len('data: '):]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            content_part = delta.get('content')
                            if content_part:
                                print(content_part, end='', flush=True)
                                json_string += content_part
                                # 死循环检测
                                if len(json_string) > CONFIG.get("ai_response_max_length", 3500):
                                    print(f"\n[错误] AI响应超过 {CONFIG.get('ai_response_max_length', 3500)} 字符，可能已进入死循环。正在终止并重试...")
                                    raise ValueError("AI response too long")
                        except json.JSONDecodeError:
                            pass
            print("\n--- [调试] 流式输出结束 ---")

            return json.loads(json_string)

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"\n获取单词 '{word}' 详情时出错 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"'{word}' 重试失败。")
        except json.JSONDecodeError as e:
            print(f"\n解析AI返回的完整JSON时出错 '{word}': {e}")
            print(f"原始JSON字符串: {json_string}")
            return None
    return None

def generate_html_from_template(words_details):
    """使用模板和单词数据生成最终的HTML文件。"""
    print("正在生成HTML文件...")
    template_file = CONFIG.get("template_file", "template.html")
    output_file = CONFIG.get("output_file", "word_book.html")
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"错误: 模板文件 '{template_file}' 未找到。")
        return

    word_entry_template_str = template_content[template_content.find('<!-- word-entry-template-start -->'):template_content.find('<!-- word-entry-template-end -->')]

    sorted_words = sorted(words_details.keys())
    
    def markdown_to_html(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    all_words_html = ""
    for word in sorted_words:
        details = words_details.get(word)
        if not details: continue

        from collections import defaultdict
        meanings = details.get("meanings", [])
        grouped_meanings = defaultdict(list)
        for m in meanings:
            grouped_meanings[m.get('part_of_speech', '')].append(markdown_to_html(m.get('translation', '')))
        
        definitions_html = "<ul>"
        if not grouped_meanings:
            definitions_html += "<li>无</li>"
        else:
            for pos, translations in grouped_meanings.items():
                translations_str = "; ".join(translations)
                definitions_html += f"<li><strong class='pos'>{pos}</strong> {translations_str}</li>"
        definitions_html += "</ul>"

        examples_html = "<div class='examples-list'>"
        has_examples = False
        sentences_html = ""
        for m in meanings:
            if m.get('sentence'):
                has_examples = True
                original_sentence = m.get('sentence', '')
                highlighted_sentence = re.sub(f'({re.escape(word)})', r'<strong>\1</strong>', original_sentence, flags=re.IGNORECASE)
                sentence = markdown_to_html(highlighted_sentence)
                sentence_translation = markdown_to_html(m.get('sentence_translation', ''))
                sentences_html += f"<div class='sentence-item'>{sentence}<br/><span class='translation'>{sentence_translation}</span></div>"
        
        phrases_and_collocations_html = ""
        phrases = details.get("phrases", [])
        collocations = details.get("collocations", [])
        for item in phrases + collocations:
            has_examples = True
            key = 'phrase' if 'phrase' in item else 'collocation'
            original_text = item.get(key, '')
            highlighted_text = re.sub(f'({re.escape(word)})', r'<strong>\1</strong>', original_text, flags=re.IGNORECASE)
            translation = item.get('translation', '')
            phrases_and_collocations_html += f"<span class='example-item'>{highlighted_text} ({translation})</span>"

        if not has_examples:
            examples_html += "<span>无</span>"
        else:
            examples_html += sentences_html + phrases_and_collocations_html
        examples_html += "</div>"

        synonyms_str = ", ".join(f"{s.get('synonym', '')} ({s.get('translation', '')})" for s in details.get("synonyms", [])) or "无"

        special_tips_raw = details.get("special_tips", "")
        special_tips_section_html = ""
        if special_tips_raw:
            special_tips_formatted = markdown_to_html(special_tips_raw.replace('\n', '<br>'))
            special_tips_section_html = f'<div class="special-tips"><strong>特别提示:</strong> {special_tips_formatted}</div>'

        phonetic = details.get('phonetic', {})
        word_html = word_entry_template_str.replace("{word}", word)
        word_html = word_html.replace("{phonetic_uk}", f"英 {phonetic.get('uk', '')}" if phonetic.get('uk') else "")
        word_html = word_html.replace("{phonetic_us}", f"美 {phonetic.get('us', '')}" if phonetic.get('us') else "")
        word_html = word_html.replace("{definitions_list}", definitions_html)
        word_html = word_html.replace("{examples_list}", examples_html)
        word_html = word_html.replace("{synonyms_list}", synonyms_str)
        word_html = word_html.replace("{special_tips_section}", special_tips_section_html)
        
        all_words_html += f"<div class='word-entry' id='{word}'>{word_html}</div>"

    toc_content_html = ""
    for word in sorted_words:
        toc_content_html += f"<a href='#{word}'>{word}</a><br>"

    final_html = template_content.replace("{toc_content}", toc_content_html)
    final_html = final_html.replace("{words_content}", all_words_html)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)

def main():
    """主函数，集成了增量缓存和AI防死循环机制。"""
    load_config()
    
    words_cache = load_cache()
    print(f"已加载 {len(words_cache)} 个已缓存的单词。")

    words_to_process = parse_word_data(CONFIG.get("input_file", "example.json"))
    words_to_process = sorted(list(set(filter(None, words_to_process))))
    
    total_words = len(words_to_process)
    print(f"准备处理 {total_words} 个单词...")
    
    new_words_count = 0
    for i, word in enumerate(words_to_process):
        print(f"\n--- 进度: [{i+1}/{total_words}] ---")
        if word in words_cache:
            print(f"单词 '{word}' 已在缓存中，直接使用。")
            continue

        details = get_word_details_from_ai(word)
        if details:
            words_cache[word] = details
            new_words_count += 1
            # 增量保存缓存
            if new_words_count % CONFIG.get("cache_save_interval", 5) == 0:
                print(f"\n--- 达到保存点，正在增量保存 {len(words_cache)} 个单词到缓存... ---")
                save_cache(words_cache)
        
        time.sleep(1)

    print("\n所有单词处理完毕，正在进行最终缓存保存...")
    save_cache(words_cache)

    # 只为本次需要处理的单词生成HTML
    final_words_details = {word: words_cache[word] for word in words_to_process if word in words_cache}

    if not final_words_details:
        print("未能获取任何单词的详细信息，程序退出。")
        return

    generate_html_from_template(final_words_details)
    print(f"\n成功生成单词书: {CONFIG.get('output_file', 'word_book.html')}")

if __name__ == "__main__":
    main()

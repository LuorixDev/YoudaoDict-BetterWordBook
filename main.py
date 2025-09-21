import json
import os
import requests
import time
import re
from tqdm import tqdm
from queue import Queue
import threading
import hashlib
from datetime import datetime

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
    prompt = f"""
    请为英语单词 "{word}" 提供一份**纯英汉词典式的简洁分析**，目标用户是**备考大学英语四六级(CET-4/6)的学生**。
    请以结构化的JSON格式返回信息。JSON对象应包含以下键：
    - "difficulty": 一个1到5的整数，表示单词的难度（1=最简单, 5=最难）。
    - "phonetic": 包含 "uk" 和 "us" 键的英美音标对象。
    - "synonyms": 近义词列表，每个对象包含 "synonym" 和 "translation"。
    - "phrases": 短语列表，每个对象包含 "phrase" 和 "translation"。
    - "collocations": 固定搭配列表，每个对象包含 "collocation" 和 "translation"。
    - "meanings": 一个**释义列表**。每个对象必须包含：
        - "part_of_speech": 词性 (例如: "n.", "v.", "adj.", "adv.")。
        - "countability": (仅当词性为名词时) 一个表示可数性的字符串 (例如: 'C' 表示可数, 'U' 表示不可数, 'C/U' 表示两者皆可, 如果不适用则返回空字符串 '')。
        - "inflections": (仅当词性为动词时) 一个包含动词变形的对象 (例如: {{ 'past': '...', 'participle': '...', 'present_participle': '...', 'third_person': '...' }}, 如果不适用则返回空对象 {{}})。
        - "translation": **符合CET-4/6难度的中文释义**。请使用Markdown的 `**text**` 语法来**加粗最常考的1-2个核心释义**。
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
            buffer = ""
            tqdm.write(f"--- [调试] AI流式输出 '{word}' ---")
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
                                json_string += content_part
                                buffer += content_part
                                while '\n' in buffer:
                                    line_to_print, buffer = buffer.split('\n', 1)
                                    tqdm.write(line_to_print)
                                
                                if len(json_string) > CONFIG.get("ai_response_max_length", 3500):
                                    tqdm.write(f"\n[错误] AI响应超长，可能已进入死循环。正在终止并重试...")
                                    raise ValueError("AI response too long")
                        except json.JSONDecodeError:
                            pass
            if buffer:
                tqdm.write(buffer)
            tqdm.write("--- [调试] 流式输出结束 ---")

            details = json.loads(json_string)
            
            # 清洗音标数据
            if 'phonetic' in details and isinstance(details['phonetic'], dict):
                if 'uk' in details['phonetic'] and isinstance(details['phonetic']['uk'], str):
                    details['phonetic']['uk'] = details['phonetic']['uk'].strip(
                        '/').replace('·', '').replace('-', '').replace('.', '')
                if 'us' in details['phonetic'] and isinstance(details['phonetic']['us'], str):
                    details['phonetic']['us'] = details['phonetic']['us'].strip(
                        '/').replace('·', '').replace('-', '').replace('.', '')
            
            return details

        except (requests.exceptions.RequestException, ValueError) as e:
            tqdm.write(f"\n获取单词 '{word}' 详情时出错 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                tqdm.write(f"'{word}' 重试失败。")
        except json.JSONDecodeError as e:
            tqdm.write(f"\n解析AI返回的完整JSON时出错 '{word}': {e}")
            tqdm.write(f"原始JSON字符串: {json_string}")
            return None
    return None

def generate_html_from_template(words_details):
    """使用模板和单词数据生成最终的HTML文件。"""
    print("正在生成HTML文件...")
    cover_template_file = CONFIG.get("cover_template_file")
    word_book_template_file = CONFIG.get("word_book_template_file")
    output_file = CONFIG.get("output_file", "word_book.html")

    cover_html = ""
    if cover_template_file:
        try:
            with open(cover_template_file, 'r', encoding='utf-8') as f:
                cover_template_content = f.read()
            
            # --- 动态计算封面元数据 ---
            word_count = len(words_details)
            generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 创建一个稳定的单词列表字符串用于哈希
            sorted_word_list_str = ",".join(sorted(words_details.keys()))
            word_list_hash = hashlib.sha256(sorted_word_list_str.encode('utf-8')).hexdigest()[:16] # 取前16位

            # --- 注入元数据 ---
            cover_html = cover_template_content.replace("{word_count}", str(word_count))
            cover_html = cover_html.replace("{generation_time}", generation_time)
            cover_html = cover_html.replace("{word_list_hash}", word_list_hash)

        except FileNotFoundError:
            print(f"警告: 封面模板文件 '{cover_template_file}' 未找到，将不生成封面。")

    try:
        with open(word_book_template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"错误: 单词书模板文件 '{word_book_template_file}' 未找到。")
        return

    word_entry_template_str = template_content[template_content.find('<!-- word-entry-template-start -->'):template_content.find('<!-- word-entry-template-end -->')]

    sorted_words = sorted(words_details.keys())
    
    def markdown_to_html(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    def highlight_word(text, current_word):
        """在文本中高亮（加粗）指定的单词，不区分大小写，且匹配整个单词。"""
        pattern = re.compile(r'\b({})\b'.format(re.escape(current_word)), re.IGNORECASE)
        return pattern.sub(r'<strong>\1</strong>', text)

    all_words_html = ""
    for word in sorted_words:
        details = words_details.get(word)
        if not details: continue

        long_word_class = "long-word" if len(word) > 15 else ""
        difficulty = details.get("difficulty", 0)
        difficulty_html = ""
        if isinstance(difficulty, int) and 1 <= difficulty <= 5:
            filled_stars = '<span class="filled-star">★</span>' * difficulty
            empty_stars = '<span class="empty-star">★</span>' * (5 - difficulty)
            difficulty_html = f'<div class="difficulty">{filled_stars}{empty_stars}</div>'

        definitions_html = ""
        meanings = details.get("meanings", [])
        if not meanings:
            definitions_html = "<div class='meaning'>无</div>"
        else:
            from collections import defaultdict
            grouped_meanings = defaultdict(lambda: {'translations': [], 'extra_info': ''})
            
            for m in meanings:
                pos = m.get('part_of_speech', '')
                grouped_meanings[pos]['translations'].append(markdown_to_html(m.get('translation', '')))
                
                if not grouped_meanings[pos]['extra_info']:
                    extra_html_parts = []
                    
                    if 'countability' in m and m['countability']:
                        countability_map = {'C': '[可数]', 'U': '[不可数]', 'C/U': '[可数/不可数]'}
                        mapped_val = countability_map.get(m['countability'], '')
                        if mapped_val:
                            extra_html_parts.append(mapped_val)
                    
                    if 'inflections' in m and m['inflections']:
                        inflection_parts = []
                        inf = m['inflections']
                        if inf.get('third_person'): inflection_parts.append(f"三单: {inf['third_person']}")
                        if inf.get('past'): inflection_parts.append(f"过去式: {inf['past']}")
                        if inf.get('participle'): inflection_parts.append(f"过去分词: {inf['participle']}")
                        if inf.get('present_participle'): inflection_parts.append(f"现在分词: {inf['present_participle']}")
                        if inflection_parts:
                            extra_html_parts.append(f"[{'; '.join(inflection_parts)}]")
                    
                    if extra_html_parts:
                         grouped_meanings[pos]['extra_info'] = f"<span class='grammar-info'>{' '.join(extra_html_parts)}</span>"

            parts = []
            for pos, data in grouped_meanings.items():
                translations_str = "; ".join(data['translations'])
                parts.append(f"<span class='pos'>{pos}</span> {data['extra_info']} {translations_str}")
            definitions_html = "<div class='meaning'>" + " ".join(parts) + "</div>"

        examples_html = ""
        has_examples = False
        for m in meanings:
            if m.get('sentence'):
                has_examples = True
                sentence = markdown_to_html(m.get('sentence', ''))
                sentence = highlight_word(sentence, word)
                sentence_translation = markdown_to_html(m.get('sentence_translation', ''))
                examples_html += f"<div class='example-item'>{sentence}<span class='zh'>{sentence_translation}</span></div>"
        for item in details.get("phrases", []) + details.get("collocations", []):
            has_examples = True
            key = 'phrase' if 'phrase' in item else 'collocation'
            text = item.get(key, '')
            text = highlight_word(text, word)
            translation = item.get('translation', '')
            examples_html += f"<div class='example-item'>{text}<span class='zh'>{translation}</span></div>"
        if not has_examples:
            examples_html = "<span>无</span>"

        synonyms_html = ""
        synonyms = details.get("synonyms", [])
        if not synonyms:
            synonyms_html = "<span>无</span>"
        else:
            for s in synonyms:
                synonyms_html += f"<div class='example-item'>{s.get('synonym', '')} ({s.get('translation', '')})</div>"

        special_tips_raw = details.get("special_tips", "")
        special_tips_section_html = ""
        if special_tips_raw:
            special_tips_formatted = markdown_to_html(special_tips_raw.replace('\n', '<br>'))
            special_tips_section_html = f'<div class="note">特别提示: {special_tips_formatted}</div>'

        phonetic = details.get('phonetic', {})
        phonetic_uk_str, phonetic_us_str = "", ""
        uk_phonetic = phonetic.get('uk')
        us_phonetic = phonetic.get('us')
        if uk_phonetic and us_phonetic and uk_phonetic == us_phonetic:
            phonetic_uk_str = f"英美 /{uk_phonetic}/"
        else:
            if uk_phonetic:
                phonetic_uk_str = f"英 /{uk_phonetic}/"
            if us_phonetic:
                phonetic_us_str = f"美 /{us_phonetic}/"

        word_html = word_entry_template_str.replace("{long_word_class_placeholder}", long_word_class)
        word_html = word_html.replace("{word}", word)
        word_html = word_html.replace("{phonetic_uk}", phonetic_uk_str)
        word_html = word_html.replace("{phonetic_us}", phonetic_us_str)
        word_html = word_html.replace("{difficulty_stars}", difficulty_html)
        word_html = word_html.replace("{definitions_list}", definitions_html)
        word_html = word_html.replace("{examples_list}", examples_html)
        word_html = word_html.replace("{synonyms_list}", synonyms_html)
        word_html = word_html.replace("{special_tips_section}", special_tips_section_html)
        
        all_words_html += f"<div class='word-entry' id='{word}'>{word_html}</div>"

    toc_content_html = ""
    for word in sorted_words:
        toc_content_html += f"<a href='#{word}'>{word}</a><br>"

    word_book_html = template_content.replace("{cover_content}", cover_html)
    word_book_html = word_book_html.replace("{toc_content}", toc_content_html)
    word_book_html = word_book_html.replace("{words_content}", all_words_html)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(word_book_html)

def main():
    """主函数，集成了增量缓存和AI防死循环机制。"""
    load_config()
    
    words_cache = load_cache()
    print(f"已加载 {len(words_cache)} 个已缓存的单词。")

    words_to_process = parse_word_data(CONFIG.get("input_file", "example.json"))
    words_to_process = sorted(list(set(filter(None, words_to_process))))
    
    total_words = len(words_to_process)
    print(f"准备处理 {total_words} 个单词...")
    
    words_in_cache_set = set(words_cache.keys())
    new_words_to_fetch = [w for w in words_to_process if w not in words_in_cache_set]
    total_new_words = len(new_words_to_fetch)
    new_words_processed_count = 0
    
    api_call_times = []

    new_words_for_cache_save = 0
    with tqdm(total=total_words, desc="处理单词", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pbar:
        for word in words_to_process:
            if word in words_cache:
                tqdm.write(f"单词 '{word}' 已在缓存中，直接使用。")
                
                eta_str = "剩余: 00:00"
                if api_call_times:
                    avg_time = sum(api_call_times) / len(api_call_times)
                    remaining_new_words = total_new_words - new_words_processed_count
                    etr_seconds = remaining_new_words * avg_time
                    etr_formatted = time.strftime('%M:%S', time.gmtime(etr_seconds))
                    eta_str = f"剩余: ~{etr_formatted}"
                elif total_new_words > 0:
                    eta_str = "剩余: 计算中..."

                pbar.set_postfix_str(f"[{eta_str} | 缓存命中]")
                pbar.update(1)
            else:
                result_queue = Queue()
                
                def worker(w, q):
                    result = get_word_details_from_ai(w)
                    q.put(result)

                api_thread = threading.Thread(target=worker, args=(word, result_queue))
                api_thread.start()
                
                current_word_start_time = time.time()

                while api_thread.is_alive():
                    time_spent_on_current = time.time() - current_word_start_time
                    
                    eta_str = "剩余: 计算中..."
                    if api_call_times:
                        avg_time = sum(api_call_times) / len(api_call_times)
                        time_remaining_for_current = max(0, avg_time - time_spent_on_current)
                        remaining_new_words_after_this = total_new_words - new_words_processed_count - 1
                        etr_for_other_words = remaining_new_words_after_this * avg_time
                        total_etr_seconds = etr_for_other_words + time_remaining_for_current
                        etr_formatted = time.strftime('%M:%S', time.gmtime(total_etr_seconds))
                        eta_str = f"剩余: ~{etr_formatted}"
                    
                    pbar.set_postfix_str(f"[{eta_str} | 查询: {word}]")
                
                details = result_queue.get()
                
                time_taken = time.time() - current_word_start_time
                api_call_times.append(time_taken)

                if details:
                    words_cache[word] = details
                    new_words_for_cache_save += 1
                    new_words_processed_count += 1
                    
                    if new_words_for_cache_save % CONFIG.get("cache_save_interval", 5) == 0:
                        save_cache(words_cache)
                
                pbar.update(1)

    print("\n所有单词处理完毕，正在进行最终缓存保存...")
    save_cache(words_cache)

    final_words_details = {word: words_cache[word] for word in words_to_process if word in words_cache}

    if not final_words_details:
        print("未能获取任何单词的详细信息，程序退出。")
        return

    generate_html_from_template(final_words_details)
    print(f"\n成功生成单词书: {CONFIG.get('output_file', 'word_book.html')}")

if __name__ == "__main__":
    main()

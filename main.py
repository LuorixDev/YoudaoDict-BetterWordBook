import json
import os
import requests
import time
import re
from tqdm import tqdm
from queue import Queue
import threading
import hashlib
import random
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import io
import base64

# --- 全局配置 ---
CONFIG = {}

# --- 中文字体设置 ---
def set_chinese_font():
    """设置matplotlib以支持中文显示。"""
    try:
        # 优先使用黑体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
        print("Matplotlib已配置为使用'SimHei'字体显示中文。")
    except Exception as e:
        print(f"警告: 设置中文字体'SimHei'失败: {e}")
        print("可视化图中的中文可能无法正常显示。请确保您的系统已安装'SimHei'字体。")


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


def load_cache(cache_file_key):
    """根据指定的配置键加载缓存文件。"""
    cache_file = CONFIG.get(cache_file_key)
    if not cache_file or not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_cache(cache_data, cache_file_key):
    """根据指定的配置键保存缓存数据，并确保目录存在。"""
    cache_file = CONFIG.get(cache_file_key)
    if not cache_file:
        print(f"警告: 未在配置中找到缓存文件路径，键: {cache_file_key}")
        return
    
    # 确保缓存目录存在
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
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


def get_word_embeddings(words, embeddings_cache):
    """批量获取单词的词向量，并使用带进度条的批处理。"""
    print("正在检查并准备获取词向量...")
    embeddings = {}
    words_to_fetch = []
    
    for word in words:
        if word in embeddings_cache:
            embeddings[word] = np.array(embeddings_cache[word])
        else:
            words_to_fetch.append(word)

    if not words_to_fetch:
        print("所有词向量均已在缓存中。")
        return embeddings, False

    batch_size = CONFIG.get("embedding_batch_size", 50)
    cache_updated = False

    with tqdm(total=len(words_to_fetch), desc="获取词向量", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        for i in range(0, len(words_to_fetch), batch_size):
            batch_words = words_to_fetch[i:i + batch_size]
            
            headers = {
                "Authorization": f"Bearer {CONFIG.get('api_key')}",
                "Content-Type": "application/json"
            }
            data = {
                "input": batch_words,
                "model": CONFIG.get("embedding_model")
            }
            
            try:
                response = requests.post(CONFIG.get("embedding_api_endpoint"), headers=headers, json=data, timeout=120)
                response.raise_for_status()
                results = response.json()['data']
                
                for item in results:
                    word_in_batch = batch_words[item['index']]
                    embedding_vector = np.array(item['embedding'])
                    embeddings[word_in_batch] = embedding_vector
                    embeddings_cache[word_in_batch] = embedding_vector.tolist()
                
                cache_updated = True
            except requests.exceptions.RequestException as e:
                tqdm.write(f"获取一批词向量时出错: {e}")
            except (KeyError, IndexError) as e:
                tqdm.write(f"解析词向量API响应时出错: {e}")

            pbar.update(len(batch_words))
            
    if cache_updated:
        print(f"词向量获取完成，共处理 {len(words_to_fetch)} 个新单词。")
    
    return embeddings, cache_updated

def generate_3d_plot(embeddings):
    """使用自适应K-Means聚类和t-SNE降维，生成经过归一化、布局优化的科学可视化3D散点图。"""
    num_words = len(embeddings)
    if not embeddings or num_words < 3:
        print("词向量不足，无法生成3D图像。")
        return ""

    print("正在生成词向量3D可视化图像...")
    words = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()))
    
    # 1. 自适应寻找最佳聚类数量
    print("正在使用“肘部法则”自适应地寻找最佳聚类数量...")
    max_clusters_to_check = min(num_words, 15)
    inertia = []
    K = range(2, max_clusters_to_check + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(vectors)
        inertia.append(kmeans.inertia_)
    
    if len(K) > 0:
        kn = KneeLocator(K, inertia, curve='convex', direction='decreasing')
        num_clusters = kn.knee if kn.knee else max_clusters_to_check
        print(f"“肘部法则”发现的最佳聚类数量为: {num_clusters}")
    else:
        num_clusters = num_words

    # 2. K-Means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    print(f"已将单词聚类为 {num_clusters} 个语义簇。")

    # 3. t-SNE降维 (使用现代最佳实践优化)
    print("正在使用t-SNE进行降维（这可能需要一点时间）...")
    tsne = TSNE(n_components=3, random_state=42, 
                perplexity=min(30, num_words - 1), 
                learning_rate='auto',
                init='pca',
                max_iter=1000, 
                n_iter_without_progress=300)
    vectors_3d = tsne.fit_transform(vectors)
    print("t-SNE降维完成。")

    # 4. 归一化坐标以填充绘图空间 (收紧范围)
    print("正在归一化坐标以填充绘图空间...")
    scaler = MinMaxScaler(feature_range=(-9, 9))
    vectors_3d_scaled = scaler.fit_transform(vectors_3d)
    print("坐标归一化完成。")

    # 5. 生成3D散点图 (使用归一化坐标并优化渲染层级)
    fig = plt.figure(figsize=(16, 12), dpi=250)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
    
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        points = vectors_3d_scaled[cluster_indices]
        cluster_words = [words[j] for j in cluster_indices]
        
        # 绘制散点，设置较低的zorder
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=[colors[i]], label=f'Cluster {i+1}', s=15, alpha=0.7, zorder=1)
        
        # 减少标签数量以避免重叠
        num_labels = min(len(cluster_words), random.randint(3, 7))
        if num_labels > 0:
            sampled_indices = random.sample(range(len(cluster_words)), num_labels)
            for k in sampled_indices:
                # 绘制文本，设置较高的zorder以确保其在点之上
                ax.text(points[k, 0], points[k, 1], points[k, 2], cluster_words[k], 
                        fontsize=6, color='black', alpha=0.9, zorder=10)

    # 进一步放大坐标轴标签，并设置与归一化匹配的固定范围
    ax.set_xlabel('Semantic Axis 1', fontsize=16, labelpad=15)
    ax.set_ylabel('Semantic Axis 2', fontsize=16, labelpad=15)
    ax.set_zlabel('Semantic Axis 3', fontsize=16, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=12, pad=5)
    
    ax.set_xlim([-9.5, 9.5])
    ax.set_ylim([-9.5, 9.5])
    ax.set_zlim([-9.5, 9.5])
    
    ax.view_init(elev=25, azim=120)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # 5. 保存图像 (减少边距)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    print("3D可视化图像生成完毕。")
    return img_base64

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
        "response_format": {"type": "json_object"},
        "stream": True
    }

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(CONFIG.get(
                "api_endpoint"), headers=headers, json=data, timeout=120, stream=True)
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
                            delta = chunk.get('choices', [{}])[
                                0].get('delta', {})
                            content_part = delta.get('content')
                            if content_part:
                                json_string += content_part
                                buffer += content_part
                                while '\n' in buffer:
                                    line_to_print, buffer = buffer.split(
                                        '\n', 1)
                                    tqdm.write(line_to_print)

                                if len(json_string) > CONFIG.get("ai_response_max_length", 3500):
                                    tqdm.write(
                                        f"\n[错误] AI响应超长，可能已进入死循环。正在终止并重试...")
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
            tqdm.write(
                f"\n获取单词 '{word}' 详情时出错 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                tqdm.write(f"'{word}' 重试失败。")
        except json.JSONDecodeError as e:
            tqdm.write(f"\n解析AI返回的完整JSON时出错 '{word}': {e}")
            tqdm.write(f"原始JSON字符串: {json_string}")
            return None
    return None


def generate_html_from_template(words_details, word_embeddings):
    """使用模板和单词数据生成最终的HTML文件。"""
    print("正在生成HTML文件...")
    
    # --- 生成封面可视化 ---
    embedding_plot_base64 = generate_3d_plot(word_embeddings)

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
            word_list_hash = hashlib.sha256(
                sorted_word_list_str.encode('utf-8')).hexdigest()[:16]  # 取前16位

            # --- 注入元数据 ---
            cover_html = cover_template_content.replace(
                "{word_count}", str(word_count))
            cover_html = cover_html.replace(
                "{generation_time}", generation_time)
            cover_html = cover_html.replace("{word_list_hash}", word_list_hash)
            cover_html = cover_html.replace("{embedding_plot_base64}", embedding_plot_base64)

        except FileNotFoundError:
            print(f"警告: 封面模板文件 '{cover_template_file}' 未找到，将不生成封面。")

    try:
        with open(word_book_template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"错误: 单词书模板文件 '{word_book_template_file}' 未找到。")
        return

    word_entry_template_str = template_content[template_content.find(
        '<!-- word-entry-template-start -->'):template_content.find('<!-- word-entry-template-end -->')]

    sorted_words = sorted(words_details.keys())

    def markdown_to_html(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    def highlight_word(text, current_word):
        """在文本中高亮（加粗）指定的单词，不区分大小写，且匹配整个单词。"""
        pattern = re.compile(r'\b({})\b'.format(
            re.escape(current_word)), re.IGNORECASE)
        return pattern.sub(r'<strong>\1</strong>', text)

    all_words_html = ""
    for word in sorted_words:
        details = words_details.get(word)
        if not details:
            continue

        long_word_class = "long-word" if len(word) > 15 else ""
        difficulty = details.get("difficulty", 0)
        difficulty_html_left, difficulty_html_right = "", ""

        is_verb = any(m.get('part_of_speech', '').strip().lower(
        ).startswith('v') for m in details.get("meanings", []))

        difficulty_stars_html = ""
        if isinstance(difficulty, int) and 1 <= difficulty <= 5:
            filled_stars = '<span class="filled-star">★</span>' * difficulty
            empty_stars = '<span class="empty-star">★</span>' * \
                (5 - difficulty)
            difficulty_stars_html = f'<div class="difficulty">{filled_stars}{empty_stars}</div>'

        if not is_verb and not long_word_class:
            difficulty_html_right = difficulty_stars_html
        else:
            difficulty_html_left = difficulty_stars_html

        definitions_html = ""
        meanings = details.get("meanings", [])
        if not meanings:
            definitions_html = "<div class='meaning'>无</div>"
        else:
            from collections import defaultdict
            grouped_meanings = defaultdict(
                lambda: {'translations': [], 'extra_info': ''})

            for m in meanings:
                pos = m.get('part_of_speech', '')
                grouped_meanings[pos]['translations'].append(
                    markdown_to_html(m.get('translation', '')))

                if not grouped_meanings[pos]['extra_info']:
                    extra_html_parts = []

                    if 'countability' in m and m['countability']:
                        countability_map = {
                            'C': '[可数]', 'U': '[不可数]', 'C/U': '[可数/不可数]'}
                        mapped_val = countability_map.get(
                            m['countability'], '')
                        if mapped_val:
                            extra_html_parts.append(mapped_val)

                    if 'inflections' in m and m['inflections']:
                        inflection_parts = []
                        inf = m['inflections']
                        if inf.get('third_person'):
                            inflection_parts.append(
                                f"三单: {inf['third_person']}")
                        if inf.get('past'):
                            inflection_parts.append(f"过去式: {inf['past']}")
                        if inf.get('participle'):
                            inflection_parts.append(
                                f"过去分词: {inf['participle']}")
                        if inf.get('present_participle'):
                            inflection_parts.append(
                                f"现在分词: {inf['present_participle']}")
                        if inflection_parts:
                            extra_html_parts.append(
                                f"[{'; '.join(inflection_parts)}]")

                    if extra_html_parts:
                        grouped_meanings[pos][
                            'extra_info'] = f"<span class='grammar-info'>{' '.join(extra_html_parts)}</span>"

            parts = []
            for pos, data in grouped_meanings.items():
                translations_str = "; ".join(data['translations'])
                parts.append(
                    f"<span class='pos'>{pos}</span> {data['extra_info']} {translations_str}")
            definitions_html = "<div class='meaning'>" + \
                " ".join(parts) + "</div>"

        examples_html = ""
        has_examples = False
        for m in meanings:
            if m.get('sentence'):
                has_examples = True
                sentence = markdown_to_html(m.get('sentence', ''))
                sentence = highlight_word(sentence, word)
                sentence_translation = markdown_to_html(
                    m.get('sentence_translation', ''))
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
            special_tips_formatted = markdown_to_html(
                special_tips_raw.replace('\n', '<br>'))
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

        word_html = word_entry_template_str.replace(
            "{long_word_class_placeholder}", long_word_class)
        word_html = word_html.replace("{word}", word)
        word_html = word_html.replace("{phonetic_uk}", phonetic_uk_str)
        word_html = word_html.replace("{phonetic_us}", phonetic_us_str)
        word_html = word_html.replace(
            "{difficulty_stars_left}", difficulty_html_left)
        word_html = word_html.replace(
            "{difficulty_stars_right}", difficulty_html_right)
        word_html = word_html.replace("{definitions_list}", definitions_html)
        word_html = word_html.replace("{examples_list}", examples_html)
        word_html = word_html.replace("{synonyms_list}", synonyms_html)
        word_html = word_html.replace(
            "{special_tips_section}", special_tips_section_html)

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
    set_chinese_font()  # 初始化时设置中文字体
    load_config()

    word_details_cache = load_cache("word_details_cache_file")
    print(f"已加载 {len(word_details_cache)} 个已缓存的单词释义。")

    words_to_process = parse_word_data(
        CONFIG.get("input_file", "example.json"))
    words_to_process = sorted(list(set(filter(None, words_to_process))))

    total_words = len(words_to_process)
    print(f"准备处理 {total_words} 个单词...")

    words_in_cache_set = set(word_details_cache.keys())
    new_words_to_fetch = [
        w for w in words_to_process if w not in words_in_cache_set]
    total_new_words = len(new_words_to_fetch)
    new_words_processed_count = 0

    api_call_times = []

    new_words_for_cache_save = 0
    with tqdm(total=total_words, desc="处理单词", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pbar:
        for word in words_to_process:
            if word in word_details_cache:
                tqdm.write(f"单词 '{word}' 已在缓存中，直接使用。")

                eta_str = "剩余: 00:00"
                if api_call_times:
                    avg_time = sum(api_call_times) / len(api_call_times)
                    remaining_new_words = total_new_words - new_words_processed_count
                    etr_seconds = remaining_new_words * avg_time
                    etr_formatted = time.strftime(
                        '%M:%S', time.gmtime(etr_seconds))
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

                api_thread = threading.Thread(
                    target=worker, args=(word, result_queue))
                api_thread.start()

                current_word_start_time = time.time()

                while api_thread.is_alive():
                    time_spent_on_current = time.time() - current_word_start_time

                    eta_str = "剩余: 计算中..."
                    if api_call_times:
                        avg_time = sum(api_call_times) / len(api_call_times)
                        time_remaining_for_current = max(
                            0, avg_time - time_spent_on_current)
                        remaining_new_words_after_this = total_new_words - new_words_processed_count - 1
                        etr_for_other_words = remaining_new_words_after_this * avg_time
                        total_etr_seconds = etr_for_other_words + time_remaining_for_current
                        etr_formatted = time.strftime(
                            '%M:%S', time.gmtime(total_etr_seconds))
                        eta_str = f"剩余: ~{etr_formatted}"

                    pbar.set_postfix_str(f"[{eta_str} | 查询: {word}]")

                details = result_queue.get()

                time_taken = time.time() - current_word_start_time
                api_call_times.append(time_taken)

                if details:
                    word_details_cache[word] = details
                    new_words_for_cache_save += 1
                    new_words_processed_count += 1

                    if new_words_for_cache_save % CONFIG.get("cache_save_interval", 5) == 0:
                        save_cache(word_details_cache, "word_details_cache_file")

                pbar.update(1)

    print("\n所有单词处理完毕，正在进行最终缓存保存...")
    save_cache(word_details_cache, "word_details_cache_file")

    final_words_details = {word: word_details_cache[word]
                           for word in words_to_process if word in word_details_cache}

    if not final_words_details:
        print("未能获取任何单词的详细信息，程序退出。")
        return

    # --- 获取词向量并生成可视化 ---
    embeddings_cache = load_cache("embeddings_cache_file")
    print(f"已加载 {len(embeddings_cache)} 个已缓存的词向量。")
    
    final_word_list = list(final_words_details.keys())
    word_embeddings, embeddings_cache_updated = get_word_embeddings(final_word_list, embeddings_cache)
    
    if embeddings_cache_updated:
        print("词向量缓存已更新，正在保存...")
        save_cache(embeddings_cache, "embeddings_cache_file")

    generate_html_from_template(final_words_details, word_embeddings)
    print(f"\n成功生成单词书: {CONFIG.get('output_file', 'word_book.html')}")


if __name__ == "__main__":
    main()

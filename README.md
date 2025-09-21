# YoudaoDict-BetterWordBook: AI-Powered Custom Word Book Generator

## 简介

这是一个功能强大且高度可定制的 Python 工具，旨在将您从有道词典导出的简单单词列表，通过调用先进的AI模型（如 Qwen、GPT 等），自动扩充和排版，最终生成一本专业级的、可直接用于A4打印的精美单词书。

从单词的深度解析，到动词变形、名词可数性等语法细节，再到支持动态元数据的封面、自动页码的目录和媲美专业词典的排版，本项目致力于为您提供一个从数据到成品的、一站式的个人单词书定制解决方案。

![项目效果图](https://raw.githubusercontent.com/LuorixDev/YoudaoDict-BetterWordBook/main/word_book_preview.png)

## 核心功能亮点 (Features)

- **🤖 AI 驱动的内容扩充**:
  - **全面信息**: 自动获取每个单词的英美音标、多重释义、难度评级、近义词、常用短语和固定搭配。
  - **智能例句**: 为每个核心释义生成贴切的例句，并提供中文翻译。
  - **深度语法分析**: 智能判断词性，自动列出**动词的所有时态变形**（第三人称单数、过去式、过去分词、现在分词），并明确标注**名词的可数性**（可数/不可数）。
  - **特别提示**: AI 会根据单词特性，提供额外的、简短精炼的特别提示。

- **📖 专业级的排版与打印**:
  - **Paged.js 驱动**: 采用先进的 `paged.js` 库，确保在浏览器中预览的效果与最终打印到A4纸上的效果**完全一致**。
  - **动态封面**: 自动生成包含**单词总数、生成时间、单词列表唯一哈希值**的动态封面，每一本单词书都独一无二。
  - **智能目录**: 自动生成多栏、超链接的目录，并能**精确计算并显示每个单词所在的页码**。
  - **优化的页面布局**: 采用媲美专业词典的双栏布局，并将页边距和分栏间距压缩到物理打印的极限，实现信息密度最大化。

- **🚀 高效与稳定的工程化设计**:
  - **智能增量缓存**: 自动缓存所有AI返回的结果到 `cache.json`，再次运行时可直接从缓存读取，极大节省时间和API调用成本。
  - **健壮的数据清洗**: 自动清洗AI返回的音标数据，移除所有不规范字符（如 `/`, `·`, `-`），确保数据格式的绝对纯净与统一。
  - **多线程处理**: 使用多线程异步调用AI API，并通过 `tqdm` 进度条提供**实时、平滑的ETA（预计剩余时间）**，优化用户体验。
  - **流式输出与调试**: 支持AI流式输出，方便在控制台实时观察AI的思考过程，便于调试 Prompt。

- **🎨 极高的可定制性**:
  - **模板化设计**: 项目结构完全模块化，封面、单词书主体、单词卡片的样式和布局，均由 `templates/` 文件夹下的HTML模板文件控制，用户可随心所欲地进行修改。
  - **配置简单**: 所有核心参数（如API密钥、模型名称、文件路径等）均在 `config.json` 中统一管理，清晰明了。

## 项目结构

```
.
├── templates/
│   ├── cover_template.html       # 封面模板
│   └── word_book_template.html   # 单词书主模板 (目录 + 内容)
├── .gitignore
├── clean_cache.py                # 用于清理旧缓存数据的一次性脚本
├── config.example.json           # 配置文件示例
├── config.json                   # 您的本地配置文件 (需自行创建)
├── example.json                  # 有道词典导出的单词列表示例
├── LICENSE
├── main.py                       # 主程序入口
├── README.md                     # 本文档
├── requirements.txt              # Python 依赖
└── word_book.html                # 最终生成的单词书文件
```

## 安装与配置

1.  **克隆项目**:
    ```bash
    git clone https://github.com/LuorixDev/YoudaoDict-BetterWordBook.git
    cd YoudaoDict-BetterWordBook
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **创建并编辑配置文件**:
    - 将 `config.example.json` 复制一份，并重命名为 `config.json`。
    - 打开 `config.json`，填入您自己的配置，尤其是 `api_key`, `api_endpoint`, 和 `ai_model`。

## 使用方法

1.  **准备单词列表**: 将您从有道词典导出的单词本 `json` 文件（格式需与 `example.json` 类似）放入项目根目录。

2.  **更新配置**: 在 `config.json` 中，将 `input_file` 的值修改为您单词列表的文件名。

3.  **运行主程序**:
    ```bash
    python main.py
    ```
    程序将开始处理单词，您会看到一个带预估剩余时间的进度条。已缓存的单词会直接跳过，新单词会通过AI进行处理。

4.  **预览与打印**:
    - 程序运行完毕后，会生成一个 `word_book.html` 文件。
    - 在现代浏览器（推荐 Chrome 或 Edge）中打开这个 `word_book.html` 文件。
    - 按下 `Ctrl + P` 打开打印预览。
    - 在打印设置中，确保目标打印机为“**另存为PDF**”，纸张大小为 **A4**，边距设置为“**无**”或“**默认**”（`paged.js` 已处理好边距）。
    - 点击保存，即可得到一本排版完美的、专业级的PDF单词书。

## 定制化指南

本项目最大的乐趣之一就是高度的可定制性。您可以：

- **修改封面**: 直接编辑 `templates/cover_template.html`，修改标题、作者信息，甚至添加您自己的Logo图片。
- **修改单词卡片样式**: `templates/word_book_template.html` 中包含了单词卡片的所有HTML结构和CSS样式。您可以修改字体、颜色、边框、布局等一切视觉元素。
- **调整页面布局**: 在 `templates/word_book_template.html` 的 `<style>` 标签中，您可以调整 `@page` 规则，来改变页边距、分栏数等核心布局参数。

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

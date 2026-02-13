# BetterPrompt: 命令行中的 AI 提示词管理利器

BetterPrompt 是一个功能强大的命令行工具，旨在帮助开发者和内容创作者高效地管理、测试和调用 AI 模型的提示词（Prompts）。

## 🚀 功能特性

*   **提示词管理**: 轻松实现提示词的增、删、改、查。
*   **模板支持**: 创建和管理提示词模板，提高复用性。
*   **模型调用**: 直接在命令行中调用 AI 模型并查看结果。
*   **本地存储**: 使用 SQLite 数据库在本地安全地存储您的提示词和历史记录。
*   **灵活配置**: 支持自定义 AI 模型的 API 地址和密钥。

## 📦 安装

通过 RubyGems 安装 BetterPrompt:

```bash
gem install better_prompt
```

## ⚙️ 配置

首次运行时，BetterPrompt 会引导您完成初始化。您也可以手动执行：

```bash
bp --init
```

该命令会在您的用户主目录下创建 `.better_prompt` 文件夹，并包含 `config.yml` 配置文件和 `prompt.db` 数据库文件。

请修改 `config.yml` 文件，填入您的 AI 模型提供商的 `api_key` 和 `api_url`。

```yaml
# ~/.better_prompt/config.yml
api_key: "YOUR_API_KEY"
api_url: "https://api.openai.com/v1/chat/completions"
model: "gpt-3.5-turbo"
```

## 🛠️ 使用方法

BetterPrompt 提供了简洁的命令行接口来管理您的提示词。

### 主要命令

*   **列出所有提示词**
    ```bash
    bp --list
    # or
    bp -l
    ```

*   **显示一个提示词的详细内容**
    ```bash
    bp --show <ID>
    # or
    bp -s <ID>
    ```

*   **添加一个新的提示词**
    *BetterPrompt 会启动您默认的文本编辑器来编写提示词内容。*
    ```bash
    bp --add
    # or
    bp -a
    ```

*   **编辑一个已存在的提示词**
    *BetterPrompt 会启动您默认的文本编辑器来修改提示词内容。*
    ```bash
    bp --edit <ID>
    # or
    bp -e <ID>
    ```

*   **删除一个提示词**
    ```bash
    bp --delete <ID>
    # or
    bp -d <ID>
    ```

*   **调用 AI 模型**
    使用指定的提示词 ID 来调用在 `config.yml` 中配置的 AI 模型。
    ```bash
    bp --call <ID>
    ```

*   **提供反馈**
    打开一个链接，为项目提供反馈。
    ```bash
    bp --feedback
    ```

## 👨‍💻 开发

如果您想为 BetterPrompt 贡献代码，请遵循以下步骤：

1.  **Fork** 本项目。
2.  创建您的功能分支 (`git checkout -b feature/AmazingFeature`)。
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)。
4.  将代码推送到分支 (`git push origin feature/AmazingFeature`)。
5.  开启一个 **Pull Request**。

## 📄 许可证

本项目采用 MIT 许可证。详情请见 `LICENSE` 文件。

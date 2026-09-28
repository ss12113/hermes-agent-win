# Hermes Agent for Windows

在 Windows 上用一条命令安装 Hermes Agent（本仓库构建版，基于上游开源项目）。

## 一键安装

打开 **PowerShell**，粘贴并回车：

```powershell
iex (irm https://raw.githubusercontent.com/ss12113/hermes-agent-win/main/install.ps1)
```

不想用命令？下载本仓库的 [`install.cmd`](install.cmd) 双击运行，效果相同。

无需预装 Python / Node.js / Git —— 安装器会自动准备所需组件。

## 安装之后

- 程序与数据位于 `%LOCALAPPDATA%\hermes`
- **新开一个终端**，输入 `hermes` 即可使用
- 首次使用：运行 `hermes setup` 完成模型与 API Key 配置

## 更新

重新运行上面的安装命令即可；配置与数据都会保留。

## 卸载

1. 删除 `%LOCALAPPDATA%\hermes` 文件夹
2. 在「系统属性 → 环境变量」里，删除指向 `%LOCALAPPDATA%\hermes\bin` 的条目

## 常见问题

**支持哪些系统？** Windows 10 / 11（x64）。

**安装时需要联网吗？** 需要，安装过程会从 GitHub 与 PyPI 获取组件；网络受限时请先在系统层面配置代理，再重新运行安装命令。

**杀毒软件报警？** 安装器会下载一些官方二进制文件（如 uv 等），偶尔会被杀毒软件误报，加入信任列表即可。

**中途失败了怎么办？** 直接重新运行安装命令。安装器是幂等的，可以安全重试。

**装到哪里、会不会影响系统？** 只写入 `%LOCALAPPDATA%\hermes` 和用户级 PATH；不需要管理员权限，也不会动系统的 Python。

## 关于本仓库

- 本仓库是 [Nous Research Hermes Agent](https://github.com/NousResearch/hermes-agent)（MIT License）的 Windows 分发构建，包含本地补丁与增强。
- 多语言 README（含中文版 `README.zh-CN.md`）与上游文档随代码一并保留在本仓库中；上游 README 原件见 `README.upstream.md`。
- 本仓库不含任何密钥或个人信息；使用你自己的模型服务凭据即可。

---

## English

One-command Windows install:

```powershell
iex (irm https://raw.githubusercontent.com/ss12113/hermes-agent-win/main/install.ps1)
```

Installs to `%LOCALAPPDATA%\hermes`. No Python, Node.js, or Git required. This repository redistributes a locally patched build of [Hermes Agent](https://github.com/NousResearch/hermes-agent) (MIT License).

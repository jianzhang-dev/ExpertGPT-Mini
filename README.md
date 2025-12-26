
# 🚀 ExpertGPT-180M 技术文档与使用指南

ExpertGPT-180M是一个高性能、轻量化的中文小语言模型（SLM）。它采用了前沿的 **Parallel Expert Attenti架构，在仅 1.8 亿参数的体量下，通过 3.6 亿 Token 的充分训练和 300 万条高质量指令微调，实现了卓越的中文理解与指令遵循能力。并行专家注意力（并行专家注意力）

声明：本项目核心的三段代码（预训练、指令微调、交互推理）均由 ，展现了人类工程思路与 AI 编写能力的高效结合。AI 在人工深度指导与逻辑设计下生成

一、模型特性（Highlights）

混合注意力架构：

滑动窗口注意力 (Sliding Window)：大部分层采用局部窗口，降低计算复杂度。

全局注意力 (Global Attention)：在模型 1/3 和 2/3 深度处嵌入全局层，打破长文本限制，增强逻辑连贯性。

并行专家机制 (Parallel Expert Attention)：

不同于传统的 Top-K MoE，本项目采用并行专家投影与门控融合机制，在保持推理速度的同时，极大提升了参数的表达效率。

现代化组件：

RoPE (旋转位置编码)：支持更好的长文本外推。

RMSNorm & GeGLU：采用与 Llama 3/DeepSeek 同级别的平滑归一化与非线性激活函数。

工业级数据治理：

内置极致的中文敏感词过滤与政治关键词清洗逻辑，确保模型输出安全、合规。

二、 环境准备 (Quick Start)

对于初学者，请按照以下步骤配置运行环境。

1. 安装 Conda

Conda 是一个环境管理工具，可以防止不同项目间的依赖冲突。

前往 Miniconda 官网 下载并安装。

打开终端（Windows 为 Anaconda Prompt），创建一个新环境：

code
Bash
download
content_copy
expand_less
conda create -n expert_gpt python=3.10
conda activate expert_gpt
2. 安装 PyTorch 与 CUDA

模型支持 GPU 加速。如果你的显卡是 NVIDIA，请安装支持 CUDA 的版本。

访问 获取安装命令。通常为：PyTorch 官网

code
Bash
download
content_copy
expand_less
# 以 CUDA 11.8 为例
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3. 安装其他依赖
code
Bash
download
content_copy
expand_less
pip install transformers tqdm ijson
三、 三大核心模块声明

本项目由三部分核心代码构成，请确保它们处于同一工作目录下。

1. 预训练模块 (pretrain.py)

功能：从零启动模型。支持混合精度 (AMP) 训练和余弦学习率调度。

配置：默认 180M 参数（768 维，12 层，2 专家）。

数据：支持多文件 JSON/JSONL 并行加载。

2. 指令微调模块 (sft.py)

功能：加载预训练好的权重，在 300 万条 SFT 数据上进行全参数微调。

特色：引入了 流式解析，能够处理数 GB 级别的超大型问答数据集（如 Why 数据集），并带有严格的文本清洗逻辑。ijson

3. 交互式推理系统 (inference.py)

功能：终端对话界面。

增强体验：

思考动画：模拟 AI 思考过程。

打字机效果：流式显示输出文本。

多主题切换：通过 命令更改配色。/theme

动态调参：通过 指令实时修改 Temperature 和 Top-P。/params

四、 运行指引
第一步：模型训练（可选）

如果你已有数据集并存放在 目录下，运行预训练：data/raw/

code
Bash
download
content_copy
expand_less
python pretrain.py
第二步：模型微调

加载 权重，进行中文对话强化：expert_gpt_model_final.pth

code
Bash
download
content_copy
expand_less
python sft.py
第三步：交互对话

微调完成后，启动交互系统，与 ExpertGPT-180M 聊天：

code
Bash
download
content_copy
expand_less
python inference.py

进入系统后，输入 查看所有快捷指令。/help

五、 模型参数概览
参数项	配置
总参数量	~180,000,000 (180M)
隐藏层维度	768
层数	12
注意力头数	12
并行专家数	2
训练 Token 数	3.6 亿 (360M)
最大上下文长度	700 tokens
Tokenizer	Bert-Base-Chinese
六、 免责声明与技术说明

本模型仅供科研与学习使用，请勿用于非法用途。

所有代码逻辑均在人工指导下由 AI 辅助生成，涵盖了底层算子到上层应用的全链路实现。

ExpertGPT-180M 证明了：通过精准的架构选择与高质量的中文数据治理，小参数模型也能释放出巨大的能量。欢迎开启你的 SLM 探索之旅！ 🌟

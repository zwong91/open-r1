# hf-dataset-generator/hf-dataset-generator/README.md

# hf-dataset-generator

该项目旨在生成基于提示词的问题，并将其发布为数据集到Hugging Face。

## 文件结构

```
hf-dataset-generator
├── src
│   └── generate.py  # 生成问题的逻辑
├── requirements.txt  # 项目所需的Python库和依赖项
└── README.md         # 项目的文档
```

## 使用说明

1. **安装依赖项**

   在项目根目录下运行以下命令以安装所需的库：

   ```
   pip install -r requirements.txt
   ```

2. **生成问题**

   运行 `generate.py` 脚本以生成1000个问题。您可以根据需要修改提示词。

   ```
   python src/generate.py --prompt "您的提示词"
   ```

3. **发布数据集到Hugging Face**

   在生成问题后，您可以使用Hugging Face的API将数据集发布到平台。请确保您已设置好Hugging Face的API密钥。

   ```python
   from huggingface_hub import HfApi

   api = HfApi()
   api.upload_dataset("您的数据集名称", "数据集文件路径")
   ```

## 贡献

欢迎任何形式的贡献！请提交问题或拉取请求。

## 许可证

该项目采用MIT许可证。
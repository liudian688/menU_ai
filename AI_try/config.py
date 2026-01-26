import os

class Config:
    # 阿里云Agent配置
    ALIYUN_AGENT_ID = os.getenv('ALIYUN_AGENT_ID')
    ALIYUN_API_KEY = os.getenv('ALIYUN_API_KEY')
    ALIYUN_ENDPOINT = os.getenv('ALIYUN_ENDPOINT')

    # Agent参数配置
    AGENT_CONFIG = {
        "model": "qwen-plus-latest",  # 或你配置的模型
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": 2000
    }
import json
import dashscope
from dashscope import Generation
from typing import List, Dict, Any
import logging
from datetime import datetime
import os

from config import Config

# 配置日志系统，用于记录运行状态和错误信息
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # 创建日志记录器


class AliAgentService:
    def __init__(self):
        """初始化函数：设置API密钥和模型配置"""
        # 1. 设置阿里云API密钥（从配置文件读取）
        api_key = Config.ALIYUN_API_KEY or os.getenv('DASHSCOPE_API_KEY')
        if api_key:
            dashscope.api_key = api_key
        else:
            raise ValueError("API密钥未设置，请检查环境变量 ALIYUN_API_KEY 或 DASHSCOPE_API_KEY 是否已配置")
        
        # 2. 加载模型配置参数（温度、top_p等）
        self.config = Config.AGENT_CONFIG

    def integrate_questions(self, questions: List[Dict]) -> Dict[str, Any]:
        """
        整合题目的主函数

        工作流程：
        1. 接收题目数据
        2. 调用阿里云Agent
        3. 解析返回结果
        4. 返回整合后的题目

        参数：
            questions: 题目列表，每个题目是字典格式
                       [{"id":1, "question":"内容", "label":"python"}, ...]

        返回：
            整合后的题目数据，格式根据Agent返回决定
        """
        try:
            # 记录开始处理的日志
            logger.info(f"开始整合 {len(questions)} 道题目")

            # ⚡ 核心步骤1：将Python对象转为JSON字符串
            # 因为大语言模型通常处理文本输入，需要将数据结构转为字符串
            questions_json = json.dumps(questions, ensure_ascii=False)

            # ⚡ 核心步骤2：调用阿里云Agent API
            # Generation.call() 是阿里云提供的SDK调用方法
            response = Generation.call(
                model=self.config["model"],  # 使用的模型名称
                prompt=questions_json,  # 输入给Agent的文本内容
                temperature=self.config["temperature"],  # 随机性参数（0-1）
                top_p=self.config["top_p"],  # 采样范围参数
                max_tokens=self.config["max_tokens"],  # 最大输出token数
                result_format='message'  # 返回格式为消息格式
            )

            # ⚡ 核心步骤3：检查API调用是否成功
            if response.status_code == 200:
                # 从响应中提取Agent生成的内容
                result = response.output.choices[0].message.content
                logger.info("Agent调用成功")

                # ⚡ 核心步骤4：解析Agent返回的内容
                try:
                    # 尝试将返回的文本解析为JSON对象
                    # 因为Agent应该返回JSON格式的整合结果
                    parsed_result = json.loads(result)
                    print(1)

                    # 判断解析后的结果类型
                    if isinstance(parsed_result, list):
                        # 如果Agent直接返回数组，包装成标准格式
                        # 例如：[{"id": "1", "question": "...", "relationship": [...]}]
                        return {"integrated_questions": parsed_result}
                    else:
                        # 如果Agent返回的是字典（可能包含metadata），直接返回
                        # 例如：{"integrated_questions": [...], "metadata": {...}}
                        return parsed_result

                except json.JSONDecodeError:
                    # ⚠️ 如果解析失败（Agent返回的不是有效JSON）
                    # 这种情况发生在Agent输出格式不符合预期时
                    # 将原始内容包装成标准格式返回，避免服务中断
                    return {
                        "integrated_questions": [
                            {
                                "id": "1",  # 默认ID
                                "question": result,  # 使用Agent返回的原始文本作为题目
                                "relationship": []  # 空关系列表
                            }
                        ],
                        "raw_response": result  # 保存原始响应，便于调试
                    }
            else:
                # ⚠️ API调用失败的处理
                error_msg = f"Agent调用失败: {response.code}"
                logger.error(error_msg)  # 记录错误日志
                return {"error": error_msg}  # 返回错误信息

        except Exception as e:
            # ⚠️ 整体异常处理
            # 捕获所有未预期的异常，确保服务不会崩溃
            error_msg = f"处理异常: {str(e)}"
            logger.error(error_msg)  # 记录详细错误信息
            return {"error": error_msg}  # 返回错误信息
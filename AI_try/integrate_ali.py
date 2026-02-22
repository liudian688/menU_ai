"""
阿里云的agent调用代码 - 支持流式返回
"""
import json
from typing import Dict, Any, List, Generator
import dashscope
from dashscope import Application
import http.client
import io
import sys
import os
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

class AliAgentService:
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []  # 添加对话历史记录
    def process_questions_stream(self, questions: List[Dict]) -> Generator[str, None, None]:
        """流式处理问题的函数 - 支持逐字返回
        参数:
            messages: 对话消息列表，格式为 [{"role": "user/assistant", "content": "消息内容"}, ...]
            
        返回:
            Generator[str, None, None]: 生成器对象，逐字返回AI响应内容
            
        功能说明:
            - 使用阿里云Application API进行流式调用
            - 支持实时逐字显示AI响应
            - 自动处理API错误和异常情况
            - 维护会话上下文，确保对话连贯性
            
        异常处理:
            - 捕获API调用异常并返回错误信息
            - 处理网络错误和响应解析错误
        """
        # 开始流式处理问题
        try:
            # 使用流式API调用
            session_to_use = self.session_id

            responses = Application.call(
                app_id="3e112af8218a4660aee38c56080cf13f",
                prompt=json.dumps(questions, ensure_ascii=False),
                session_id=session_to_use,
                stream=True  # 启用流式模式
            )

            # 流式返回结果
            full_content = ""
            for response in responses:
                if response.status_code == 200:
                    try:
                        content = ""
                        
                        # 只处理有实际内容的响应块
                        if hasattr(response, 'output') and response.output:
                            output = response.output
                            
                            # 检查是否有text属性（基于调试确认的正确位置）
                            if hasattr(output, 'text') and output.text:
                                content = output.text
                        
                        # 如果当前响应块有内容，则处理
                        if content:
                            # 只返回新增的内容（避免重复）
                            new_content = content[len(full_content):]
                            if new_content:
                                yield new_content
                                full_content = content
                        
                        # 如果没有内容，静默跳过（这是正常的流式行为）
                            
                    except Exception as e:
                        # 只对严重错误显示错误信息，忽略属性访问相关的轻微错误
                        error_msg = str(e)
                        if "text" not in error_msg and "content" not in error_msg:
                            yield f"\n[处理响应时出错: {error_msg}]"
                        
                else:
                    yield f"\n[API错误: {response.status_code}]"
                    if hasattr(response, 'message'):
                        yield f"\n[错误信息: {response.message}]"
                    break
                    
        except Exception as e:
            yield f"\n[错误：{str(e)}]"

    def process_questions(self, questions: List[Dict]) -> str:
        """兼容旧版本的同步处理函数"""
        result = ""
        for chunk in self.process_questions_stream(questions):
            result += chunk
        return result

if __name__ == "__main__":
    # 创建系统实例
    system = AliAgentService()
    
    print("=== 学习评估系统 ===")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ['结束', 'quit', 'exit', 'q']:
                break
            
            # 构建包含历史记录的对话
            system.conversation_history.append({"role": "user", "content": user_input})
            
            # 处理用户输入
            print("\n正在处理您的请求...")
            
             # 将用户输入和历史记录转换为问题格式
            questions = {
                "current_input": user_input,
                "conversation_history": system.conversation_history
            }
            
            # 调用流式处理函数
            print("\n=== AI响应 ===")
            full_response = ""
            
            # 实时显示流式输出
            for chunk in system.process_questions_stream(questions):
                print(chunk, end="", flush=True)
                full_response += chunk
                
            print("\n" + "=" * 50)
            
            # 记录会话历史
            system.conversation_history.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n处理过程中出现错误: {e}")
            print("请重新输入或输入'退出'结束程序。")
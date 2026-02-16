"""
阿里云的agent调用代码
"""
import json
from typing import Dict, Any, List
import dashscope
from dashscope import Application
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

class AliAgentService:
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []  # 添加对话历史记录
    def process_questions(self, questions: List[Dict]) -> str:
        """处理问题的函数 - 直接返回完整响应
        参数:
            questions: 问题数据字典
            
        返回:
            str: 完整的AI响应内容
            
        功能说明:
            - 使用阿里云Application API进行调用
            - 直接返回完整响应内容
            - 自动处理API错误和异常情况
            - 维护会话上下文，确保对话连贯性
            
        异常处理:
            - 捕获API调用异常并返回错误信息
            - 处理网络错误和响应解析错误
        """
        # 开始处理问题
        try:
            # 使用普通API调用
            session_to_use = self.session_id

            response = Application.call(
                app_id="83155b5d536b4980a98fd733affae07b",
                prompt=json.dumps(questions, ensure_ascii=False),
                session_id=session_to_use,
            )

            # 处理响应
            if response.status_code == 200:
                if hasattr(response, 'output') and response.output:
                    output = response.output
                    
                    # 检查是否有text属性
                    if hasattr(output, 'text') and output.text:
                        return output.text
                    else:
                        return "[错误：响应中没有找到text内容]"
                else:
                    return "[错误：响应中没有output内容]"
            else:
                error_msg = f"[API错误: {response.status_code}]"
                if hasattr(response, 'message'):
                    error_msg += f"\n[错误信息: {response.message}]"
                return error_msg
                    
        except Exception as e:
            return f"[错误：{str(e)}]"

if __name__ == "__main__":
    # 创建系统实例
    system = AliAgentService()
    
    print("=== 学习规划系统 ===")
    
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
            
            # 调用普通处理函数
            print("\n=== AI响应 ===")
            
            # 直接获取完整响应
            full_response = system.process_questions(questions)
            print(full_response)
            
            print("\n" + "=" * 50)
            
            # 记录会话历史
            system.conversation_history.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n处理过程中出现错误: {e}")
            print("请重新输入或输入'退出'结束程序。")
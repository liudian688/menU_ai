"""
阿里云的agent调用代码
"""
import json
from typing import Dict, List
import dashscope
from dashscope import Application
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

class AliAgentService:
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    
    def __init__(self):
        # 用户会话存储字典，键为用户ID，值为用户会话数据
        self.user_sessions = {}
    
    def get_user_session(self, user_id: str):
        """获取或创建用户会话
        参数:
            user_id: 用户唯一标识符
        """
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'session_id': str(uuid.uuid4()),
                'conversation_history': []
            }
        return self.user_sessions[user_id]
    def process_questions(self, questions: List[Dict], user_id: str = "default") -> str:
        """处理问题的函数 
        参数:
            questions: 问题数据字典
            user_id: 用户唯一标识符，默认为"default"
            
        返回:
            str: 完整的AI响应内容
            
        功能说明:
            - 支持多用户访问，每个用户有独立的会话
            - 使用阿里云Application API进行普通调用
            - 直接返回完整响应内容
            - 自动处理API错误和异常情况
            - 维护用户独立的会话上下文
            
        异常处理:
            - 捕获API调用异常并返回错误信息
            - 处理网络错误和响应解析错误
        """
        # 开始处理问题
        try:
            # 获取用户会话
            user_session = self.get_user_session(user_id)
            session_to_use = user_session['session_id']

            response = Application.call(
                app_id="e186f1abdbf848bd9a5281341bc2f628",
                prompt=json.dumps(questions, ensure_ascii=False),
                session_id=session_to_use,
                stream=False  # 禁用流式模式
            )

            # 处理响应
            if response.status_code == 200:
                if hasattr(response, 'output') and response.output:
                    output = response.output
                    
                    # 检查是否有text属性
                    if hasattr(output, 'text') and output.text:
                        response_text = output.text
                        # 更新用户对话历史
                        user_session['conversation_history'].extend(questions.get('conversation_history', []))
                        user_session['conversation_history'].append({"role": "assistant", "content": response_text})
                        return response_text
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
    
    def clear_user_session(self, user_id: str):
        """清除指定用户的会话历史
        参数:
            user_id: 用户唯一标识符
        """
        if user_id in self.user_sessions:
            self.user_sessions[user_id]['conversation_history'] = []
            return f"用户 {user_id} 的会话历史已清除"
        else:
            return f"用户 {user_id} 不存在"
    
    def get_user_conversation_history(self, user_id: str):
        """获取指定用户的对话历史
        参数:
            user_id: 用户唯一标识符
        """
        if user_id in self.user_sessions:
            return self.user_sessions[user_id]['conversation_history']
        else:
            return []

if __name__ == "__main__":
    # 创建系统实例
    system = AliAgentService()
    
    # 默认用户ID
    current_user = "default_user"
    
    print("=== 学习规划系统 ===")
    print("输入 'clear' 清除当前会话历史")
    print("输入 'reset' 完全重置会话")
    print("输入 'history' 查看对话历史")
    print("输入 '结束/quit/exit/q' 退出程序")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ['结束', 'quit', 'exit', 'q']:
                break
            
            # 处理特殊命令
            if user_input.lower() == 'clear':
                print(system.clear_user_session(current_user))
                continue
            elif user_input.lower() == 'reset':
                print(system.clear_user_session(current_user, reset_session_id=True))
                continue
            elif user_input.lower() == 'history':
                history = system.get_user_conversation_history(current_user)
                if history:
                    print("\n=== 对话历史 ===")
                    for i, msg in enumerate(history, 1):
                        print(f"{i}. {msg['role']}: {msg['content'][:100]}...")
                else:
                    print("暂无对话历史")
                continue
            
            # 获取当前用户的对话历史
            user_history = system.get_user_conversation_history(current_user)
            
            # 构建问题格式（包含历史记录）
            questions = {
                "current_input": user_input,
                "conversation_history": user_history
            }
            
            # 处理用户输入
            print("\n正在处理您的请求...")
            
            # 调用普通处理函数（会自动更新历史）
            print("\n=== AI响应 ===")
            full_response = system.process_questions(questions, current_user)
            print(full_response)
            
            print("\n" + "=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n处理过程中出现错误: {e}")
            print("请重新输入或输入'退出'结束程序。")
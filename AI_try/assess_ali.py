"""
用于与用户对话(用户技能评估)的agent调用代码 - 支持流式返回
"""
import json
from typing import Dict, List, Generator
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
        self.messages = []  # 使用标准的 messages 格式记录会话历史
    def process_messages_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """流式处理消息的函数 - 支持逐字返回"""        
        # 开始流式处理问题
        try:
            # 使用流式API调用
            session_to_use = self.session_id

            responses = Application.call(
                app_id="e186f1abdbf848bd9a5281341bc2f628",
                prompt=json.dumps(messages, ensure_ascii=False),
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
                            
                    except Exception as e:
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
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息到会话历史"""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息到会话历史"""
        self.messages.append({"role": "assistant", "content": content})

    def get_conversation_summary(self) -> str:
        """获取会话摘要"""
        return f"当前会话共 {len(self.messages)} 条消息"

if __name__ == "__main__":
    # 创建系统实例
    system = AliAgentService()
    
    print("=== 技能评估系统 ===")
    print("请输入您的需求（输入'结束'或'quit'结束程序）：")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ['结束', 'quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            # 检查空输入
            if not user_input:
                print("输入不能为空，请重新输入。")
                continue
            
            # 添加用户消息到会话历史
            system.add_user_message(user_input)
            
            # 处理用户输入
            print("\n正在处理您的请求...")
            
            # 调用流式处理函数，直接传递 messages
            print("\n=== AI响应 ===")
            full_response = ""
            
            # 实时显示流式输出
            for chunk in system.process_messages_stream(system.messages):
                print(chunk, end="", flush=True)
                full_response += chunk
                
            print("\n" + "=" * 50)
            
            # 添加助手回复到会话历史
            system.add_assistant_message(full_response)
            
            # 显示会话摘要
            print(f"会话状态: {system.get_conversation_summary()}")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n处理过程中出现错误: {e}")
            print("请重新输入或输入'退出'结束程序。")
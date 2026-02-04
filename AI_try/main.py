"""
阿里云的agent调用代码
"""
import json
from typing import Dict, Any, List
import dashscope
from dashscope import Application
import http.client
import io
import sys
import os
import uuid

class AliAgentService:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []  # 添加对话历史记录
    def process_questions(self, questions: List[Dict]) -> str:
        """整合题目的主函数 - 添加完整验证"""
        if sys.platform == 'win32':
            # 1. 设置环境变量
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.environ['LC_ALL'] = 'C.UTF-8'

            # 2. 替换有问题的编码函数
            original_putheader = http.client.HTTPConnection.putheader

            def safe_putheader(self, header, *values):
                safe_values = []
                for v in values:
                    if isinstance(v, str):
                        # 过滤掉所有非ASCII字符
                        v = v.encode('ascii', 'ignore').decode('ascii')
                    safe_values.append(v)
                return original_putheader(self, header, *safe_values)

            http.client.HTTPConnection.putheader = safe_putheader

            # 3. 确保标准流使用UTF-8
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            else:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

            dashscope.api_key = "sk-774594d6824b4897a600606f8ca90f99"

        # 开始处理问题并进行整合
        try:
            # 1. 调用API
            session_to_use = self.session_id

            response = Application.call(
                app_id="cd2e28797bd04ee38c7be01743677d35",
                prompt=json.dumps(questions, ensure_ascii=False),
                session_id=session_to_use
            )

            # 2. 详细检查响应结构
            if response.status_code == 200:
                print("API调用成功，开始解析响应...")

                # 方法1：检查是否有output属性
                if not hasattr(response, 'output'):
                    print("响应缺少output属性")
                    return {"error": "响应缺少output属性", "raw_response": str(response)}

                output = response.output
                print(f"output类型: {type(output)}")

                # 方法2：尝试获取文本内容（不同版本SDK可能不同）
                content = None

                # 尝试方式A：output.text
                if hasattr(output, 'text') and output.text:
                    content = output.text
                    print("通过output.text获取内容")

                if content:
                    # 直接返回原始内容，不解析为JSON
                    return content
                else:
                    print("无法从响应中提取内容")
                    print(f"output对象: {output}")
                    print(f"output属性: {dir(output)}")
                    return f"错误：无法提取响应内容\n原始输出：{str(output)}"

            else:
                error_msg = f"API调用失败: {response.status_code}"
                if hasattr(response, 'message'):
                    error_msg += f" - {response.message}"
                print(f"{error_msg}")
                return f"错误：{error_msg}"

        except Exception as e:
            print(f"处理异常: {type(e).__name__}")
            print(f"详细错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"错误：处理异常: {str(e)}"

if __name__ == "__main__":
    # 创建系统实例
    system = AliAgentService()
    
    print("=== 阿里云智能题目处理系统 ===")
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
            
            # 构建包含历史记录的对话
            system.conversation_history.append({"role": "user", "content": user_input})
            
            # 处理用户输入
            print("\n正在处理您的请求...")
            
             # 将用户输入和历史记录转换为问题格式
            questions = {
                "current_input": user_input,
                "conversation_history": system.conversation_history
            }
            
            # 调用处理函数
            parsed_result = system.process_questions(questions)

            # 记录会话历史
            system.conversation_history.append({"role": "assistant", "content": parsed_result})
            
            # 输出结果
            print("\n=== 处理结果 ===")
            print(parsed_result)
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n处理过程中出现错误: {e}")
            print("请重新输入或输入'退出'结束程序。")
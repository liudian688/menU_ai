"""
baidu qianfan agent调用代码(未完成)
"""
import requests
import json
import os
import argparse

# 从环境变量获取Access Token
ACCESS_TOKEN = os.environ.get('QIANFAN_ACCESS_TOKEN')

if not ACCESS_TOKEN:
    raise ValueError("请设置环境变量 QIANFAN_ACCESS_TOKEN 存储百度千帆Access Token")


def get_user_input():
    """
    获取用户输入
    :return: 用户输入的字符串
    """
    return input("请输入您的问题或对话内容：")


def call_qianfan_api(app_id, user_input, access_token, stream=False):
    """
    调用百度千帆对话API
    """
    url = "https://qianfan.baidubce.com/v2/app/conversation/runs"
    
    payload = {
        "app_id": app_id,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    
    try:
        print(f"\n正在调用API，应用ID: {app_id}")
        print(f"用户输入: {user_input}")
        
        response = requests.post(url, headers=headers, json=payload, stream=stream)
        response.raise_for_status()
        
        if stream:
            print("\n=== 回复（流式）===")
            assistant_reply = ""
            
            for chunk in response.iter_lines():
                if chunk:
                    chunk_str = chunk.decode('utf-8')
                    
                    if chunk_str.startswith('data: '):
                        chunk_str = chunk_str[6:]
                    
                    try:
                        chunk_data = json.loads(chunk_str)
                        
                        if 'result' in chunk_data and isinstance(chunk_data['result'], list):
                            for item in chunk_data['result']:
                                if item.get('role') == 'assistant' and item.get('content'):
                                    new_content = item['content'][len(assistant_reply):]
                                    print(new_content, end='', flush=True)
                                    assistant_reply = item['content']
                    except json.JSONDecodeError:
                        print(f"\n[警告] 无法解析响应块: {chunk_str}")
            
            print("\n=== 对话结束 ===")
            return {"status": "success", "content": assistant_reply}
        else:
            result = response.json()
            print("\n=== 回复（完整）===")
            
            # 提取助手回复
            if 'result' in result and isinstance(result['result'], list):
                for item in result['result']:
                    if item.get('role') == 'assistant' and item.get('content'):
                        print(item.get('content'))
                        return result
            
            # 如果无法提取，打印完整响应
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return result
            
    except requests.exceptions.RequestException as e:
        print(f"\nAPI调用失败: {e}")
        if response:
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
        raise


def main():
    """
    主函数，处理命令行参数并执行API调用
    """
    parser = argparse.ArgumentParser(description='百度千帆对话API调用工具')
    parser.add_argument('--app-id', type=str, default='8ec5bf03-00f6-4d3c-8019-d254d6ea3f84', 
                      help='百度千帆应用ID')
    parser.add_argument('--stream', action='store_true', help='启用流式响应')
    parser.add_argument('--input', type=str, help='直接提供输入内容，不通过交互方式获取')
    
    args = parser.parse_args()
    
    try:
        # 获取用户输入
        if args.input:
            user_input = args.input
        else:
            user_input = get_user_input()
        
        # 调用API
        result = call_qianfan_api(args.app_id, user_input, ACCESS_TOKEN, args.stream)
        
        return result
        
    except Exception as e:
        print(f"\n程序执行失败: {e}")
        exit(1)


if __name__ == '__main__':
    main()
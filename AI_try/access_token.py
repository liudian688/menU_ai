"""
access_token获取代码
"""
import requests
import json


def main():
    # 确保这里的xxxxx已经替换成真实的API Key和Secret Key
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=ALTAKvuXvikGnwd7yRQJAky0Kh&client_secret=22eb9a0a7d674ff8ba310d6c332960a7"
    
    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    try:
        # 添加timeout参数，防止请求卡住
        response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
        
        # 打印状态码和响应
        print(f"HTTP状态码: {response.status_code}")
        print("响应内容:")
        print(response.text)
        
    except requests.exceptions.Timeout:
        print("错误: 请求超时（超过10秒）")
        print("可能是网络问题或URL错误")
    except requests.exceptions.ConnectionError:
        print("错误: 连接失败")
        print("请检查网络连接")
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}")


if __name__ == '__main__':
    print("开始获取access_token...")
    main()
    print("程序执行完毕")
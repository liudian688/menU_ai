from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import requests
import os
from dotenv import load_dotenv  # 添加这一行来加载.env文件
from main import AliAgentService
import dashscope

# 加载.env文件中的环境变量
load_dotenv()

DATABASE_URL='sqlite:///sqlite.db'

engine=create_engine(DATABASE_URL)
SessionLocal=sessionmaker(autocommit=False, autoflush=False, bind=engine)

base = declarative_base()

class questions(base):
    __tablename__ = 'questions'
    id=Column(Integer, primary_key=True, autoincrement=True)
    question=Column(String, nullable=False, index=True)
    label=Column(String, nullable=False, index=True)

questions_dict = [
    {"id":1,"question":"判断:Python中变量名可以包含下划线，且下划线可出现在任意位置","label":"python"},
    {"id":2,"question":"判断:布尔值True在数值运算中等价于1，False等价于0","label":"python"},
    {"id":3,"question":"判断:Python中print(10 // 3)的结果是3.333","label":"python"},
    {"id":4,"question":"判断:赋值运算符+=的作用是先加后赋值，例如a += 5等价于a = a + 5","label":"python"},
    {"id":5,"question":"判断:字符串是不可变数据类型，无法直接修改其中某一个字符","label":"python"},
    {"id":6,"question":"判断:元组中如果只有一个元素，必须在元素后加逗号，如(1,)，否则会被判定为普通数值","label":"python"},
    {"id":7,"question":"判断:if '' 中的条件会被判定为False","label":"python"},
    {"id":8,"question":"判断:while循环的循环体至少会执行一次","label":"python"},
    {"id":9,"question":"判断:函数调用时，参数传递的顺序必须和函数定义时的形参顺序完全一致","label":"python"},
    {"id":10,"question":"判断:Python中单行注释以#开头，多行注释只能用\"\"\"包裹","label":"python"},
    {"id":11,"question":"判断:字典的值可以是任意数据类型，包括列表、字典等可变类型","label":"python"},
    {"id":12,"question":"判断:continue语句可以终止整个循环，break语句仅跳过本次循环","label":"python"},
    {"id":13,"question":"判断:列表的索引从1开始，例如[1,2,3]中第一个元素的索引是1","label":"python"},
    {"id":14,"question":"判断:range(1, 5)生成的序列是1,2,3,4","label":"python"},
    {"id":15,"question":"判断:集合支持索引取值，例如s = {1,2,3}; print(s[0])可以正常输出1","label":"python"},
    {"id":16,"question":"判断:函数定义时，参数可以设置默认值，例如def func(a=10):","label":"python"},
    {"id":17,"question":"判断:Python中and运算符的规则是“一假即假”，or运算符是“一真即真”","label":"python"},
    {"id":18,"question":"判断:字符串的strip()方法只能去除首尾的空格，无法去除中间空格","label":"python"},
    {"id":19,"question":"判断:del语句删除列表元素后，列表的索引会自动重新排列","label":"python"},
    {"id":20,"question":"判断:全局变量在函数内部可以直接修改，无需任何关键字声明","label":"python"},
    {"id":21,"question":"判断:Python中is运算符判断的是两个对象的内存地址是否相同，==判断的是值是否相等","label":"python"},
    {"id":22,"question":"判断:列表的append()方法可以一次添加多个元素，例如lst.append(1,2,3)","label":"python"},
    {"id":23,"question":"判断:空字典、空列表、空字符串在布尔判断中都等价于False","label":"python"},
    {"id":24,"question":"判断:for i in 'python': 循环会执行6次，因为字符串有6个字符","label":"python"},
    {"id":25,"question":"判断:函数的返回值可以是多个，多个返回值会以元组的形式返回","label":"python"}
]

if __name__=='__main__':
    base.metadata.create_all(bind=engine)
    # input_label=input('Enter label: ')
    print("别催，等着！")
    input_label='python'
    db_session=SessionLocal()
    query_questions_all=db_session.query(questions).filter(questions.label==input_label).all()
    while query_questions_all == []:
        db_session=SessionLocal()
        for question in questions_dict:
            db_session.add(questions(**question))
        db_session.commit()
        db_session.close()
        query_questions_all=db_session.query(questions).filter(questions.label==input_label).all()

    questions_list = [{"id": q.id, "question": q.question, "label": q.label} for q in query_questions_all]
    
    # 确保API密钥被设置
    api_key = os.getenv('ALIYUN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if api_key:
        dashscope.api_key = api_key
    else:
        print("警告：未找到API密钥，请检查环境变量设置")
    
    # 创建 AliAgentService 实例并调用方法
    service = AliAgentService()
    result = service.integrate_questions(questions_list)
    if "error" not in result:
        # 如果API调用成功，提取相关信息
        choices = result.get("choices", [])
        extracted_data = []
    
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            extracted_data.append(content)
    
        # 打印列表形式的结果
        for item in extracted_data:
            print(item)
    else:
        print("API调用失败:", result.get("error"))
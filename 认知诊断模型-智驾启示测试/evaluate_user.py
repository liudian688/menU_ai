import torch
import json
import sys
from model import Net, KnowledgeBase


def load_config():
    '''加载配置文件'''  
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))
    return student_n, exer_n, knowledge_n


def evaluate_user_skill(user_answers, domain_knowledge_codes, model_epoch=5):
    '''
    评估用户在特定领域的技能水平
    :param user_answers: 用户答题记录，格式为 [{"exer_id": int, "score": int, "knowledge_code": [int, ...]}, ...]
    :param domain_knowledge_codes: 特定领域包含的知识点代码列表
    :param model_epoch: 使用的模型 epoch
    :return: 用户在该领域的技能水平评分（0-1之间），以及各知识点的掌握情况
    '''
    # 加载配置
    student_n, exer_n, knowledge_n = load_config()
    
    # 初始化知识基地和模型
    knowledge_base = KnowledgeBase(knowledge_n)
    net = Net(exer_n, knowledge_n)
    
    # 加载模型
    device = torch.device('cpu')
    load_snapshot(net, 'model/model_epoch' + str(model_epoch))
    net = net.to(device)
    net.eval()
    
    # 处理用户答题记录
    user_id = 99999  # 临时用户ID，不与现有用户冲突
    
    for answer in user_answers:
        exer_id = answer['exer_id']
        score = answer['score']
        knowledge_code = answer['knowledge_code']
        
        # 更新用户知识状态
        knowledge_base.update_user_state(user_id, exer_id, score, knowledge_code)
    
    # 获取用户知识状态
    user_state = knowledge_base.get_user_state(user_id).tolist()
    
    # 计算特定领域的技能水平
    domain_skills = []
    for kn_code in domain_knowledge_codes:
        if 1 <= kn_code <= knowledge_n:
            domain_skills.append(user_state[kn_code - 1])
    
    if domain_skills:
        domain_level = sum(domain_skills) / len(domain_skills)
    else:
        domain_level = 0.0
    
    # 构建各知识点的掌握情况字典
    knowledge_mastery = {}
    for i, mastery_level in enumerate(user_state):
        knowledge_mastery[i + 1] = mastery_level
    
    return domain_level, knowledge_mastery


def load_snapshot(model, filename):
    '''加载模型快照'''  
    f = open(filename, 'rb')
    state_dict = torch.load(f, map_location=lambda s, loc: s)
    # 过滤掉模型中不存在的参数（如旧版本的student_emb）
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)
    f.close()


def get_user_input():
    '''获取用户输入的答题记录和领域知识点'''  
    print("=== 输入用户答题记录 ===")
    print("请按照以下格式输入用户的答题记录，每行一条，输入空行结束：")
    print("格式：题目ID 得分 知识点代码1 知识点代码2 ...")
    print("示例：1 1 1 2")
    print("（得分1表示正确，0表示错误）")
    print()
    
    user_answers = []
    while True:
        line = input("输入答题记录：").strip()
        if not line:
            break
        try:
            parts = list(map(int, line.split()))
            if len(parts) < 3:
                print("输入格式错误，请重新输入。")
                continue
            exer_id = parts[0]
            score = parts[1]
            knowledge_code = parts[2:]
            user_answers.append({
                "exer_id": exer_id,
                "score": score,
                "knowledge_code": knowledge_code
            })
        except ValueError:
            print("输入格式错误，请重新输入。")
    
    print()
    print("=== 输入特定领域知识点 ===")
    print("请输入该领域包含的知识点代码，用空格分隔：")
    domain_input = input("输入知识点代码：").strip()
    domain_knowledge_codes = list(map(int, domain_input.split())) if domain_input else []
    
    return user_answers, domain_knowledge_codes

def main():
    '''主函数'''  
    print("=== 用户技能水平评估系统 ===")
    print("该系统基于NeuralCD模型，通过用户的答题记录评估其在特定领域的技能水平。")
    print()
    
    # 选择输入方式
    print("请选择输入方式：")
    print("1. 使用示例数据")
    print("2. 手动输入数据")
    choice = input("请输入选项（1/2）：").strip()
    print()
    
    if choice == '1':
        # 加载示例数据（如果存在）
        try:
            with open('data/test_set.json', 'r', encoding='utf8') as f:
                test_data = json.load(f)
            if test_data:
                # 使用第一个测试用户的数据作为示例
                example_user = test_data[0]
                user_answers = []
                for log in example_user['logs'][:3]:  # 只取前3条记录作为示例
                    user_answers.append({
                        "exer_id": log['exer_id'],
                        "score": log['score'],
                        "knowledge_code": log['knowledge_code']
                    })
                
                print("=== 使用示例数据 ===")
                print("示例用户答题记录：")
                print(json.dumps(user_answers, indent=2, ensure_ascii=False))
                print()
                
                # 假设前5个知识点为一个领域
                domain_knowledge_codes = list(range(1, 6))
                print(f"示例领域知识点：{domain_knowledge_codes}")
                print()
        except Exception as e:
            print(f"加载示例数据失败：{e}")
            print("将使用手动输入方式。")
            user_answers, domain_knowledge_codes = get_user_input()
    else:
        # 手动输入数据
        user_answers, domain_knowledge_codes = get_user_input()
    
    if not user_answers:
        print("没有输入答题记录，程序退出。")
        return
    
    if not domain_knowledge_codes:
        print("没有输入特定领域知识点，程序退出。")
        return
    
    # 评估用户技能水平
    domain_level, knowledge_mastery = evaluate_user_skill(user_answers, domain_knowledge_codes)
    
    print("=== 评估结果 ===")
    print(f"用户在该领域的技能水平：{domain_level:.4f} ({domain_level*100:.1f}%)")
    print()
    
    # 根据技能水平给出评价
    if domain_level >= 0.8:
        print("评价：您在该领域的技能水平非常高，掌握得很扎实！")
    elif domain_level >= 0.6:
        print("评价：您在该领域的技能水平良好，有一定的基础。")
    elif domain_level >= 0.4:
        print("评价：您在该领域的技能水平一般，需要加强练习。")
    elif domain_level >= 0.2:
        print("评价：您在该领域的技能水平较低，建议系统学习。")
    else:
        print("评价：您在该领域的技能水平很低，需要从头开始学习。")
    print()
    
    print("各知识点掌握情况：")
    for kn_code in domain_knowledge_codes:
        mastery = knowledge_mastery.get(kn_code, 0.0)
        print(f"知识点 {kn_code}: {mastery:.4f} ({mastery*100:.1f}%)")
    print()
    
    # 显示所有知识点的掌握情况
    non_zero_mastery = {kn: m for kn, m in knowledge_mastery.items() if m > 0}
    if non_zero_mastery:
        print("=== 所有知识点掌握情况 ===")
        print("（仅显示掌握度大于0的知识点）")
        for kn_code, mastery in non_zero_mastery.items():
            print(f"知识点 {kn_code}: {mastery:.4f} ({mastery*100:.1f}%)")
    else:
        print("=== 所有知识点掌握情况 ===")
        print("没有掌握任何知识点。")


if __name__ == '__main__':
    main()

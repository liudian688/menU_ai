import torch
import json
from model import Net, KnowledgeBase
import os
import sqlite3
from typing import List, Dict


def load_config():
    '''加载配置文件'''  
    config_path = os.path.join(os.path.dirname(__file__), 'config1.txt')
    with open(config_path) as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))
    return student_n, exer_n, knowledge_n


def evaluate_user_skill(user_answers, domain_knowledge_codes, model_epoch=5, user_id=None):
    '''
    使用认知诊断模型评估用户在特定领域的技能水平
    :param user_answers: 用户答题记录，格式为 [{"exer_id": int, "score": int, "knowledge_code": [int, ...]}, ...]
    :param domain_knowledge_codes: 特定领域包含的知识点代码列表
    :param model_epoch: 使用的模型 epoch
    :param user_id: 用户ID，如果为None则使用临时用户ID（99999）
    :return: 用户在该领域的技能水平评分（0-1之间），各知识点的掌握情况，以及诊断过程信息
    '''
    # 加载配置
    student_n, exer_n, knowledge_n = load_config()
    
    # 初始化知识基地和模型
    knowledge_base = KnowledgeBase(knowledge_n)
    net = Net(exer_n, knowledge_n)
    
    # 加载模型
    device = torch.device('cpu')
    model_path = os.path.join(os.path.dirname(__file__), 'model/model_epoch' + str(model_epoch))
    load_snapshot(net, model_path)
    net = net.to(device)
    net.eval()
    
    # 处理用户答题记录 - 使用模型进行认知诊断
    if user_id is None:
        user_id = 99999  # 临时用户ID
    else:
        user_id = int(user_id)  # 确保用户ID为整数
    
    # 记录诊断过程
    diagnostic_process = []
    
    for i, answer in enumerate(user_answers):
        exer_id = answer['exer_id']
        score = answer['score']
        knowledge_code = answer['knowledge_code']
        
        # 获取当前知识状态（用于诊断过程记录）
        current_state_before = knowledge_base.get_user_state(user_id).clone()
        
        # 使用模型进行认知诊断更新
        knowledge_base.update_user_state(user_id, exer_id, score, knowledge_code, net)
        
        # 获取更新后的知识状态
        current_state_after = knowledge_base.get_user_state(user_id).clone()
        
        # 记录诊断过程
        diagnostic_info = {
            'step': i + 1,
            'exer_id': exer_id,
            'score': score,
            'knowledge_code': knowledge_code,
            'state_before': current_state_before.tolist(),
            'state_after': current_state_after.tolist(),
            'state_changes': (current_state_after - current_state_before).tolist()
        }
        diagnostic_process.append(diagnostic_info)
    
    # 获取最终的用户知识状态
    final_user_state = knowledge_base.get_user_state(user_id).tolist()
    
    # 计算特定领域的技能水平
    domain_skills = []
    for kn_code in domain_knowledge_codes:
        if 1 <= kn_code <= knowledge_n:
            domain_skills.append(final_user_state[kn_code - 1])
    
    if domain_skills:
        domain_level = sum(domain_skills) / len(domain_skills)
    else:
        domain_level = 0.0
    
    # 构建各知识点的掌握情况字典
    knowledge_mastery = {}
    for i, mastery_level in enumerate(final_user_state):
        knowledge_mastery[i + 1] = mastery_level
    
    # 计算诊断过程的统计信息
    total_steps = len(diagnostic_process)
    if total_steps > 0:
        # 计算平均状态变化幅度
        avg_change = sum(
            sum(abs(change) for change in step_info['state_changes'])
            for step_info in diagnostic_process
        ) / (total_steps * knowledge_n)
        
        # 计算知识状态的稳定性（最后几步的变化幅度）
        last_steps = min(3, total_steps)
        if last_steps > 0:
            recent_changes = [
                sum(abs(change) for change in step_info['state_changes'])
                for step_info in diagnostic_process[-last_steps:]
            ]
            stability = 1.0 - (sum(recent_changes) / (last_steps * knowledge_n))
        else:
            stability = 1.0
    else:
        avg_change = 0.0
        stability = 1.0
    
    diagnostic_summary = {
        'total_steps': total_steps,
        'avg_state_change': avg_change,
        'stability': stability,
        'diagnostic_process': diagnostic_process
    }
    
    return domain_level, knowledge_mastery, diagnostic_summary

def load_snapshot(model, filename):
    '''加载模型快照'''  
    f = open(filename, 'rb')
    state_dict = torch.load(f, map_location=lambda s, loc: s)
    # 过滤掉模型中不存在的参数（如旧版本的student_emb）
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)
    f.close()


class DatabaseConnector:
    """数据库连接器，用于从数据库读取用户答题记录"""
    
    def __init__(self, db_path: str = None):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径，如果为None则使用默认路径
        """
        if db_path is None:
            # 使用默认数据库路径
            db_path = os.path.join(os.path.dirname(__file__), 'data', 'user_responses.db')
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """连接到数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False
    
    def get_user_responses(self, user_id: int) -> List[Dict]:
        """
        从数据库获取指定用户的所有答题记录
        
        Args:
            user_id: 用户ID
            
        Returns:
            user_responses: 用户答题记录列表
                        格式: [{'exer_id': int, 'score': int, 'knowledge_code': [int, ...]}, ...]
        """
        if self.conn is None:
            if not self.connect():
                return []
        
        # 查询用户答题记录
        try:
            query = """
            SELECT exer_id, score, knowledge_codes 
            FROM user_responses 
            WHERE user_id = ?
            ORDER BY response_time
            """
            self.cursor.execute(query, (user_id,))
            rows = self.cursor.fetchall()
            
            user_responses = []
            for row in rows:
                exer_id, score, knowledge_codes_str = row
                # 将知识点代码字符串转换为列表
                knowledge_codes = list(map(int, knowledge_codes_str.split(','))) if knowledge_codes_str else []
                
                user_responses.append({
                    "exer_id": exer_id,
                    "score": int(score),  # 确保score为整数
                    "knowledge_code": knowledge_codes
                })
            
            return user_responses
            
        except Exception as e:
            print(f"查询用户答题记录失败: {e}")
            return []
    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.conn = None
        self.cursor = None
    
    def __del__(self):
        """析构函数，确保连接被关闭"""
        self.close()
        return []
    
    def get_available_users(self) -> List[int]:
        """获取数据库中存在的用户ID列表"""
        if not hasattr(self, 'conn'):
            if not self.connect():
                return []
        
        try:
            query = "SELECT DISTINCT user_id FROM user_responses ORDER BY user_id"
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            print(f"获取用户列表失败: {e}")
            return []

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

def get_user_from_database():
    """从数据库获取用户答题记录"""
    print("=== 从数据库读取用户数据 ===")
    
    # 创建数据库连接器
    db_connector = DatabaseConnector()
    
    # 获取可用的用户列表
    available_users = db_connector.get_available_users()
    
    if not available_users:
        print("数据库中暂无用户数据，请先添加用户答题记录。")
        print("将使用手动输入方式。")
        return get_user_input()
    
    # 显示可用的用户
    print("可用的用户ID：", available_users)
    
    # 获取用户输入
    while True:
        try:
            user_id = int(input("请输入要评估的用户ID：").strip())
            if user_id in available_users:
                break
            else:
                print(f"用户ID {user_id} 不存在，请重新输入。")
        except ValueError:
            print("请输入有效的数字用户ID。")
    
    # 从数据库获取用户答题记录
    user_answers = db_connector.get_user_responses(user_id)
    
    if not user_answers:
        print(f"用户 {user_id} 没有答题记录。")
        print("将使用手动输入方式。")
        return get_user_input()
    
    print(f"成功获取用户 {user_id} 的 {len(user_answers)} 条答题记录")
    
    # 获取领域知识点
    print()
    print("=== 输入特定领域知识点 ===")
    print("请输入该领域包含的知识点代码，用空格分隔：")
    domain_input = input("输入知识点代码：").strip()
    domain_knowledge_codes = list(map(int, domain_input.split())) if domain_input else []
    
    db_connector.close()
    
    return user_answers, domain_knowledge_codes


def main():
    '''主函数'''  
    print("=== 用户技能水平评估系统 ===")
    print("该系统基于NeuralCD模型，通过用户的答题记录评估其在特定领域的技能水平。")
    print()
    
    # 选择输入方式
    print("请选择输入方式：")
    print("1. 使用示例数据")
    print("2. 从数据库读取数据")
    print("3. 手动输入数据")
    choice = input("请输入选项（1/2/3）：").strip()
    print()
    
    if choice == '1':
        # 加载示例数据（如果存在）
        try:
            test_set_path = os.path.join(os.path.dirname(__file__), 'data', 'test_set.json')
            with open(test_set_path, 'r', encoding='utf8') as f:
                test_data = json.load(f)
            if test_data:
                # 使用第一个测试用户的数据作为示例
                example_user = test_data[0]
                user_answers = []
                for log in example_user['logs'][:3]:  # 只取前3条记录作为示例
                    user_answers.append({
                        "exer_id": log['exer_id'],
                        "score": log['score']
                    })
                
                print("=== 使用示例数据 ===")
                print("示例用户答题记录（只包含题目和得分）：")
                print(json.dumps(user_answers, indent=2, ensure_ascii=False))
                print()
                
                # 提示用户需要输入知识点信息
                print("注意：答题记录已生成，但需要您输入特定领域的知识点代码。")
                print("请输入该领域包含的知识点代码，用空格分隔：")
                domain_input = input("输入知识点代码：").strip()
                domain_knowledge_codes = list(map(int, domain_input.split())) if domain_input else []
                
                if not domain_knowledge_codes:
                    print("未输入知识点，将使用默认知识点1-5")
                    domain_knowledge_codes = list(range(1, 6))
                
                print(f"领域知识点：{domain_knowledge_codes}")
                print()
        except Exception as e:
            print(f"加载示例数据失败：{e}")
            print("将使用手动输入方式。")
            user_answers, domain_knowledge_codes = get_user_input()
    elif choice == '2':
        # 从数据库读取数据
        user_answers, domain_knowledge_codes = get_user_from_database()
    else:
        # 手动输入数据
        user_answers, domain_knowledge_codes = get_user_input()
    
    if not user_answers:
        print("没有输入答题记录，程序退出。")
        return
    
    if not domain_knowledge_codes:
        print("没有输入特定领域知识点，程序退出。")
        return
    
    # 评估用户技能水平（使用认知诊断模型）
    domain_level, knowledge_mastery, diagnostic_summary = evaluate_user_skill(user_answers, domain_knowledge_codes)
    
    print("=== 认知诊断评估结果 ===")
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
    
    # 显示诊断过程统计信息
    print("=== 诊断过程统计 ===")
    print(f"诊断步骤数：{diagnostic_summary['total_steps']}")
    print(f"平均状态变化幅度：{diagnostic_summary['avg_state_change']:.4f}")
    print(f"诊断稳定性：{diagnostic_summary['stability']:.4f}")
    
    if diagnostic_summary['stability'] > 0.9:
        print("诊断结果稳定性：高（诊断过程收敛良好）")
    elif diagnostic_summary['stability'] > 0.7:
        print("诊断结果稳定性：中等")
    else:
        print("诊断结果稳定性：较低（建议增加更多答题记录）")
    print()
    
    print("=== 各知识点掌握情况 ===")
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
    print()
    
    # 显示详细的诊断过程（可选）
    show_detailed_diagnosis = input("是否显示详细诊断过程？(y/n): ").strip().lower()
    if show_detailed_diagnosis == 'y':
        print("\n=== 详细诊断过程 ===")
        for step_info in diagnostic_summary['diagnostic_process']:
            print(f"\n步骤 {step_info['step']}:")
            print(f"  题目ID: {step_info['exer_id']}, 得分: {step_info['score']}")
            print(f"  涉及知识点: {step_info['knowledge_code']}")
            
            # 显示主要变化的知识点
            changes = step_info['state_changes']
            significant_changes = [(i+1, change) for i, change in enumerate(changes) if abs(change) > 0.01]
            
            if significant_changes:
                print("  主要状态变化:")
                for kn_code, change in significant_changes:
                    if change > 0:
                        print(f"    知识点 {kn_code}: +{change:.4f} (增强)")
                    else:
                        print(f"    知识点 {kn_code}: {change:.4f} (减弱)")
            else:
                print("  状态变化较小（<0.01）")


if __name__ == '__main__':
    main()
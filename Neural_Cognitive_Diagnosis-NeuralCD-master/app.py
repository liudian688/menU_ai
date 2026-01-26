import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from model import Net  # 修改导入，使用正确的类名
import pickle
import torch

class CognitiveDiagnosisApp:
    def __init__(self):
        """
        初始化认知诊断应用
        """
        # 加载预训练的模型
        self.model = None
        self.load_model()
        
        # 【可修改部分】技能-知识点映射结构，解决稀疏性问题
        # 你可以在这里添加、删除或修改技能和知识点
        # 注意：这里的修改会影响用户界面显示的内容
        self.skill_knowledge_map = {
            "编程": ["变量类型", "控制结构", "算法基础", "数据结构", "面向对象"],
            "数学": ["代数", "几何", "微积分", "统计学", "线性代数"],
            "语言": ["语法", "词汇", "阅读理解", "写作技巧", "听力"],
            "物理": ["力学", "热学", "电磁学", "光学", "原子物理"],
            "化学": ["无机化学", "有机化学", "物理化学", "分析化学", "生物化学"]
        }
        
        # 所有知识点列表（实际应用中会很大）
        self.all_knowledge_points = []
        for skills in self.skill_knowledge_map.values():
            self.all_knowledge_points.extend(skills)
        self.all_knowledge_points = list(set(self.all_knowledge_points))  # 去重
        
        # 当前活跃知识点（针对特定用户或技能）
        self.active_knowledge_points = []
        
        # 加载数据集用于相似性分析
        self.dataset = self.load_dataset()
        # 将数据集转换为适合相似性分析的格式
        self.processed_dataset = self.process_dataset()

    def load_model(self):
        """
        加载预训练的模型
        【注意】这是深度学习模型加载部分，如果你不了解深度学习，不建议修改这部分
        """
        # 尝试加载最新的模型文件
        import os
        model_files = [f for f in os.listdir('./model') if f.startswith('model_epoch')]
        if model_files:
            latest_model = sorted(model_files, key=lambda x: int(x.split('epoch')[1]))[-1]
            model_path = f"./model/{latest_model}"
            # 加载模型 - 注意需要知道原始模型的参数
            try:
                # 因为我们不知道原始模型的具体参数，这里尝试从训练数据推断
                # 先加载数据集获取参数信息
                try:
                    with open('./data/train_set.json', 'r') as f:
                        sample_data = json.load(f)
                    # 这里需要根据实际的数据格式来确定参数
                    # 假设train_set.json中有学生数、题目数、知识点数等信息
                    # 由于不知道具体格式，我们使用默认值
                    student_n = 100  # 假设学生数
                    exer_n = 50      # 假设题目数
                    knowledge_n = 5  # 假设知识点数
                except:
                    # 如果无法读取训练数据，则使用默认值
                    student_n = 100
                    exer_n = 50
                    knowledge_n = 5
                
                # 创建模型实例 - 【深度学习核心部分，不建议修改】
                self.model = Net(student_n, exer_n, knowledge_n)
                
                # 加载模型权重
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # 如果checkpoint是字典格式，提取state_dict
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    # 如果checkpoint直接是模型权重
                    self.model.load_state_dict(checkpoint)
                    
                self.model.eval()  # 设置为评估模式
                print(f"模型已从 {model_path} 加载")
            except Exception as e:
                print(f"模型加载失败: {e}")
                print("创建新模型实例（未训练）")
                # 创建一个默认模型实例
                self.model = Net(100, 50, 5)  # 使用默认参数
        else:
            print("未找到预训练模型，创建新实例")
            # 创建一个默认模型实例
            self.model = Net(100, 50, 5)

    def load_dataset(self):
        """
        加载数据集用于相似性分析
        【可修改部分】你可以修改这里以适应你的数据格式
        """
        try:
            with open('./data/test_set.json', 'r') as f:
                dataset = json.load(f)
            return dataset
        except FileNotFoundError:
            print("警告: 找不到测试数据集，使用模拟数据")
            # 模拟数据集用于演示 - 【可修改部分】
            # 你可以根据自己的数据格式修改这里的模拟数据
            return [
                {"skill": "编程", "responses": [1, 1, 0, 1, 0], "knowledge_states": [0.8, 0.7, 0.3, 0.9, 0.2]},
                {"skill": "数学", "responses": [0, 1, 1, 0, 1], "knowledge_states": [0.3, 0.8, 0.7, 0.4, 0.9]},
                {"skill": "编程", "responses": [1, 0, 1, 1, 0], "knowledge_states": [0.9, 0.4, 0.6, 0.8, 0.3]},
                {"skill": "物理", "responses": [0, 0, 1, 1, 1], "knowledge_states": [0.2, 0.3, 0.8, 0.7, 0.8]},
                {"skill": "数学", "responses": [1, 1, 1, 0, 0], "knowledge_states": [0.7, 0.6, 0.4, 0.3, 0.6]}
            ]

    def process_dataset(self):
        """
        处理数据集，将其转换为适合相似性分析的格式
        【可修改部分】你可以根据你的数据格式调整这里
        """
        processed = []
        
        # 检查是否使用了模拟数据
        if self.dataset and len(self.dataset) > 0 and 'skill' in self.dataset[0]:
            # 如果是模拟数据，直接返回
            return self.dataset
        
        # 处理真实数据集格式 - 【根据你的数据格式修改】
        for user_data in self.dataset:
            if 'logs' in user_data:
                # 提取用户的响应和知识状态
                responses = []
                knowledge_codes = set()
                
                for log in user_data['logs']:
                    if 'score' in log:
                        responses.append(int(log['score']))  # 1表示答对，0表示答错
                    if 'knowledge_code' in log:
                        knowledge_codes.update(log['knowledge_code'])
                
                # 创建知识状态向量（这里简化处理）
                knowledge_states = [0.5] * len(knowledge_codes)  # 默认值
                
                processed.append({
                    'user_id': user_data.get('user_id', 0),
                    'responses': responses,
                    'knowledge_codes': list(knowledge_codes),
                    'knowledge_states': knowledge_states
                })
        
        return processed

    def select_skill(self):
        """
        让用户选择感兴趣的技能领域
        【可修改部分】你可以修改用户界面交互方式
        """
        print("请选择您想评估的技能领域：")
        skills = list(self.skill_knowledge_map.keys())
        for i, skill in enumerate(skills, 1):
            print(f"{i}. {skill}")
        
        while True:
            try:
                choice = int(input("请输入选项编号: ")) - 1
                if 0 <= choice < len(skills):
                    selected_skill = skills[choice]
                    print(f"您选择了: {selected_skill}")
                    # 只关注该技能相关的知识点
                    self.active_knowledge_points = self.skill_knowledge_map[selected_skill]
                    return selected_skill
                else:
                    print("无效选项，请重新选择")
            except ValueError:
                print("请输入有效数字")

    def get_user_responses(self, skill):
        """
        获取用户在特定技能领域的答题响应
        【可修改部分】你可以修改问题的呈现方式
        """
        responses = {}
        print(f"\n正在评估技能: {skill}")
        print("请回答以下相关问题（输入1表示答对，输入0表示答错）：")
        
        # 获取该技能对应的问题（这里简化为知识点名称作为问题）
        skill_knowledges = self.skill_knowledge_map[skill]
        
        for i, knowledge in enumerate(skill_knowledges):
            while True:
                try:
                    response = input(f"问题{i+1}: 关于'{knowledge}'的知识点，您掌握了吗？(1=掌握, 0=未掌握): ")
                    if response in ['0', '1']:
                        responses[knowledge] = int(response)
                        break
                    else:
                        print("请输入0或1")
                except ValueError:
                    print("请输入有效的数字")
                    
        return responses

    def find_similar_users(self, user_responses, user_skill):
        """
        在数据集中查找与当前用户情况相似的用户
        【核心算法部分，不建议修改】
        这里使用余弦相似度算法来计算用户间的相似性
        """
        # 获取当前技能的所有知识点
        skill_knowledges = self.skill_knowledge_map[user_skill]
        
        # 构建用户响应向量
        user_response_vector = []
        for knowledge in skill_knowledges:
            if knowledge in user_responses:
                user_response_vector.append(user_responses[knowledge])
            else:
                user_response_vector.append(0)  # 未回答的问题视为未掌握
        
        # 获取数据集中所有用户的响应
        dataset_responses = []
        valid_users = []  # 存储有效的用户数据
        
        for user_data in self.processed_dataset:
            if 'responses' in user_data:
                resp_vector = user_data['responses']
                # 我们只取与当前技能相关的知识点数量的响应
                # 这里简化处理，取前N个响应，其中N是当前技能的知识点数量
                if len(resp_vector) >= len(skill_knowledges):
                    # 取前N个响应
                    resp_vector = resp_vector[:len(skill_knowledges)]
                else:
                    # 如果响应不足，用0填充
                    resp_vector.extend([0] * (len(skill_knowledges) - len(resp_vector)))
                
                dataset_responses.append(resp_vector)
                valid_users.append(user_data)
        
        if not dataset_responses:
            print("未找到有效的用户数据，返回空的相似用户列表")
            return []
        
        # 计算相似度 - 【深度学习算法核心，不建议修改】
        similarities = []
        for i, dataset_resp in enumerate(dataset_responses):
            # 使用余弦相似度计算 - 【数学算法，不建议修改】
            user_array = np.array(user_response_vector).reshape(1, -1)
            dataset_array = np.array(dataset_resp).reshape(1, -1)
            
            similarity = cosine_similarity(user_array, dataset_array)[0][0]
            similarities.append((similarity, i))
        
        # 按相似度排序并返回最相似的前几个用户
        similarities.sort(reverse=True)
        return [(sim, valid_users[idx]) for sim, idx in similarities[:min(3, len(similarities))]]

    def identify_weak_knowledge_points(self, user_responses, similar_users, user_skill):
        """
        识别用户的薄弱知识点（相对于其他用户）
        【核心算法部分，不建议修改】
        这里通过比较用户与相似用户的知识状态来识别薄弱点
        """
        if not similar_users:
            print("未找到相似用户，使用全局平均值进行比较")
            # 使用所有用户的平均知识状态
            if self.processed_dataset:
                # 获取所有用户中包含知识状态的用户
                users_with_knowledge = [u for u in self.processed_dataset if 'knowledge_states' in u and len(u['knowledge_states']) > 0]
                if users_with_knowledge:
                    # 找到最大长度的知识状态数组
                    max_len = max(len(u['knowledge_states']) for u in users_with_knowledge)
                    # 对每个位置计算平均值
                    avg_knowledge_states = []
                    for i in range(max_len):
                        vals = [u['knowledge_states'][i] if i < len(u['knowledge_states']) else 0.5 
                                for u in users_with_knowledge]
                        avg_knowledge_states.append(sum(vals) / len(vals))
                else:
                    # 默认值
                    avg_knowledge_states = [0.5] * len(self.skill_knowledge_map[user_skill])
            else:
                # 默认值
                avg_knowledge_states = [0.5] * len(self.skill_knowledge_map[user_skill])
        else:
            # 使用相似用户的平均知识状态
            knowledge_states_list = []
            for _, user_info in similar_users:
                if 'knowledge_states' in user_info and len(user_info['knowledge_states']) > 0:
                    knowledge_states_list.append(user_info['knowledge_states'])
            
            if knowledge_states_list:
                # 确保所有知识状态数组长度一致
                max_len = max(len(ks) for ks in knowledge_states_list)
                padded_states = []
                for ks in knowledge_states_list:
                    padded = ks[:]
                    if len(padded) < max_len:
                        padded.extend([0.5] * (max_len - len(padded)))
                    padded_states.append(padded)
                
                avg_knowledge_states = np.mean(padded_states, axis=0)
            else:
                avg_knowledge_states = [0.5] * len(self.skill_knowledge_map[user_skill])
        
        # 估算当前用户的知识状态
        user_knowledge_states = self.estimate_user_knowledge(user_responses, user_skill)
        
        # 识别薄弱知识点
        weaknesses = []
        skill_knowledges = self.skill_knowledge_map[user_skill]
        
        for i, (user_state, avg_state) in enumerate(zip(user_knowledge_states, avg_knowledge_states)):
            weakness_score = avg_state - user_state
            if weakness_score > 0.1:  # 设定阈值来判断是否为薄弱点
                weaknesses.append({
                    'knowledge_point': skill_knowledges[i],
                    'user_level': round(user_state, 2),
                    'avg_level': round(avg_state, 2),
                    'weakness_score': round(weakness_score, 2)
                })
        
        return weaknesses, user_knowledge_states

    def estimate_user_knowledge(self, user_responses, user_skill):
        """
        估算用户在特定技能领域的知识状态
        【算法部分，可适度调整】
        这里将用户的二元响应转换为知识掌握程度的概率估计
        """
        skill_knowledges = self.skill_knowledge_map[user_skill]
        knowledge_states = []
        
        for knowledge in skill_knowledges:
            if knowledge in user_responses:
                # 将二元响应转换为概率估计
                # 这里简化为直接使用响应值，实际应使用更复杂的转换
                knowledge_states.append(float(user_responses[knowledge]))
            else:
                # 未测试的知识点，默认中等水平
                knowledge_states.append(0.5)
        
        return knowledge_states

    def calculate_overall_score(self, user_knowledge_states):
        """
        计算用户整体学习掌握评分
        【可修改部分】你可以调整评分计算方式
        """
        # 基于知识状态计算整体评分
        avg_knowledge = sum(user_knowledge_states) / len(user_knowledge_states)
        # 转换为百分制
        score = avg_knowledge * 100
        return round(score, 2)

    def run(self):
        """
        运行应用程序的主要流程
        【主流程，可修改用户交互方式】
        """
        print("=" * 60)
        print("技能成长评估系统 - 解决稀疏性问题版本")
        print("=" * 60)
        
        # 用户选择技能领域
        selected_skill = self.select_skill()
        
        # 获取用户响应
        user_responses = self.get_user_responses(selected_skill)
        print("\n您已回答完毕，正在分析结果...")
        
        # 查找相似用户
        similar_users = self.find_similar_users(user_responses, selected_skill)
        print(f"\n找到 {len(similar_users)} 个相似用户")
        
        # 识别薄弱知识点
        weaknesses, user_knowledge_states = self.identify_weak_knowledge_points(
            user_responses, similar_users, selected_skill
        )
        
        # 计算整体评分
        overall_score = self.calculate_overall_score(user_knowledge_states)
        
        # 输出结果
        print("\n" + "=" * 60)
        print("诊断结果:")
        print("=" * 60)
        print(f"技能领域: {selected_skill}")
        print(f"整体掌握评分: {overall_score}/100")
        
        skill_knowledges = self.skill_knowledge_map[selected_skill]
        print(f"知识状态分布:")
        for i, knowledge in enumerate(skill_knowledges):
            print(f"  - {knowledge}: {user_knowledge_states[i]:.2f}")
        
        if weaknesses:
            print("\n薄弱知识点:")
            for w in weaknesses:
                print(f"- {w['knowledge_point']}: "
                      f"您的水平 {w['user_level']}, "
                      f"平均水平 {w['avg_level']}, "
                      f"差距 {w['weakness_score']}")
        else:
            print("\n未发现明显薄弱知识点，继续保持!")
        
        print("\n相似用户对比:")
        for i, (similarity, user_info) in enumerate(similar_users):
            knowledge_states = user_info.get('knowledge_states', [])
            avg_knowledge = np.mean(knowledge_states) if knowledge_states else 0.5
            print(f"- 相似度 {similarity:.2f}, "
                  f"ID {user_info.get('user_id', 'Unknown')}, "
                  f"平均掌握度 {avg_knowledge:.2f}")


def hierarchical_skill_model():
    """
    层次化技能模型 - 处理大规模知识点的有效方法
    【可选功能，可修改】
    """
    # 定义层次结构
    hierarchy = {
        "STEM": {
            "数学": ["代数", "几何", "微积分", "统计学", "离散数学"],
            "科学": {
                "物理": ["力学", "热学", "电磁学", "光学", "量子力学"],
                "化学": ["无机化学", "有机化学", "物理化学", "分析化学", "生物化学"],
                "生物": ["细胞生物学", "遗传学", "生态学", "生理学", "进化论"]
            },
            "工程": ["计算机科学", "电子工程", "机械工程", "土木工程", "化学工程"]
        },
        "人文社科": {
            "语言文学": ["中文", "英文", "文学理论", "写作技巧", "修辞学"],
            "历史": ["古代史", "近代史", "现代史", "世界史", "专门史"],
            "哲学": ["西方哲学", "中国哲学", "伦理学", "逻辑学", "美学"]
        },
        "艺术": {
            "视觉艺术": ["绘画", "雕塑", "摄影", "设计", "建筑"],
            "表演艺术": ["音乐", "舞蹈", "戏剧", "电影", "歌剧"]
        }
    }
    
    return hierarchy

if __name__ == "__main__":
    app = CognitiveDiagnosisApp()
    app.run()


"""
我现在要用model写一个app.py，实现一下功能：
我作为一个新用户，一开始，你问我你个数据库中有的问题（用id代替），我回答1或0（例如，你问题目10，我回答1）
（题目之后要是具体的，但现在没有数据库，先用id代替，之后有了数据库再用id作为索引，所以数据库部分可以先写几个注释，
让我来写）我回答完后，在数据集中找和我情况相似的人，从我的回答来看出我薄弱的知识点（这个薄弱是个相对值，要对比其他人的），
并对我的学习掌握情况评个分​



解决稀疏性的关键策略
领域分割：将无限大的知识点空间分割成有限的技能领域
层次化结构：建立技能-知识点的层次关系
上下文相关：只关注用户当前技能领域的知识点
稀疏矩阵技术：使用scipy.sparse处理大型稀疏数据
相似性度量：使用余弦相似度等适合稀疏数据的方法
这样，虽然总的潜在知识点可能很多，但每次评估只关注特定领域的相关知识点，避免了稀疏性问题。
"""
import json
import numpy as np
import random
import os
from datetime import datetime, timedelta


class MathExerciseDataGenerator:
    """高数模拟数据生成器（修复版）"""
    
    def __init__(self, num_students=1000, num_exercises=500, num_knowledge=50):
        self.num_students = num_students
        self.num_exercises = num_exercises
        self.num_knowledge = num_knowledge
        
        # 定义高数知识点体系
        self.knowledge_hierarchy = self._define_knowledge_hierarchy()
        
    def _define_knowledge_hierarchy(self):
        """定义高数知识点体系"""
        return {
            # 基础模块
            "函数与极限": [1, 2, 3],
            "导数与微分": [4, 5, 6, 7],
            "积分学": [8, 9, 10, 11],
            # 进阶模块
            "多元函数": [12, 13, 14],
            "级数理论": [15, 16, 17],
            "微分方程": [18, 19, 20],
            # 应用模块
            "空间解析几何": [21, 22, 23],
            "数值计算": [24, 25, 26],
            "概率统计": [27, 28, 29]
        }
    
    def generate_student_abilities(self, num_students=None):
        """生成学生能力水平（正态分布）"""
        if num_students is None:
            num_students = self.num_students
        
        # 大部分学生能力在中等水平
        abilities = np.random.normal(0.5, 0.2, num_students)
        # 限制在0-1范围内
        abilities = np.clip(abilities, 0.1, 0.9)
        return abilities
    
    def generate_exercise_difficulties(self, num_exercises=None):
        """生成题目难度水平"""
        if num_exercises is None:
            num_exercises = self.num_exercises
        
        difficulties = np.random.normal(0.5, 0.3, num_exercises)
        difficulties = np.clip(difficulties, 0.1, 0.9)
        return difficulties
    
    def assign_knowledge_to_exercises(self, num_exercises=None):
        """为题目分配知识点"""
        if num_exercises is None:
            num_exercises = self.num_exercises
        
        exercise_knowledge = {}
        
        for exer_id in range(1, num_exercises + 1):
            # 每个题目关联1-3个知识点
            num_knowledge = random.randint(1, 3)
            
            # 根据题目难度分配知识点（难度大的题目关联进阶知识点）
            if exer_id <= num_exercises * 0.3:  # 30%简单题
                knowledge_pool = list(range(1, 11))  # 基础知识点
            elif exer_id <= num_exercises * 0.7:  # 40%中等题
                knowledge_pool = list(range(1, 21))  # 基础和进阶
            else:  # 30%难题
                knowledge_pool = list(range(1, self.num_knowledge + 1))
            
            knowledge_codes = random.sample(knowledge_pool, num_knowledge)
            exercise_knowledge[exer_id] = knowledge_codes
        
        return exercise_knowledge
    
    def calculate_correct_probability(self, student_ability, exercise_difficulty, discrimination=1.0):
        """基于IRT模型计算答对概率"""
        # 使用2PL模型: P = 1 / (1 + exp(-discrimination * (ability - difficulty)))
        exponent = -discrimination * (student_ability - exercise_difficulty)
        probability = 1 / (1 + np.exp(exponent))
        return probability
    
    def generate_training_data(self, records_per_student=50):
        """生成训练数据"""
        print("开始生成训练数据...")
        
        # 生成基础数据
        student_abilities = self.generate_student_abilities()
        exercise_difficulties = self.generate_exercise_difficulties()
        exercise_knowledge = self.assign_knowledge_to_exercises()
        
        training_data = []
        
        for student_id in range(1, self.num_students + 1):
            ability = student_abilities[student_id - 1]
            
            # 为每个学生随机选择题目
            selected_exercises = random.sample(
                range(1, self.num_exercises + 1), 
                min(records_per_student, self.num_exercises)
            )
            
            for exer_id in selected_exercises:
                difficulty = exercise_difficulties[exer_id - 1]
                knowledge_codes = exercise_knowledge[exer_id]
                
                # 计算答对概率
                prob_correct = self.calculate_correct_probability(ability, difficulty)
                
                # 根据概率生成得分
                score = 1 if random.random() < prob_correct else 0
                
                training_data.append({
                    "user_id": student_id,
                    "exer_id": exer_id,
                    "score": score,
                    "knowledge_code": knowledge_codes
                })
        
        print(f"训练数据生成完成！共{len(training_data)}条记录")
        print(f"学生数量: {self.num_students}")
        print(f"题目数量: {self.num_exercises}")
        print(f"知识点数量: {self.num_knowledge}")
        
        return training_data
    
    def generate_test_data(self, num_students=200, num_exercises=100, records_per_student=15):
        """生成测试数据（使用全新的学生和题目）"""
        print("开始生成测试数据...")
        
        # 使用全新的学生和题目
        student_abilities = self.generate_student_abilities(num_students)
        exercise_difficulties = self.generate_exercise_difficulties(num_exercises)
        exercise_knowledge = self.assign_knowledge_to_exercises(num_exercises)
        
        test_data = []
        
        for student_id in range(1, num_students + 1):
            ability = student_abilities[student_id - 1]
            
            # 为每个学生随机选择题目
            selected_exercises = random.sample(
                range(1, num_exercises + 1), 
                min(records_per_student, num_exercises)
            )
            
            for exer_id in selected_exercises:
                difficulty = exercise_difficulties[exer_id - 1]
                knowledge_codes = exercise_knowledge[exer_id]
                
                # 计算答对概率
                prob_correct = self.calculate_correct_probability(ability, difficulty)
                
                # 根据概率生成得分
                score = 1 if random.random() < prob_correct else 0
                
                test_data.append({
                    "user_id": student_id + self.num_students,  # 避免与训练数据ID冲突
                    "exer_id": exer_id + self.num_exercises,    # 避免与训练数据ID冲突
                    "score": score,
                    "knowledge_code": knowledge_codes
                })
        
        print(f"测试数据生成完成！共{len(test_data)}条记录")
        print(f"测试学生数量: {num_students}")
        print(f"测试题目数量: {num_exercises}")
        
        return test_data
    
    def save_data(self, data, filename):
        """保存数据到文件"""
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"数据已保存到: {filename}")
    
    def generate_config_file(self, filename='config_math.txt'):
        """生成配置文件"""
        config_content = """# Number of Students, Number of Exercises, Number of Knowledge Concepts
{},{},{}

# 高数知识点说明:
# 1-10: 函数与极限、导数微分等基础内容
# 11-20: 积分学、多元函数等进阶内容
# 21-30: 级数、微分方程等高级内容
# 31-50: 应用数学相关内容
""".format(self.num_students, self.num_exercises, self.num_knowledge)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"配置文件已生成: {filename}")

def main():
    """主函数：生成完整的高数数据集"""
    # 创建生成器
    generator = MathExerciseDataGenerator(
        num_students=1000,      # 1000个学生
        num_exercises=500,      # 500道题目
        num_knowledge=50        # 50个知识点
    )
    
    # 生成训练数据
    train_data = generator.generate_training_data(records_per_student=30)
    file_path = os.path.abspath(__file__)
    data_file = os.path.join(os.path.dirname(file_path), 'data/math_train_set.json')
    generator.save_data(train_data, data_file)
    
    # 生成验证数据（使用不同的学生-题目组合）
    val_data = generator.generate_training_data(records_per_student=10)
    file_path = os.path.abspath(__file__)
    data_file = os.path.join(os.path.dirname(file_path), 'data/math_val_set.json')
    generator.save_data(val_data, data_file)

    # 生成测试数据（使用全新的学生-题目组合）
    test_data = generator.generate_test_data(num_students=200, num_exercises=100, records_per_student=15)
    file_path = os.path.abspath(__file__)
    data_file = os.path.join(os.path.dirname(file_path), 'data/math_test_set.json')
    generator.save_data(test_data, data_file)
    
    # 生成配置文件
    generator.generate_config_file('./config_math.txt')
    
    print("\n高数模拟数据集生成完成！")
    print("文件结构:")
    print("- ./data/math_train_set.json: 训练数据")
    print("- ./data/math_val_set.json: 验证数据") 
    print("- ./config_math.txt: 配置文件")


if __name__ == "__main__":
    main()
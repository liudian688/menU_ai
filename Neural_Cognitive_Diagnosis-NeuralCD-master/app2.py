"""
认知诊断模型评估引擎
负责核心的模型推理、相似性分析和薄弱点识别
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json


class CognitiveDiagnosisEngine:
    def __init__(self, model_path=None, student_n=4163, exer_n=17746, knowledge_n=123):
        """
        初始化认知诊断引擎
        
        Args:
            model_path: 预训练模型路径
            student_n: 学生总数
            exer_n: 题目总数  
            knowledge_n: 知识点总数
        """
        self.student_n = student_n
        self.exer_n = exer_n
        self.knowledge_n = knowledge_n
        
        # 加载模型
        self.model = None
        if model_path:
            self.load_model(model_path)
        else:
            # 创建默认模型实例
            from model import Net
            self.model = Net(student_n, exer_n, knowledge_n)
    
    def load_model(self, model_path):
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
        """
        from model import Net
        try:
            # 创建模型实例
            self.model = Net(self.student_n, self.exer_n, self.knowledge_n)
            
            # 加载模型权重
            device = torch.device("cpu")  # 使用CPU加载
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # 如果checkpoint是字典格式，提取state_dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # 假设整个checkpoint就是state_dict
                    self.model.load_state_dict(checkpoint)
            else:
                # 如果checkpoint直接是模型权重
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()  # 设置为评估模式
            print(f"模型已从 {model_path} 加载")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def build_knowledge_vector(self, user_responses):
        """
        根据用户答题记录构建知识向量
        
        Args:
            user_responses: 用户答题记录列表
                          格式: [{'exer_id': int, 'score': float, 'knowledge_code': [int, ...]}, ...]
        
        Returns:
            knowledge_vector: 知识点掌握程度向量 [float, ...]
        """
        knowledge_vector = [0.0] * self.knowledge_n
        
        # 累积每个知识点的得分
        knowledge_scores = {}
        knowledge_counts = {}
        
        for response in user_responses:
            score = response['score']
            knowledge_codes = response.get('knowledge_code', [])
            
            if not knowledge_codes:
                # 如果没有指定知识点，随机分配一些知识点（模拟）
                import random
                knowledge_codes = random.sample(range(1, self.knowledge_n+1), min(3, self.knowledge_n))
            
            for code in knowledge_codes:
                if 1 <= code <= self.knowledge_n:  # 确保在范围内
                    code_idx = code - 1  # 转换为0基索引
                    if code_idx not in knowledge_scores:
                        knowledge_scores[code_idx] = 0
                        knowledge_counts[code_idx] = 0
                    knowledge_scores[code_idx] += score
                    knowledge_counts[code_idx] += 1
        
        # 计算每个知识点的平均得分
        for i in range(self.knowledge_n):
            if i in knowledge_counts and knowledge_counts[i] > 0:
                knowledge_vector[i] = knowledge_scores[i] / knowledge_counts[i]
        
        return knowledge_vector

    def find_similar_users(self, user_knowledge_vector, dataset):
        """
        在数据集中查找与当前用户情况相似的用户
        
        Args:
            user_knowledge_vector: 用户知识向量
            dataset: 数据集，格式: [{'user_id': int, 'logs': [...], ...}, ...]
        
        Returns:
            similar_users: 相似用户列表 [(similarity, user_data), ...]
        """
        similarities = []
        
        for user_data in dataset:
            # 构建用户的知识向量
            user_logs = user_data.get('logs', [])
            dataset_knowledge_vector = self._build_knowledge_vector_from_logs(user_logs)
            
            # 计算余弦相似度
            user_array = np.array(user_knowledge_vector).reshape(1, -1)
            dataset_array = np.array(dataset_knowledge_vector).reshape(1, -1)
            
            try:
                similarity = cosine_similarity(user_array, dataset_array)[0][0]
                similarities.append((similarity, user_data))
            except:
                # 如果计算相似度失败，跳过该用户
                continue
        
        # 按相似度排序并返回最相似的前几个用户
        similarities.sort(reverse=True)
        return similarities[:min(10, len(similarities))]  # 返回最多10个最相似用户

    def _build_knowledge_vector_from_logs(self, logs):
        """
        从日志构建知识向量
        
        Args:
            logs: 用户答题日志
            
        Returns:
            knowledge_vector: 知识点掌握程度向量
        """
        knowledge_vector = [0.0] * self.knowledge_n
        
        # 累积每个知识点的得分
        knowledge_scores = {}
        knowledge_counts = {}
        
        for log in logs:
            score = log['score']
            knowledge_codes = log['knowledge_code']
            
            for code in knowledge_codes:
                if 1 <= code <= self.knowledge_n:  # 确保在范围内
                    code_idx = code - 1  # 转换为0基索引
                    if code_idx not in knowledge_scores:
                        knowledge_scores[code_idx] = 0
                        knowledge_counts[code_idx] = 0
                    knowledge_scores[code_idx] += score
                    knowledge_counts[code_idx] += 1
        
        # 计算每个知识点的平均得分
        for i in range(self.knowledge_n):
            if i in knowledge_counts and knowledge_counts[i] > 0:
                knowledge_vector[i] = knowledge_scores[i] / knowledge_counts[i]
        
        return knowledge_vector

    def identify_weak_knowledge_points(self, user_knowledge_vector, similar_users):
        """
        识别用户的薄弱知识点（相对于相似用户）
        
        Args:
            user_knowledge_vector: 用户知识向量
            similar_users: 相似用户列表 [(similarity, user_data), ...]
        
        Returns:
            weaknesses: 薄弱点列表 [{'knowledge_point': str, 'user_level': float, ...}, ...]
            user_knowledge_vector: 更新后的用户知识向量
        """
        if not similar_users:
            print("未找到相似用户，使用全局平均值进行比较")
            # 使用整个数据集的平均知识状态
            avg_knowledge_states = [0.5] * self.knowledge_n  # 默认值
        else:
            # 使用相似用户的平均知识状态
            knowledge_vectors = []
            for _, user_data in similar_users:
                user_logs = user_data.get('logs', [])
                knowledge_vector = self._build_knowledge_vector_from_logs(user_logs)
                knowledge_vectors.append(knowledge_vector)
            
            if knowledge_vectors:
                avg_knowledge_states = np.mean(knowledge_vectors, axis=0)
            else:
                avg_knowledge_states = [0.5] * self.knowledge_n
        
        # 识别薄弱知识点
        weaknesses = []
        
        for i, (user_state, avg_state) in enumerate(zip(user_knowledge_vector, avg_knowledge_states)):
            weakness_score = avg_state - user_state
            if weakness_score > 0.15:  # 设定阈值来判断是否为薄弱点
                weaknesses.append({
                    'knowledge_point': f"知识点{i+1}",
                    'user_level': round(user_state, 2),
                    'avg_level': round(avg_state, 2),
                    'weakness_score': round(weakness_score, 2)
                })
        
        return weaknesses, user_knowledge_vector

    def calculate_overall_score(self, user_knowledge_vector):
        """
        计算用户整体学习掌握评分
        
        Args:
            user_knowledge_vector: 用户知识向量
        
        Returns:
            overall_score: 整体评分 (0-100)
        """
        # 基于知识状态计算整体评分
        non_zero_knowledge = [k for k in user_knowledge_vector if k > 0]
        if non_zero_knowledge:
            avg_knowledge = sum(non_zero_knowledge) / len(non_zero_knowledge)
        else:
            avg_knowledge = 0.0
        # 转换为百分制
        score = avg_knowledge * 100
        return round(score, 2)

    def get_top_knowledge_points(self, user_knowledge_vector, top_n=5):
        """
        获取用户掌握最好的知识点
        
        Args:
            user_knowledge_vector: 用户知识向量
            top_n: 返回前N个知识点
        
        Returns:
            top_knowledges: 掌握最好的知识点列表 [(index, level), ...]
        """
        indexed_levels = [(i, level) for i, level in enumerate(user_knowledge_vector)]
        top_knowledges = sorted(indexed_levels, key=lambda x: x[1], reverse=True)[:top_n]
        return [(f"知识点{idx+1}", level) for idx, level in top_knowledges if level > 0]

    def analyze_learning_progression(self, historical_responses):
        """
        分析用户学习进度趋势
        
        Args:
            historical_responses: 历史答题记录，按时间顺序排列
                                [{'timestamp': ..., 'responses': [...]}, ...]
        
        Returns:
            progression_analysis: 进步分析结果
        """
        if len(historical_responses) < 2:
            return {"trend": "insufficient_data", "message": "需要至少两次测试才能分析进步趋势"}
        
        scores_over_time = []
        for record in historical_responses:
            knowledge_vector = self.build_knowledge_vector(record['responses'])
            score = self.calculate_overall_score(knowledge_vector)
            scores_over_time.append(score)
        
        # 计算趋势
        initial_score = scores_over_time[0]
        final_score = scores_over_time[-1]
        avg_score = sum(scores_over_time) / len(scores_over_time)
        
        if final_score > initial_score:
            trend = "improving"
            trend_desc = "进步"
        elif final_score < initial_score:
            trend = "declining" 
            trend_desc = "退步"
        else:
            trend = "stable"
            trend_desc = "稳定"
        
        return {
            "trend": trend,
            "trend_description": trend_desc,
            "initial_score": initial_score,
            "final_score": final_score,
            "average_score": avg_score,
            "progress": round(final_score - initial_score, 2),
            "scores_history": scores_over_time
        }


class ModelEvaluator:
    """
    模型评估器，封装完整的评估流程
    """
    def __init__(self, model_path=None, student_n=4163, exer_n=17746, knowledge_n=123):
        self.engine = CognitiveDiagnosisEngine(model_path, student_n, exer_n, knowledge_n)
    
    def evaluate_user(self, user_responses, dataset=None):
        """
        完整评估用户的学习状况
        
        Args:
            user_responses: 用户答题记录
            dataset: 用于相似性分析的数据集
        
        Returns:
            evaluation_result: 评估结果字典
        """
        # 构建用户知识向量
        user_knowledge_vector = self.engine.build_knowledge_vector(user_responses)
        
        # 查找相似用户
        similar_users = []
        if dataset:
            similar_users = self.engine.find_similar_users(user_knowledge_vector, dataset)
        
        # 识别薄弱知识点
        weaknesses, final_user_knowledge = self.engine.identify_weak_knowledge_points(
            user_knowledge_vector, similar_users
        )
        
        # 计算整体评分
        overall_score = self.engine.calculate_overall_score(final_user_knowledge)
        
        # 获取掌握最好的知识点
        top_knowledges = self.engine.get_top_knowledge_points(final_user_knowledge)
        
        # 构建评估结果
        evaluation_result = {
            'overall_score': overall_score,
            'knowledge_vector': final_user_knowledge,
            'weaknesses': weaknesses,
            'strengths': top_knowledges,
            'similar_users': similar_users,
            'analysis_summary': self._generate_analysis_summary(
                overall_score, weaknesses, top_knowledges, similar_users
            )
        }
        
        return evaluation_result
    
    def _generate_analysis_summary(self, overall_score, weaknesses, strengths, similar_users):
        """
        生成分析摘要
        """
        summary = {
            'overall_performance': '优秀' if overall_score >= 80 else '良好' if overall_score >= 70 else '一般' if overall_score >= 60 else '需加强',
            'weakness_count': len(weaknesses),
            'strength_count': len(strengths),
            'similar_user_count': len(similar_users)
        }
        return summary
"""
认知诊断模型评估引擎
负责核心的模型推理、相似性分析和薄弱点识别
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import torch

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
        try:
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
    
    def build_knowledge_vector(self, user_responses, involved_knowledge_indices):
        """
        根据用户答题记录构建知识掌握程度向量
        只计算涉及知识点的掌握程度（加权平均得分）
        
        Args:
            user_responses: 用户答题记录列表
                          格式: [{'exer_id': int, 'score': float, 'knowledge_code': [int, ...]}, ...]
            involved_knowledge_indices: 涉及的知识点索引列表（必须提供）
        
        Returns:
            knowledge_vector: 知识点掌握程度向量 [float, ...]
        """
        # 只使用涉及的知识点
        knowledge_n = len(involved_knowledge_indices)
        knowledge_vector = [0.0] * knowledge_n
        
        # 累积每个知识点的得分
        knowledge_scores = {}
        knowledge_counts = {}
        
        for response in user_responses:
            score = response['score']
            knowledge_codes = response.get('knowledge_code', [])
            
            # 只处理涉及的知识点
            for code in knowledge_codes:
                if 1 <= code <= self.knowledge_n:  # 确保知识点id在范围内
                    code_idx = code - 1  # 转换为0基索引
                    
                    # 只处理涉及的知识点
                    if code_idx in involved_knowledge_indices:
                        # 映射到新的索引位置
                        mapped_idx = involved_knowledge_indices.index(code_idx)
                        
                        if mapped_idx not in knowledge_scores:
                            knowledge_scores[mapped_idx] = 0
                            knowledge_counts[mapped_idx] = 0
                        knowledge_scores[mapped_idx] += score  # 累计知识点得分
                        knowledge_counts[mapped_idx] += 1  # 累计知识点出现次数
        
        # 计算加权平均
        for i in range(knowledge_n):
            if i in knowledge_counts and knowledge_counts[i] > 0:
                knowledge_vector[i] = knowledge_scores[i] / knowledge_counts[i]
        
        return knowledge_vector

    def find_similar_users(self, user_knowledge_vector, dataset, user_responses=None):
        """
        在数据集中查找与当前用户情况相似的用户
        
        Args:
            user_knowledge_vector: 用户知识掌握程度向量
            dataset: 数据集，格式: [{'user_id': int, 'logs': [...], ...}, ...]
            user_responses: 当前用户的答题记录，用于确定涉及的知识点
        
        Returns:
            similar_users: 相似用户列表 [(similarity, user_data), ...]
        """
        similarities = []
        
        # 确定涉及的知识点索引
        involved_knowledge_indices = []
        if user_responses:
            # 从用户答题记录中提取涉及的知识点
            for response in user_responses:
                knowledge_codes = response.get('knowledge_code', [])
                for code in knowledge_codes:
                    if 1 <= code <= self.knowledge_n:
                        idx = code - 1
                        if idx not in involved_knowledge_indices:
                            involved_knowledge_indices.append(idx)
        
        # 如果没有找到涉及的知识点，使用所有知识点
        if not involved_knowledge_indices:
            involved_knowledge_indices = list(range(self.knowledge_n))
        
        # 构建当前用户的知识向量
        if user_responses:
            user_knowledge_vector = self.build_knowledge_vector(user_responses, involved_knowledge_indices)
        
        # 遍历数据集，依赖build_knowledge_vector自动处理知识点过滤
        for user_data in dataset:
            user_logs = user_data.get('logs', [])
            
            # 构建数据集用户知识点掌握向量
            dataset_knowledge_vector = self.build_knowledge_vector(user_logs, involved_knowledge_indices)
            
            # 转换为numpy数组并重塑为(1, -1)形状
            user_array = np.array(user_knowledge_vector).reshape(1, -1)
            dataset_array = np.array(dataset_knowledge_vector).reshape(1, -1)
            
            try:
                # 计算余弦相似度(核心步骤)
                similarity = cosine_similarity(user_array, dataset_array)[0][0]
                similarities.append((similarity, user_data)) #保存相似度和用户数据
            except:
                # 如果计算相似度失败，跳过该用户
                continue
        
        # 按相似度排序并返回最相似的前几个用户
        similarities.sort(reverse=True)
        return similarities[:min(10, len(similarities))]  # 返回最多10个最相似用户

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
        # 确定涉及的知识点索引（基于用户知识向量）
        involved_knowledge_indices = []
        for i, level in enumerate(user_knowledge_vector):
            if level > 0:  # 如果用户在该知识点上有得分，说明涉及该知识点
                involved_knowledge_indices.append(i)
        
        # 如果没有涉及的知识点，使用所有知识点
        if not involved_knowledge_indices:
            involved_knowledge_indices = list(range(self.knowledge_n))
        
        if not similar_users:
            print("未找到相似用户，使用全局平均值进行比较")
            # 使用整个数据集的平均知识状态
            avg_knowledge_states = [0.5] * len(involved_knowledge_indices)  # 默认值
        else:
            # 使用相似用户的平均知识状态
            knowledge_vectors = []
            for _, user_data in similar_users:
                user_logs = user_data.get('logs', [])
                knowledge_vector = self.build_knowledge_vector(user_logs, involved_knowledge_indices)
                knowledge_vectors.append(knowledge_vector)
            
            if knowledge_vectors:
                avg_knowledge_states = np.mean(knowledge_vectors, axis=0)
            else:
                avg_knowledge_states = [0.5] * len(involved_knowledge_indices)
        
        # 识别薄弱知识点
        weaknesses = []
        
        # 同时遍历用户知识向量和平均知识掌握向量，进行比较
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
        # 从用户答题记录中提取涉及的知识点索引
        involved_knowledge_indices = self._extract_involved_knowledge_indices(user_responses)
        
        # 构建用户知识向量（使用涉及的知识点）
        user_knowledge_vector = self.engine.build_knowledge_vector(user_responses, involved_knowledge_indices)
        
        # 查找相似用户
        similar_users = []
        if dataset:
            similar_users = self.engine.find_similar_users(user_knowledge_vector, dataset, user_responses)
        
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
    
    def _extract_involved_knowledge_indices(self, user_responses):
        """
        从用户答题记录中提取涉及的知识点索引
        
        Args:
            user_responses: 用户答题记录
        
        Returns:
            involved_knowledge_indices: 涉及的知识点索引列表
        """
        involved_knowledge_indices = []
        
        for response in user_responses:
            knowledge_codes = response.get('knowledge_code', [])
            for code in knowledge_codes:
                if 1 <= code <= self.engine.knowledge_n:
                    idx = code - 1
                    if idx not in involved_knowledge_indices:
                        involved_knowledge_indices.append(idx)
        
        # 如果没有找到涉及的知识点，使用所有知识点
        if not involved_knowledge_indices:
            involved_knowledge_indices = list(range(self.engine.knowledge_n))
        
        return involved_knowledge_indices
    
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


if __name__ == "__main__":
    """
    主函数 - 可以直接运行app2.py进行测试
    """
    # 创建评估器实例（使用ModelEvaluator类，它包含evaluate_user方法）
    evaluator = ModelEvaluator(
        student_n=4163,      # 学生总数（根据您的数据调整）
        exer_n=17746,         # 题目总数（根据您的数据调整）
        knowledge_n=123,     # 知识点总数（根据您的数据调整）
        model_path=None     # 预训练模型路径（可选）
    )
    
    # 示例测试数据
    user_responses = [
        {'exer_id': 1, 'score': 0.8, 'knowledge_code': [1, 2, 3]},
        {'exer_id': 2, 'score': 0.6, 'knowledge_code': [2, 3, 4]},
        {'exer_id': 3, 'score': 0.9, 'knowledge_code': [1, 4, 5]},
        {'exer_id': 4, 'score': 0.7, 'knowledge_code': [3, 5, 6]},
        {'exer_id': 5, 'score': 0.5, 'knowledge_code': [1, 6, 7]}
    ]
    
    # 示例数据集（相似用户数据）
    dataset = [
        {
            'user_id': 1, 
            'logs': [
                {'exer_id': 1, 'score': 0.9, 'knowledge_code': [1, 2, 3]},
                {'exer_id': 2, 'score': 0.7, 'knowledge_code': [2, 3, 4]},
                {'exer_id': 3, 'score': 0.8, 'knowledge_code': [1, 4, 5]}
            ]
        },
        {
            'user_id': 2,
            'logs': [
                {'exer_id': 1, 'score': 0.6, 'knowledge_code': [1, 2, 3]},
                {'exer_id': 2, 'score': 0.5, 'knowledge_code': [2, 3, 4]},
                {'exer_id': 3, 'score': 0.7, 'knowledge_code': [1, 4, 5]}
            ]
        }
    ]
    
    print("开始认知诊断评估...")
    
    # 执行评估
    try:
        result = evaluator.evaluate_user(user_responses, dataset)
        
        # 输出结果
        print("\n=== 认知诊断结果 ===")
        print(f"整体评分: {result.get('overall_score', 0)}分")
        print(f"学习表现: {result.get('analysis_summary', {}).get('overall_performance', '未知')}")
        
        print("\n=== 薄弱知识点 ===")
        weaknesses = result.get('weaknesses', [])
        for weakness in weaknesses[:3]:  # 显示前3个薄弱点
            print(f"  {weakness.get('knowledge_point', '未知')}: 掌握度{weakness.get('user_level', 0)} (平均{weakness.get('avg_level', 0)})")
        
        print("\n=== 优势知识点 ===")
        strengths = result.get('strengths', [])
        for strength in strengths[:3]:  # 显示前3个优势点
            print(f"  {strength[0]}: 掌握度{strength[1]:.2f}")
            
        print(f"\n找到相似用户: {result.get('analysis_summary', {}).get('similar_user_count', 0)}个")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
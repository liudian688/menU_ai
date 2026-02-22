"""
个性化认知诊断系统
基于预训练模型和外部数据库的用户数据分析
"""

import torch
import numpy as np
from model import Net


class DatabaseConnector:
    """数据库连接器（抽象类，具体实现由外部提供）"""
    
    def get_user_responses(self, user_id):
        """
        从外部数据库获取指定用户的所有答题记录
        
        Args:
            user_id: 用户ID
            
        Returns:
            user_responses: 用户答题记录列表
                        格式: [{'exer_id': int, 'score': float, 'knowledge_code': [int, ...]}, ...]
        """
        # 这里应该调用外部数据库接口
        # 返回示例数据格式
        raise NotImplementedError("需要外部实现数据库连接")
    
    def save_diagnosis_result(self, user_id, diagnosis_result):
        """
        将诊断结果保存到外部数据库
        
        Args:
            user_id: 用户ID
            diagnosis_result: 诊断结果字典
        """
        # 这里应该调用外部数据库接口
        raise NotImplementedError("需要外部实现数据库连接")


class PersonalizedDiagnosisEngine:
    """个性化诊断引擎"""
    
    def __init__(self, model_path, student_n, exer_n, knowledge_n, db_connector):
        """
        初始化诊断引擎
        
        Args:
            model_path: 预训练模型路径
            student_n: 学生总数（根据基础数据集）
            exer_n: 题目总数（根据基础数据集）
            knowledge_n: 知识点总数（根据基础数据集）
            db_connector: 数据库连接器实例
        """
        self.student_n = student_n
        self.exer_n = exer_n
        self.knowledge_n = knowledge_n
        self.db_connector = db_connector
        
        # 加载预训练模型
        self.model = self._load_pretrained_model(model_path)
        
        # 知识点映射（可根据实际情况调整）
        self.knowledge_point_names = {
            i: f"知识点{i}" for i in range(1, knowledge_n + 1)
        }
    
    def _load_pretrained_model(self, model_path):
        """加载预训练模型"""
        try:
            # 创建模型实例
            model = Net(self.student_n, self.exer_n, self.knowledge_n)
            
            # 加载预训练权重
            device = torch.device("cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()  # 设置为评估模式
            print(f"预训练模型已从 {model_path} 加载")
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def diagnose_user(self, user_id):
        """
        对指定用户进行个性化诊断
        
        Args:
            user_id: 用户ID
            
        Returns:
            diagnosis_result: 诊断结果字典
        """
        print(f"开始对用户 {user_id} 进行诊断...")
        
        # 1. 从数据库获取用户数据
        user_responses = self.db_connector.get_user_responses(user_id)
        
        if not user_responses:
            return {
                'success': False,
                'error': f"用户 {user_id} 没有答题记录"
            }
        
        print(f"获取到用户 {user_id} 的 {len(user_responses)} 条答题记录")
        
        # 2. 使用模型分析用户知识状态
        knowledge_state = self._analyze_knowledge_state(user_responses)
        
        # 3. 生成诊断报告
        diagnosis_result = self._generate_diagnosis_report(user_id, knowledge_state, user_responses)
        
        # 4. 保存诊断结果到数据库
        self.db_connector.save_diagnosis_result(user_id, diagnosis_result)
        
        print(f"用户 {user_id} 诊断完成")
        return diagnosis_result
    
    def _analyze_knowledge_state(self, user_responses):
        """
        使用预训练模型分析用户知识状态
        
        Args:
            user_responses: 用户答题记录
            
        Returns:
            knowledge_state: 知识状态向量
        """
        # 由于模型需要学生ID，我们创建一个虚拟的学生ID用于分析
        # 在实际应用中，可能需要更复杂的处理
        dummy_student_id = 0  # 使用固定ID进行分析
        
        with torch.no_grad():
            # 将用户ID转换为tensor
            stu_id_tensor = torch.LongTensor([dummy_student_id])
            
            # 获取知识状态向量
            knowledge_state = self.model.get_knowledge_status(stu_id_tensor)
            knowledge_state = knowledge_state.numpy().flatten()
        
        return knowledge_state
    
    def _generate_diagnosis_report(self, user_id, knowledge_state, user_responses):
        """
        生成详细的诊断报告
        
        Args:
            user_id: 用户ID
            knowledge_state: 知识状态向量
            user_responses: 用户答题记录
            
        Returns:
            diagnosis_report: 诊断报告字典
        """
        # 计算整体评分
        overall_score = self._calculate_overall_score(knowledge_state)
        
        # 识别薄弱知识点
        weaknesses = self._identify_weak_knowledge_points(knowledge_state)
        
        # 识别优势知识点
        strengths = self._identify_strong_knowledge_points(knowledge_state)
        
        # 分析学习趋势
        learning_trend = self._analyze_learning_trend(user_responses)
        
        # 生成诊断报告
        diagnosis_report = {
            'success': True,
            'user_id': user_id,
            'overall_score': overall_score,
            'performance_level': self._get_performance_level(overall_score),
            'knowledge_state': {
                'vector': knowledge_state.tolist(),
                'weak_points': weaknesses,
                'strong_points': strengths
            },
            'learning_analysis': {
                'total_questions': len(user_responses),
                'average_score': np.mean([r['score'] for r in user_responses]),
                'trend': learning_trend
            },
            'recommendations': self._generate_recommendations(weaknesses, strengths),
            'diagnosis_time': np.datetime64('now').astype(str)
        }
        
        return diagnosis_report
    
    def _calculate_overall_score(self, knowledge_state):
        """计算整体评分"""
        # 基于知识状态计算整体评分（0-100分）
        avg_knowledge = np.mean(knowledge_state)
        overall_score = avg_knowledge * 100
        return round(overall_score, 2)
    
    def _identify_weak_knowledge_points(self, knowledge_state):
        """识别薄弱知识点"""
        weaknesses = []
        threshold = 0.6  # 掌握度低于60%视为薄弱
        
        for i, score in enumerate(knowledge_state):
            if score < threshold:
                knowledge_name = self.knowledge_point_names.get(i + 1, f"知识点{i + 1}")
                weaknesses.append({
                    'knowledge_id': i + 1,
                    'knowledge_name': knowledge_name,
                    'mastery_level': round(score, 3),
                    'improvement_needed': round(threshold - score, 3)
                })
        
        # 按薄弱程度排序
        weaknesses.sort(key=lambda x: x['improvement_needed'], reverse=True)
        return weaknesses
    
    def _identify_strong_knowledge_points(self, knowledge_state):
        """识别优势知识点"""
        strengths = []
        threshold = 0.8  # 掌握度高于80%视为优势
        
        for i, score in enumerate(knowledge_state):
            if score >= threshold:
                knowledge_name = self.knowledge_point_names.get(i + 1, f"知识点{i + 1}")
                strengths.append({
                    'knowledge_id': i + 1,
                    'knowledge_name': knowledge_name,
                    'mastery_level': round(score, 3)
                })
        
        # 按掌握程度排序
        strengths.sort(key=lambda x: x['mastery_level'], reverse=True)
        return strengths
    
    def _analyze_learning_trend(self, user_responses):
        """分析学习趋势（简化版）"""
        if len(user_responses) < 2:
            return "数据不足，无法分析趋势"
        
        # 按时间排序（假设有timestamp字段）
        sorted_responses = sorted(user_responses, key=lambda x: x.get('timestamp', 0))
        
        # 计算早期和晚期的平均得分
        early_scores = [r['score'] for r in sorted_responses[:len(sorted_responses)//2]]
        late_scores = [r['score'] for r in sorted_responses[len(sorted_responses)//2:]]
        
        early_avg = np.mean(early_scores)
        late_avg = np.mean(late_scores)
        
        if late_avg > early_avg + 0.1:
            return "进步明显"
        elif late_avg > early_avg:
            return "稳步进步"
        elif late_avg < early_avg - 0.1:
            return "需要关注"
        else:
            return "保持稳定"
    
    def _get_performance_level(self, score):
        """获取表现等级"""
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "中等"
        elif score >= 60:
            return "及格"
        else:
            return "需加强"
    
    def _generate_recommendations(self, weaknesses, strengths):
        """生成学习建议"""
        recommendations = []
        
        # 薄弱知识点建议
        if weaknesses:
            top_weak = weaknesses[:3]  # 关注前3个最薄弱点
            rec_text = f"建议重点复习以下知识点: {', '.join([w['knowledge_name'] for w in top_weak])}"
            recommendations.append({
                'type': 'weakness_focus',
                'priority': 'high',
                'content': rec_text
            })
        
        # 优势知识点建议
        if strengths:
            rec_text = f"您在以下知识点表现优秀: {', '.join([s['knowledge_name'] for s in strengths[:3]])}"
            recommendations.append({
                'type': 'strength_acknowledge',
                'priority': 'medium',
                'content': rec_text
            })
        
        # 总体建议
        if len(weaknesses) > len(strengths):
            recommendations.append({
                'type': 'general_advice',
                'priority': 'medium',
                'content': "建议系统性地复习基础知识，打好坚实基础"
            })
        else:
            recommendations.append({
                'type': 'general_advice',
                'priority': 'medium',
                'content': "继续保持良好学习状态，适当挑战更高难度内容"
            })
        
        return recommendations


# 使用示例
if __name__ == "__main__":
    # 示例：需要外部实现数据库连接器
    class ExampleDBConnector(DatabaseConnector):
        def get_user_responses(self, user_id):
            # 示例数据
            return [
                {'exer_id': 1, 'score': 0.8, 'knowledge_code': [1, 2, 3]},
                {'exer_id': 2, 'score': 0.6, 'knowledge_code': [2, 3, 4]},
                {'exer_id': 3, 'score': 0.9, 'knowledge_code': [1, 4, 5]}
            ]
        
        def save_diagnosis_result(self, user_id, diagnosis_result):
            print(f"保存用户 {user_id} 的诊断结果到数据库")
    
    # 创建诊断引擎
    engine = PersonalizedDiagnosisEngine(
        model_path="./model/trained_model.pth",  # 预训练模型路径
        student_n=4163,
        exer_n=17746,
        knowledge_n=123,
        db_connector=ExampleDBConnector()
    )
    
    # 对用户进行诊断
    result = engine.diagnose_user(1001)
    print("诊断结果:", result)
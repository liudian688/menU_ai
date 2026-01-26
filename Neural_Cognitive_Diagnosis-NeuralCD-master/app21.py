from app21_models import get_db, User, Exercise, KnowledgePoint, UserResponse, Assessment, Skill
from app2 import ModelEvaluator
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class DatabaseInterface:
    """
    数据库接口类，提供数据访问功能
    """
    
    def __init__(self):
        self.db = next(get_db())
    
    def __del__(self):
        self.db.close()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_exercises_by_skill(self, skill_id: int, limit: int = 10) -> List[Exercise]:
        """根据技能ID获取练习题"""
        return self.db.query(Exercise).filter(Exercise.skill_id == skill_id).limit(limit).all()
    
    def get_knowledge_points_by_skill(self, skill_id: int) -> List[KnowledgePoint]:
        """根据技能ID获取知识点"""
        return self.db.query(KnowledgePoint).filter(KnowledgePoint.skill_id == skill_id).all()
    
    def get_user_responses(self, user_id: int, limit: int = 100) -> List[Dict]:
        """获取用户答题记录"""
        responses = self.db.query(UserResponse).filter(UserResponse.user_id == user_id).limit(limit).all()
        
        result = []
        for resp in responses:
            # 获取该回答涉及的知识点ID
            knowledge_ids = []
            for assoc in resp.knowledge_points:
                knowledge_ids.append(assoc.id)
            
            result.append({
                'exer_id': resp.exercise_id,
                'score': resp.score,
                'knowledge_code': knowledge_ids,
                'timestamp': resp.timestamp.isoformat() if resp.timestamp else None
            })
        
        return result
    
    def get_similar_users_data(self, current_user_id: int, skill_id: int, limit: int = 20) -> List[Dict]:
        """获取相似用户的数据用于比较"""
        # 获取具有相同技能练习记录的其他用户
        similar_users_data = []
        
        # 查询做过相同练习的其他用户
        subquery = self.db.query(UserResponse.exercise_id).filter(
            UserResponse.user_id == current_user_id
        ).subquery()
        
        other_user_responses = self.db.query(UserResponse).filter(
            UserResponse.exercise_id.in_(subquery),
            UserResponse.user_id != current_user_id
        ).all()
        
        # 按用户分组
        user_logs_map = {}
        for resp in other_user_responses:
            user_id = resp.user_id
            if user_id not in user_logs_map:
                user_logs_map[user_id] = []
            
            # 获取该回答涉及的知识点ID
            knowledge_ids = []
            for assoc in resp.knowledge_points:
                knowledge_ids.append(assoc.id)
            
            user_logs_map[user_id].append({
                'exer_id': resp.exercise_id,
                'score': resp.score,
                'knowledge_code': knowledge_ids
            })
        
        # 转换为所需格式
        for user_id, logs in user_logs_map.items():
            if len(logs) >= 3:  # 至少3条记录才纳入考虑
                similar_users_data.append({
                    'user_id': user_id,
                    'logs': logs
                })
        
        return similar_users_data[:limit]  # 限制数量
    
    def save_assessment_result(self, user_id: int, result: Dict[str, Any]) -> Assessment:
        """保存评估结果到数据库"""
        assessment = Assessment(
            user_id=user_id,
            overall_score=result.get('overall_score', 0),
            weaknesses_json=json.dumps(result.get('weaknesses', [])),
            strengths_json=json.dumps(result.get('strengths', [])),
            knowledge_vector_json=json.dumps(result.get('knowledge_vector', [])),
            similar_users_json=json.dumps(result.get('similar_users', [])),
            analysis_summary_json=json.dumps(result.get('analysis_summary', {}))
        )
        
        self.db.add(assessment)
        self.db.commit()
        self.db.refresh(assessment)
        
        return assessment


class CognitiveDiagnosisAPI:
    """
    认知诊断API类，整合数据库和模型评估功能
    """
    
    def __init__(self, model_path: Optional[str] = "./model/model_epoch5"):
        self.db_interface = DatabaseInterface()
        try:
            self.evaluator = ModelEvaluator(model_path=model_path)
        except Exception as e:
            print(f"加载模型失败: {e}，使用默认模型")
            self.evaluator = ModelEvaluator()  # 使用默认模型
    
    def conduct_assessment(self, username: str, skill_name: str) -> Dict[str, Any]:
        """
        执行完整的认知评估
        
        Args:
            username: 用户名
            skill_name: 技能名称
        
        Returns:
            评估结果字典
        """
        # 1. 获取用户信息
        user = self.db_interface.get_user_by_username(username)
        if not user:
            raise ValueError(f"用户 {username} 不存在")
        
        # 2. 获取技能信息
        skill = self.db_interface.db.query(Skill).filter(Skill.name == skill_name).first()
        if not skill:
            raise ValueError(f"技能 {skill_name} 不存在")
        
        # 3. 获取用户答题记录
        user_responses = self.db_interface.get_user_responses(user.id)
        
        if not user_responses:
            # 如果用户没有答题记录，返回提示
            return {
                'status': 'no_data',
                'message': '用户暂无答题记录，无法进行评估',
                'user_id': user.id,
                'skill_id': skill.id
            }
        
        # 4. 获取相似用户数据
        similar_users_data = self.db_interface.get_similar_users_data(user.id, skill.id)
        
        # 5. 执行模型评估
        result = self.evaluator.evaluate_user(user_responses, dataset=similar_users_data)
        
        # 6. 保存评估结果到数据库
        assessment = self.db_interface.save_assessment_result(user.id, result)
        
        # 添加额外信息
        result['user_id'] = user.id
        result['skill_name'] = skill_name
        result['assessment_id'] = assessment.id
        result['status'] = 'success'
        
        return result
    
    def get_recommendation_exercises(self, username: str, skill_name: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        获取推荐练习题（基于薄弱知识点）
        
        Args:
            username: 用户名
            skill_name: 技能名称
            count: 返回练习题数量
        
        Returns:
            推荐练习题列表
        """
        # 首先执行评估以获取薄弱知识点
        assessment_result = self.conduct_assessment(username, skill_name)
        
        if assessment_result.get('status') != 'success':
            return []
        
        weaknesses = assessment_result.get('weaknesses', [])
        
        if not weaknesses:
            # 如果没有明显的薄弱点，返回随机练习题
            skill = self.db_interface.db.query(Skill).filter(Skill.name == skill_name).first()
            exercises = self.db_interface.get_exercises_by_skill(skill.id, limit=count)
            return [{
                'id': ex.id,
                'title': ex.title,
                'content': ex.content,
                'difficulty_level': ex.difficulty_level
            } for ex in exercises]
        
        # 基于薄弱知识点推荐练习题
        recommended_exercises = []
        for weakness in weaknesses[:2]:  # 只看前2个最薄弱的知识点
            knowledge_point_name = weakness['knowledge_point'].replace('知识点', '')
            # 这里需要根据实际知识点ID来查找对应的练习题
            # 简化实现：返回该技能下的练习题
            skill = self.db_interface.db.query(Skill).filter(Skill.name == skill_name).first()
            exercises = self.db_interface.get_exercises_by_skill(skill.id, limit=count)
            
            for ex in exercises:
                recommended_exercises.append({
                    'id': ex.id,
                    'title': ex.title,
                    'content': ex.content,
                    'difficulty_level': ex.difficulty_level,
                    'target_weakness': weakness['knowledge_point']
                })
        
        return recommended_exercises[:count]
    
    def get_user_progress(self, username: str, skill_name: str) -> Dict[str, Any]:
        """
        获取用户学习进度
        
        Args:
            username: 用户名
            skill_name: 技能名称
        
        Returns:
            学习进度信息
        """
        user = self.db_interface.get_user_by_username(username)
        if not user:
            raise ValueError(f"用户 {username} 不存在")
        
        skill = self.db_interface.db.query(Skill).filter(Skill.name == skill_name).first()
        if not skill:
            raise ValueError(f"技能 {skill_name} 不存在")
        
        # 获取用户的历史评估记录
        assessments = self.db_interface.db.query(Assessment).filter(
            Assessment.user_id == user.id
        ).order_by(Assessment.assessment_date).all()
        
        progress_data = []
        for assessment in assessments:
            progress_data.append({
                'date': assessment.assessment_date.isoformat(),
                'overall_score': assessment.overall_score,
                'analysis_summary': json.loads(assessment.analysis_summary_json)
            })
        
        return {
            'user_id': user.id,
            'skill_name': skill_name,
            'progress_history': progress_data,
            'total_assessments': len(progress_data)
        }


# 示例使用函数
def example_api_usage():
    """
    API使用示例
    """
    print("=" * 60)
    print("认知诊断API使用示例")
    print("=" * 60)
    
    # 初始化API
    api = CognitiveDiagnosisAPI()
    
    # 示例1: 执行评估
    try:
        result = api.conduct_assessment("student1", "编程")
        print(f"评估状态: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"整体评分: {result.get('overall_score')}")
            print(f"薄弱知识点数量: {result['analysis_summary']['weakness_count']}")
            print(f"优势知识点数量: {result['analysis_summary']['strength_count']}")
    except Exception as e:
        print(f"评估过程中出错: {e}")
    
    # 示例2: 获取推荐练习题
    try:
        recommendations = api.get_recommendation_exercises("student1", "编程", 3)
        print(f"\n推荐练习题数量: {len(recommendations)}")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['title']} (难度: {rec['difficulty_level']})")
    except Exception as e:
        print(f"获取推荐时出错: {e}")
    
    # 示例3: 获取学习进度
    try:
        progress = api.get_user_progress("student1", "编程")
        print(f"\n学习进度记录数: {progress['total_assessments']}")
        for record in progress['progress_history']:
            print(f"  - 日期: {record['date'][:10]}, 评分: {record['overall_score']:.2f}")
    except Exception as e:
        print(f"获取进度时出错: {e}")


if __name__ == "__main__":
    example_api_usage()
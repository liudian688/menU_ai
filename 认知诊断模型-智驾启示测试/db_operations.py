from flask import Flask
import pandas as pd
from typing import List, Dict, Optional
import logging

from model_SQLAIchemy import db, KnowledgePoint, Question, QMatrix, Student, AnswerRecord

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NeuralCDDB:
    """NeuralCD数据库操作类（SQLAlchemy版本）"""
    
    def __init__(self, app: Flask = None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """初始化应用"""
        self.app = app
        db.init_app(app)
    
    def create_tables(self):
        """创建所有表"""
        with self.app.app_context():
            db.create_all()
            logger.info("所有表创建成功")
    
    def drop_tables(self):
        """删除所有表（谨慎使用）"""
        with self.app.app_context():
            db.drop_all()
            logger.warning("所有表已删除")
    
    # ==================== 知识点操作 ====================
    
    def add_knowledge_point(self, kp_code: str, kp_name: str, 
                           description: str = '', category: str = '', 
                           difficulty_level: float = 0.5) -> KnowledgePoint:
        """添加单个知识点"""
        with self.app.app_context():
            kp = KnowledgePoint(
                kp_code=kp_code,
                kp_name=kp_name,
                description=description,
                category=category,
                difficulty_level=difficulty_level
            )
            db.session.add(kp)
            db.session.commit()
            logger.info(f"添加知识点: {kp_code} - {kp_name}")
            return kp
    
    def add_knowledge_points_batch(self, kp_list: List[Dict]) -> List[KnowledgePoint]:
        """批量添加知识点"""
        with self.app.app_context():
            added_kps = []
            for kp_data in kp_list:
                kp = KnowledgePoint(
                    kp_code=kp_data['kp_code'],
                    kp_name=kp_data['kp_name'],
                    description=kp_data.get('description', ''),
                    category=kp_data.get('category', ''),
                    difficulty_level=kp_data.get('difficulty_level', 0.5)
                )
                db.session.add(kp)
                added_kps.append(kp)
            
            db.session.commit()
            logger.info(f"批量添加 {len(added_kps)} 个知识点")
            return added_kps
    
    def get_knowledge_point(self, kp_code: str = None, kp_id: int = None) -> Optional[KnowledgePoint]:
        """获取知识点"""
        with self.app.app_context():
            if kp_code:
                return KnowledgePoint.query.filter_by(kp_code=kp_code).first()
            elif kp_id:
                return KnowledgePoint.query.get(kp_id)
            return None
    
    def get_all_knowledge_points(self) -> List[KnowledgePoint]:
        """获取所有知识点"""
        with self.app.app_context():
            return KnowledgePoint.query.all()
    
    def update_knowledge_point(self, kp_code: str, **kwargs) -> Optional[KnowledgePoint]:
        """更新知识点"""
        with self.app.app_context():
            kp = KnowledgePoint.query.filter_by(kp_code=kp_code).first()
            if kp:
                for key, value in kwargs.items():
                    if hasattr(kp, key):
                        setattr(kp, key, value)
                db.session.commit()
                logger.info(f"更新知识点: {kp_code}")
            return kp
    
    def delete_knowledge_point(self, kp_code: str) -> bool:
        """删除知识点"""
        with self.app.app_context():
            kp = KnowledgePoint.query.filter_by(kp_code=kp_code).first()
            if kp:
                db.session.delete(kp)
                db.session.commit()
                logger.info(f"删除知识点: {kp_code}")
                return True
            return False
    
    # ==================== 题目操作 ====================
    
    def add_question(self, question_code: str, content: str = '',
                    difficulty: float = 0.5, discrimination: float = 1.0,
                    type: str = '', source: str = '') -> Question:
        """添加单个题目"""
        with self.app.app_context():
            q = Question(
                question_code=question_code,
                content=content,
                difficulty=difficulty,
                discrimination=discrimination,
                type=type,
                source=source
            )
            db.session.add(q)
            db.session.commit()
            logger.info(f"添加题目: {question_code}")
            return q
    
    def add_questions_batch(self, q_list: List[Dict]) -> List[Question]:
        """批量添加题目"""
        with self.app.app_context():
            added_qs = []
            for q_data in q_list:
                q = Question(
                    question_code=q_data['question_code'],
                    content=q_data.get('content', ''),
                    difficulty=q_data.get('difficulty', 0.5),
                    discrimination=q_data.get('discrimination', 1.0),
                    type=q_data.get('type', ''),
                    source=q_data.get('source', '')
                )
                db.session.add(q)
                added_qs.append(q)
            
            db.session.commit()
            logger.info(f"批量添加 {len(added_qs)} 道题目")
            return added_qs
    
    def get_question(self, question_code: str = None, question_id: int = None) -> Optional[Question]:
        """获取题目"""
        with self.app.app_context():
            if question_code:
                return Question.query.filter_by(question_code=question_code).first()
            elif question_id:
                return Question.query.get(question_id)
            return None
    
    def get_all_questions(self) -> List[Question]:
        """获取所有题目"""
        with self.app.app_context():
            return Question.query.all()
    
    # ==================== Q矩阵操作 ====================
    
    def add_q_matrix_item(self, question_code: str, kp_code: str, 
                          is_relevant: bool = True, weight: float = 1.0) -> Optional[QMatrix]:
        """添加单个Q矩阵条目"""
        with self.app.app_context():
            question = self.get_question(question_code=question_code)
            kp = self.get_knowledge_point(kp_code=kp_code)
            
            if not question or not kp:
                logger.error(f"找不到题目 {question_code} 或知识点 {kp_code}")
                return None
            
            # 检查是否已存在
            existing = QMatrix.query.filter_by(
                question_id=question.id,
                knowledge_point_id=kp.id
            ).first()
            
            if existing:
                existing.is_relevant = is_relevant
                existing.weight = weight
                item = existing
                logger.info(f"更新Q矩阵条目: {question_code} - {kp_code}")
            else:
                item = QMatrix(
                    question_id=question.id,
                    knowledge_point_id=kp.id,
                    is_relevant=is_relevant,
                    weight=weight
                )
                db.session.add(item)
                logger.info(f"添加Q矩阵条目: {question_code} - {kp_code}")
            
            db.session.commit()
            return item
    
    def add_q_matrix_batch(self, qm_list: List[Dict]) -> int:
        """批量添加Q矩阵条目"""
        with self.app.app_context():
            count = 0
            for item in qm_list:
                result = self.add_q_matrix_item(
                    question_code=item['question_code'],
                    kp_code=item['kp_code'],
                    is_relevant=item.get('is_relevant', True),
                    weight=item.get('weight', 1.0)
                )
                if result:
                    count += 1
            
            logger.info(f"批量添加/更新 {count} 条Q矩阵记录")
            return count
    
    def get_q_matrix_dataframe(self) -> pd.DataFrame:
        """
        获取Q矩阵（DataFrame格式）
        返回：行=题目，列=知识点，值=is_relevant
        """
        with self.app.app_context():
            # 查询所有Q矩阵数据
            query = db.session.query(
                Question.question_code,
                KnowledgePoint.kp_code,
                QMatrix.is_relevant
            ).join(
                QMatrix, QMatrix.question_id == Question.id
            ).join(
                KnowledgePoint, KnowledgePoint.id == QMatrix.knowledge_point_id
            ).all()
            
            # 转换成DataFrame
            df = pd.DataFrame(query, columns=['question_code', 'kp_code', 'is_relevant'])
            
            # 转换成矩阵格式
            if not df.empty:
                q_matrix = df.pivot_table(
                    index='question_code',
                    columns='kp_code',
                    values='is_relevant',
                    fill_value=0,
                    aggfunc='first'
                )
                # 确保值是整数
                q_matrix = q_matrix.astype(int)
            else:
                q_matrix = pd.DataFrame()
            
            logger.info(f"获取Q矩阵，形状: {q_matrix.shape}")
            return q_matrix
    
    def get_question_knowledge_points(self, question_code: str) -> List[str]:
        """获取题目涉及的知识点"""
        with self.app.app_context():
            question = self.get_question(question_code=question_code)
            if not question:
                return []
            
            results = db.session.query(KnowledgePoint.kp_code).join(
                QMatrix, QMatrix.knowledge_point_id == KnowledgePoint.id
            ).filter(
                QMatrix.question_id == question.id,
                QMatrix.is_relevant == True
            ).all()
            
            return [r[0] for r in results]
    
    def get_knowledge_point_questions(self, kp_code: str) -> List[str]:
        """获取考察某个知识点的题目"""
        with self.app.app_context():
            kp = self.get_knowledge_point(kp_code=kp_code)
            if not kp:
                return []
            
            results = db.session.query(Question.question_code).join(
                QMatrix, QMatrix.question_id == Question.id
            ).filter(
                QMatrix.knowledge_point_id == kp.id,
                QMatrix.is_relevant == True
            ).all()
            
            return [r[0] for r in results]
    
    # ==================== 学生操作 ====================
    
    def add_student(self, student_code: str, name: str = '',
                   grade: str = '', class_name: str = '') -> Student:
        """添加学生"""
        with self.app.app_context():
            student = Student(
                student_code=student_code,
                name=name,
                grade=grade,
                class_name=class_name
            )
            db.session.add(student)
            db.session.commit()
            logger.info(f"添加学生: {student_code}")
            return student
    
    # ==================== 答题记录操作 ====================
    
    def add_answer_record(self, student_code: str, question_code: str,
                          score: float, time_spent: int = None) -> Optional[AnswerRecord]:
        """添加答题记录"""
        with self.app.app_context():
            student = Student.query.filter_by(student_code=student_code).first()
            question = Question.query.filter_by(question_code=question_code).first()
            
            if not student or not question:
                logger.error(f"找不到学生 {student_code} 或题目 {question_code}")
                return None
            
            record = AnswerRecord(
                student_id=student.id,
                question_id=question.id,
                score=score,
                is_correct=(score >= 0.5),  # 假设0.5以上算正确
                time_spent=time_spent
            )
            db.session.add(record)
            db.session.commit()
            logger.info(f"添加答题记录: {student_code} - {question_code} - 得分:{score}")
            return record
    
    # ==================== 导入/导出功能 ====================
    
    def import_from_csv(self, kp_file: str = None, q_file: str = None, 
                        qm_file: str = None):
        """从CSV文件导入数据"""
        import pandas as pd
        
        with self.app.app_context():
            # 导入知识点
            if kp_file:
                kp_df = pd.read_csv(kp_file)
                kp_list = kp_df.to_dict('records')
                self.add_knowledge_points_batch(kp_list)
            
            # 导入题目
            if q_file:
                q_df = pd.read_csv(q_file)
                q_list = q_df.to_dict('records')
                self.add_questions_batch(q_list)
            
            # 导入Q矩阵
            if qm_file:
                qm_df = pd.read_csv(qm_file)
                qm_list = qm_df.to_dict('records')
                self.add_q_matrix_batch(qm_list)
    
    def export_to_csv(self, output_dir: str):
        """导出数据到CSV文件"""
        import os
        
        with self.app.app_context():
            # 导出知识点
            kps = self.get_all_knowledge_points()
            if kps:
                kp_df = pd.DataFrame([kp.to_dict() for kp in kps])
                kp_df.to_csv(os.path.join(output_dir, 'knowledge_points.csv'), index=False)
            
            # 导出题目
            questions = self.get_all_questions()
            if questions:
                q_df = pd.DataFrame([q.to_dict() for q in questions])
                q_df.to_csv(os.path.join(output_dir, 'questions.csv'), index=False)
            
            # 导出Q矩阵
            q_matrix = self.get_q_matrix_dataframe()
            if not q_matrix.empty:
                q_matrix.to_csv(os.path.join(output_dir, 'q_matrix.csv'))
            
            logger.info(f"数据已导出到 {output_dir}")
    
    # ==================== 统计信息 ====================
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self.app.app_context():
            stats = {
                'knowledge_points': KnowledgePoint.query.count(),
                'questions': Question.query.count(),
                'q_matrix': QMatrix.query.count(),
                'students': Student.query.count(),
                'answer_records': AnswerRecord.query.count()
            }
            
            # 计算Q矩阵密度
            if stats['q_matrix'] > 0 and stats['questions'] > 0 and stats['knowledge_points'] > 0:
                density = stats['q_matrix'] / (stats['questions'] * stats['knowledge_points'])
                stats['q_matrix_density'] = f"{density:.2%}"
            
            logger.info(f"数据库统计: {stats}")
            return stats
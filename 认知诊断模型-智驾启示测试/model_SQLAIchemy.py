from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# ==================== 模型定义 ====================

class KnowledgePoint(db.Model):
    """知识点模型"""
    __tablename__ = 'knowledge_points'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    kp_code = db.Column(db.String(50), unique=True, nullable=False, comment='知识点编码')
    kp_name = db.Column(db.String(100), nullable=False, comment='知识点名称')
    description = db.Column(db.Text, comment='知识点描述')
    category = db.Column(db.String(50), comment='知识点分类')
    difficulty_level = db.Column(db.Float, default=0.5, comment='难度等级')
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关系
    q_matrix_items = db.relationship('QMatrix', back_populates='knowledge_point', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<KnowledgePoint {self.kp_code}: {self.kp_name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'kp_code': self.kp_code,
            'kp_name': self.kp_name,
            'description': self.description,
            'category': self.category,
            'difficulty_level': self.difficulty_level
        }


class Question(db.Model):
    """题目模型"""
    __tablename__ = 'questions'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    question_code = db.Column(db.String(50), unique=True, nullable=False, comment='题目编号')
    content = db.Column(db.Text, comment='题目内容')
    difficulty = db.Column(db.Float, default=0.5, comment='题目难度')
    discrimination = db.Column(db.Float, default=1.0, comment='题目区分度')
    type = db.Column(db.String(50), comment='题目类型')
    source = db.Column(db.String(100), comment='题目来源')
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关系
    q_matrix_items = db.relationship('QMatrix', back_populates='question', cascade='all, delete-orphan')
    answer_records = db.relationship('AnswerRecord', back_populates='question', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Question {self.question_code}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'question_code': self.question_code,
            'content': self.content,
            'difficulty': self.difficulty,
            'discrimination': self.discrimination,
            'type': self.type,
            'source': self.source
        }


class QMatrix(db.Model):
    """Q矩阵模型（题目-知识点关系）"""
    __tablename__ = 'q_matrix'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'), nullable=False)
    knowledge_point_id = db.Column(db.Integer, db.ForeignKey('knowledge_points.id', ondelete='CASCADE'), nullable=False)
    is_relevant = db.Column(db.Boolean, default=True, comment='是否相关')
    weight = db.Column(db.Float, default=1.0, comment='知识点权重')
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关系
    question = db.relationship('Question', back_populates='q_matrix_items')
    knowledge_point = db.relationship('KnowledgePoint', back_populates='q_matrix_items')
    
    __table_args__ = (
        db.UniqueConstraint('question_id', 'knowledge_point_id', name='uk_question_knowledge_point'),
    )
    
    def __repr__(self):
        return f'<QMatrix Q:{self.question_id} KP:{self.knowledge_point_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'question_code': self.question.question_code if self.question else None,
            'kp_code': self.knowledge_point.kp_code if self.knowledge_point else None,
            'is_relevant': self.is_relevant,
            'weight': self.weight
        }


class Student(db.Model):
    """学生模型"""
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_code = db.Column(db.String(50), unique=True, nullable=False, comment='学生编号')
    name = db.Column(db.String(100), comment='学生姓名')
    grade = db.Column(db.String(20), comment='年级')
    class_name = db.Column(db.String(50), comment='班级')
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关系
    answer_records = db.relationship('AnswerRecord', back_populates='student', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Student {self.student_code}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_code': self.student_code,
            'name': self.name,
            'grade': self.grade,
            'class_name': self.class_name
        }


class AnswerRecord(db.Model):
    """答题记录模型"""
    __tablename__ = 'answer_records'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id', ondelete='CASCADE'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'), nullable=False)
    score = db.Column(db.Float, nullable=False, comment='得分（0-1之间）')
    is_correct = db.Column(db.Boolean, comment='是否正确')
    answer_time = db.Column(db.DateTime, default=datetime.now, comment='答题时间')
    time_spent = db.Column(db.Integer, comment='耗时（秒）')
    
    # 关系
    student = db.relationship('Student', back_populates='answer_records')
    question = db.relationship('Question', back_populates='answer_records')
    
    def __repr__(self):
        return f'<AnswerRecord S:{self.student_id} Q:{self.question_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_code': self.student.student_code if self.student else None,
            'question_code': self.question.question_code if self.question else None,
            'score': self.score,
            'is_correct': self.is_correct,
            'answer_time': self.answer_time.isoformat() if self.answer_time else None,
            'time_spent': self.time_spent
        }
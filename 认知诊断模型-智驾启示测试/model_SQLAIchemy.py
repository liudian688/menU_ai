from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class AnswerRecord(db.Model):
    """答题记录模型"""
    __tablename__ = 'answer_records'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id', ondelete='CASCADE'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'), nullable=False)
    score = db.Column(db.Float, nullable=False, comment='得分（0-1之间）')
    is_correct = db.Column(db.Boolean, comment='是否正确')
    
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
        }
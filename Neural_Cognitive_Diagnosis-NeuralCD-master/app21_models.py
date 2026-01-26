from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL='sqlite:///sqlite.db'

engine=create_engine(DATABASE_URL)
SessionLocal=sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base=declarative_base()

class User(Base):
    """
    用户表
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关系
    responses = relationship("UserResponse", back_populates="user")
    assessments = relationship("Assessment", back_populates="user")

class Skill(Base):
    """
    技能表 - 表示大类技能（如编程、数学等）
    """
    __tablename__ = "skills"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    description = Column(Text)
    category = Column(String(50))  # 如 STEM, 人文社科, 艺术等
    created_at = Column(DateTime, default=datetime.now)
    
    # 关系
    knowledge_points = relationship("KnowledgePoint", back_populates="skill")
    exercises = relationship("Exercise", back_populates="skill")

class KnowledgePoint(Base):
    """
    知识点表 - 技能下的具体知识点
    """
    __tablename__ = "knowledge_points"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False)
    difficulty_level = Column(Integer, default=1)  # 1-5等级
    importance_score = Column(Float, default=1.0)  # 重要性分数
    created_at = Column(DateTime, default=datetime.now)
    
    # 关系
    skill = relationship("Skill", back_populates="knowledge_points")
    exercises = relationship("Exercise", secondary="exercise_knowledge_assoc", back_populates="knowledge_points")
    user_responses = relationship("UserResponse", secondary="response_knowledge_assoc", back_populates="knowledge_points")

class Exercise(Base):
    """
    练习题/测试题表
    """
    __tablename__ = "exercises"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    content = Column(Text, nullable=False)  # 题目内容
    answer = Column(Text)  # 标准答案
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False)
    difficulty_level = Column(Integer, default=1)  # 1-5等级
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    skill = relationship("Skill", back_populates="exercises")
    knowledge_points = relationship("KnowledgePoint", secondary="exercise_knowledge_assoc", back_populates="exercises")
    responses = relationship("UserResponse", back_populates="exercise")

class ExerciseKnowledgeAssoc(Base):
    """
    练习题和知识点的多对多关联表
    """
    __tablename__ = "exercise_knowledge_assoc"
    
    exercise_id = Column(Integer, ForeignKey("exercises.id"), primary_key=True)
    knowledge_point_id = Column(Integer, ForeignKey("knowledge_points.id"), primary_key=True)

class ResponseKnowledgeAssoc(Base):
    """
    用户答题记录和知识点的多对多关联表
    """
    __tablename__ = "response_knowledge_assoc"
    
    response_id = Column(Integer, ForeignKey("user_responses.id"), primary_key=True)
    knowledge_point_id = Column(Integer, ForeignKey("knowledge_points.id"), primary_key=True)

class UserResponse(Base):
    """
    用户答题记录表
    """
    __tablename__ = "user_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exercise_id = Column(Integer, ForeignKey("exercises.id"), nullable=False)
    score = Column(Float, nullable=False)  # 0.0-1.0之间的得分
    is_correct = Column(Boolean, nullable=False)  # 是否正确
    response_time = Column(Float)  # 答题用时（秒）
    response_content = Column(Text)  # 用户的回答内容
    timestamp = Column(DateTime, default=datetime.utcnow)  # 答题时间
    
    # 关系
    user = relationship("User", back_populates="responses")
    exercise = relationship("Exercise", back_populates="responses")
    knowledge_points = relationship("KnowledgePoint", secondary="response_knowledge_assoc", back_populates="user_responses")

class Assessment(Base):
    """
    评估结果表
    """
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assessment_date = Column(DateTime, default=datetime.utcnow)
    overall_score = Column(Float, nullable=False)  # 总体评分
    weaknesses_json = Column(Text)  # 薄弱点信息（JSON格式）
    strengths_json = Column(Text)  # 优势点信息（JSON格式）
    knowledge_vector_json = Column(Text)  # 知识向量（JSON格式）
    similar_users_json = Column(Text)  # 相似用户信息（JSON格式）
    analysis_summary_json = Column(Text)  # 分析摘要（JSON格式）
    
    # 关系
    user = relationship("User", back_populates="assessments")

class LearningPath(Base):
    """
    学习路径表 - 为用户推荐的学习路径
    """
    __tablename__ = "learning_paths"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False)
    path_name = Column(String(200), nullable=False)
    path_description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
    
    # 关系
    user = relationship("User")
    skill = relationship("Skill")


def init_database():
    """
    初始化数据库，创建所有表
    """
    Base.metadata.create_all(bind=engine)
    print("数据库表创建完成")

def get_db():
    """
    获取数据库会话的依赖函数
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 数据库操作辅助函数
def create_sample_data(db):
    """
    创建示例数据
    """
    from sqlalchemy.exc import IntegrityError
    
    # 创建示例技能
    skills_data = [
        {"name": "编程", "description": "编程技能", "category": "STEM"},
        {"name": "数学", "description": "数学技能", "category": "STEM"},
        {"name": "英语", "description": "英语语言技能", "category": "语言"}
    ]
    
    for skill_data in skills_data:
        skill = db.query(Skill).filter(Skill.name == skill_data["name"]).first()
        if not skill:
            skill = Skill(**skill_data)
            db.add(skill)
    
    db.commit()
    
    # 创建示例知识点
    knowledge_data = [
        {"name": "变量类型", "skill_id": 1, "difficulty_level": 1},
        {"name": "控制结构", "skill_id": 1, "difficulty_level": 2},
        {"name": "算法基础", "skill_id": 1, "difficulty_level": 3},
        {"name": "代数", "skill_id": 2, "difficulty_level": 2},
        {"name": "几何", "skill_id": 2, "difficulty_level": 2},
        {"name": "微积分", "skill_id": 2, "difficulty_level": 4},
        {"name": "语法", "skill_id": 3, "difficulty_level": 1},
        {"name": "词汇", "skill_id": 3, "difficulty_level": 1}
    ]
    
    for knowledge_data_item in knowledge_data:
        knowledge = db.query(KnowledgePoint).filter(
            KnowledgePoint.name == knowledge_data_item["name"]
        ).first()
        if not knowledge:
            knowledge = KnowledgePoint(**knowledge_data_item)
            db.add(knowledge)
    
    db.commit()
    
    # 创建示例练习题
    exercises_data = [
        {
            "title": "变量类型基础题",
            "content": "在Python中，下面哪个是字符串类型？",
            "answer": "str",
            "skill_id": 1,
            "difficulty_level": 1
        },
        {
            "title": "简单代数题",
            "content": "解方程: 2x + 3 = 7",
            "answer": "x = 2",
            "skill_id": 2,
            "difficulty_level": 2
        }
    ]
    
    for exercise_data in exercises_data:
        exercise = db.query(Exercise).filter(
            Exercise.title == exercise_data["title"]
        ).first()
        if not exercise:
            exercise = Exercise(**exercise_data)
            db.add(exercise)
    
    db.commit()
    
    # 创建示例用户
    users_data = [
        {"username": "student1", "email": "student1@example.com", "full_name": "学生1"},
        {"username": "student2", "email": "student2@example.com", "full_name": "学生2"}
    ]
    
    for user_data in users_data:
        user = db.query(User).filter(User.username == user_data["username"]).first()
        if not user:
            user = User(**user_data)
            db.add(user)
    
    db.commit()
    
    print("示例数据创建完成")

if __name__ == "__main__":
    init_database()
    
    # 创建示例数据
    db = SessionLocal()
    try:
        create_sample_data(db)
    except Exception as e:
        print(f"创建示例数据时出错: {e}")
        db.rollback()
    finally:
        db.close()
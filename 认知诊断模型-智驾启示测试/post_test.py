from flask import Flask, jsonify, request
from db_operations import NeuralCDDB
from model_SQLAIchemy import db

def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    
    # 配置数据库
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:your_password@localhost/neuralcd_db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ECHO'] = True  # 打印SQL语句，调试用
     
    # 初始化数据库
    db.init_app(app)
    
    return app

app = create_app()
neuralcd_db = NeuralCDDB(app)


# ==================== API路由 ====================

@app.route('/api/answer', methods=['POST'])
def add_answer():
    """添加答题记录"""
    data = request.json
    record = neuralcd_db.add_answer_record(
        student_code=data['student_code'],
        question_code=data['question_code'],
        score=data['score'],
        time_spent=data.get('time_spent')
    )
    if record:
        return jsonify(record.to_dict()), 201
    return jsonify({'error': '添加失败'}), 400


# ==================== 初始化数据库 ====================

@app.cli.command('init-db')
def init_db_command():
    """初始化数据库（创建表）"""
    with app.app_context():
        db.create_all()
    print('数据库初始化完成')

@app.cli.command('drop-db')
def drop_db_command():
    """删除所有表（谨慎使用）"""
    with app.app_context():
        db.drop_all()
    print('所有表已删除')


# ==================== 示例数据导入 ====================

def import_sample_data():
    """导入示例数据"""
    with app.app_context():  
        # 添加答题记录
        neuralcd_db.add_answer_record('S001', 'Q001', 1.0)
        neuralcd_db.add_answer_record('S001', 'Q002', 0.0)
        neuralcd_db.add_answer_record('S002', 'Q001', 1.0)
        neuralcd_db.add_answer_record('S002', 'Q002', 1.0)
        
        print("示例数据导入完成")


# ==================== 主函数 ====================

if __name__ == '__main__':
    # 确保在应用上下文中
    with app.app_context():
        # 创建表
        db.create_all()
        print("数据库表创建成功")
        
        # 导入示例数据（可选）
        import_sample_data()
        
        # 打印统计信息
        stats = neuralcd_db.get_statistics()
        print(f"统计信息: {stats}")
        
        # 获取并打印Q矩阵
        q_matrix = neuralcd_db.get_q_matrix_dataframe()
        print("\nQ矩阵:")
        print(q_matrix)
    
    # 启动Fl应用
    app.run(debug=True, host='0.0.0.0', port=5000)
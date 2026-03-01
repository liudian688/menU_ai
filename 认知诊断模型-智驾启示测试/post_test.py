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

@app.route('/api/knowledge_points', methods=['POST'])
def get_knowledge_points():
    """获取所有知识点"""
    kps = neuralcd_db.get_all_knowledge_points()
    return jsonify([kp.to_dict() for kp in kps])

@app.route('/api/knowledge_points', methods=['POST'])
def add_knowledge_point():
    """添加知识点"""
    data = request.json
    kp = neuralcd_db.add_knowledge_point(
        kp_code=data['kp_code'],
        kp_name=data['kp_name'],
        description=data.get('description', ''),
        category=data.get('category', ''),
        difficulty_level=data.get('difficulty_level', 0.5)
    )
    return jsonify(kp.to_dict()), 201

@app.route('/api/questions', methods=['POST'])
def add_question():
    """添加题目"""
    data = request.json
    q = neuralcd_db.add_question(
        question_code=data['question_code'],
        question_name=data['question_name'],
        description=data.get('description', ''),
        category=data.get('category', ''),
        difficulty_level=data.get('difficulty_level', 0.5)
    )
    return jsonify(q.to_dict()), 201

@app.route('/api/q_matrix', methods=['POST'])
def add_q_matrix():
    """添加Q矩阵条目"""
    data = request.json
    item = neuralcd_db.add_q_matrix_item(
        question_code=data['question_code'],
        kp_code=data['kp_code'],
        is_relevant=data.get('is_relevant', True),
        weight=data.get('weight', 1.0)
    )
    return jsonify(item.to_dict()), 201

@app.route('/api/q_matrix', methods=['POST'])
def get_q_matrix():
    """获取Q矩阵"""
    q_matrix_df = neuralcd_db.get_q_matrix_dataframe()
    return jsonify({
        'questions': q_matrix_df.index.tolist(),
        'knowledge_points': q_matrix_df.columns.tolist(),
        'matrix': q_matrix_df.values.tolist()
    })

@app.route('/api/question/<question_code>/knowledge_points', methods=['POST'])
def get_question_kps(question_code):
    """获取题目涉及的知识点"""
    kps = neuralcd_db.get_question_knowledge_points(question_code)
    return jsonify({'question_code': question_code, 'knowledge_points': kps})

@app.route('/api/knowledge_point/<kp_code>/questions', methods=['POST'])
def get_kp_questions(kp_code):
    """获取考察某个知识点的题目"""
    questions = neuralcd_db.get_knowledge_point_questions(kp_code)
    return jsonify({'kp_code': kp_code, 'questions': questions})

@app.route('/api/statistics', methods=['POST'])
def get_statistics():
    """获取统计信息"""
    stats = neuralcd_db.get_statistics()
    return jsonify(stats)

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


# ==================== 主函数 ====================

if __name__ == '__main__':
    print("启动认知诊断服务...")
    print("可用接口:")
    print("  - POST /api/evaluate: 接收用户答题记录并返回诊断结果")
    print("  - GET  /api/health:   健康检查接口")
    print("")
    print("示例请求数据格式:")
    print("""
    {
        "user_answers": [
            {
                "exer_id": 1,
                "score": 1,
                "knowledge_code": [1, 2]
            },
            {
                "exer_id": 2,
                "score": 0,
                "knowledge_code": [2, 3]
            }
        ],
        "domain_knowledge_codes": [1, 2, 3],
        "model_epoch": 5,
        "user_id": "123"
    }
    """)
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)
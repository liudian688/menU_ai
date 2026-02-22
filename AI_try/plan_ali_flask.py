from flask import Flask, request, jsonify
import config
from plan_ali import AliAgentService
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate


app = Flask(__name__)
app.config.from_object(config)
db = SQLAlchemy(app)

#=======================================================
#数据库表
class Bubble(db.Model):
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    bubble_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    password = db.Column(db.String(120), nullable=False, default="123456")

#=======================================================

@app.route('/plan', methods=['POST'])
def plan():
    json_data = request.get_json()
    if json_data is None:
        return jsonify({'result': False, 'message': 'json is None'})
    bubble_id = json_data.get('bubble_id')
    if bubble_id is None:
        return jsonify({'result': False, 'message': 'bubble_id is None'})
    
    
    # 调用 AliAgentService 处理用户输入
    response = AliAgentService.process_questions(user_id, question)
    
    return jsonify({'result': True, 'message': "successfully"})

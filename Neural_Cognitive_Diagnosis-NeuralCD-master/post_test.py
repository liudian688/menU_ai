import flask
from flask import request, jsonify
from flask_sqlalchemy import SQLAlchemy
import diagnosis_system

app = flask.Flask(__name__)
db = SQLAlchemy(app)

class User(db.Model):
    user_id = db.Column(db.Integer, unique=True, nullable=False)
    field = db.Column(db.String(80), nullable=False)
    skill = db.Column(db.String(80), nullable=False)
    assessment = db.Column(db.String(80), nullable=False)

@app.route('/post_test', methods=['POST'])
def post_test():
    data = request.get_json()
    user_id = data['user_id']
    field = data['field']
    message = data['message']

    for item in message:
        skill = item['skill']
        assessment = item['assessment']

    user = User(user_id=user_id, field=field, skill=skill, assessment=assessment)
    db.session.add(user)
    db.session.commit()

    return jsonify(data)

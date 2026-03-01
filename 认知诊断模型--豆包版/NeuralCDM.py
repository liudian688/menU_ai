import torch
import torch.nn as nn
import torch.nn.functional as F

# 完全对应原论文的NeuralCDM模型
class NeuralCDM(nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num, hidden_dim=64):
        """
        初始化模型，参数说明（小白必看）：
        student_num: 训练集里的学生总数（冷启动用模拟数据的话随便设，比如1000）
        exercise_num: 这个领域的题目总数（比如Python领域有200道题，就填200）
        knowledge_num: 这个领域的知识点总数K（比如Python拆了30个知识点，就填30）
        hidden_dim: 神经网络隐藏层维度，小白不用改，默认64就行
        """
        super(NeuralCDM, self).__init__()
        self.knowledge_num = knowledge_num

        # ---------------------- 对应论文3.3.1 学生因子模块 ----------------------
        # 对应论文公式(2)：学生知识状态矩阵，每一行是一个学生的知识点掌握度向量
        # 最终输出会过sigmoid，保证取值在0-1之间，0=完全没掌握，1=完全掌握
        self.student_emb = nn.Embedding(student_num, knowledge_num)

        # ---------------------- 对应论文3.3.2 习题因子模块 ----------------------
        # 对应论文公式(4)：习题的知识点难度矩阵，每一行是一道题的每个知识点的难度
        self.exercise_diff_emb = nn.Embedding(exercise_num, knowledge_num)
        # 对应论文公式(5)：习题的区分度，每一个值对应一道题的区分能力
        self.exercise_disc_emb = nn.Embedding(exercise_num, 1)

        # ---------------------- 对应论文3.3.3 交互函数模块 ----------------------
        # 对应论文单调性假设：全连接层权重必须非负，保证知识点掌握度越高，答对概率越高
        # 第一层全连接层
        self.fc1 = nn.Linear(knowledge_num, hidden_dim)
        # 第二层全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层，输出答对概率
        self.fc3 = nn.Linear(hidden_dim, 1)

        # 初始化参数，小白不用改
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, student_id, exercise_id, q_matrix):
        """
        模型前向传播，对应论文的完整计算流程
        参数说明：
        student_id: 学生的ID
        exercise_id: 题目的ID
        q_matrix: 论文里的Q矩阵，shape是(题目总数, 知识点总数)，1=题目考察这个知识点，0=不考察
        """
        # 1. 取出学生的知识状态向量 h_s，对应论文公式(2)
        h_s = torch.sigmoid(self.student_emb(student_id))  # 过sigmoid，保证0-1之间
        # 2. 取出习题的知识点难度向量 h_diff，对应论文公式(4)
        h_diff = torch.sigmoid(self.exercise_diff_emb(exercise_id))
        # 3. 取出习题的区分度 h_disc，对应论文公式(5)
        h_disc = torch.sigmoid(self.exercise_disc_emb(exercise_id))
        # 4. 取出当前题目对应的知识点关联向量 Q_e
        q_e = q_matrix[exercise_id]

        # ---------------------- 对应论文公式(6) 交互层第一层计算 ----------------------
        # 公式：x = Q_e ∘ (h_s - h_diff) * h_disc
        # ∘ 是按元素相乘，只计算题目考察的知识点，没考察的直接置0
        x = q_e * (h_s - h_diff) * h_disc

        # ---------------------- 对应论文公式(7)(8)(9) 全连接层计算 ----------------------
        # 约束全连接层权重非负，保证单调性假设（核心！）
        self.fc1.weight.data.clamp_(min=0)
        self.fc2.weight.data.clamp_(min=0)
        self.fc3.weight.data.clamp_(min=0)

        # 过两层全连接层+激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出答对概率，过sigmoid保证0-1之间，对应论文公式(9)
        y_pred = torch.sigmoid(self.fc3(x))

        return y_pred, h_s  # 返回答对概率预测值，和学生的知识点掌握度向量
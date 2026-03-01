import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from NeuralCDM import NeuralCDM
import os

# ---------------------- 小白只需要改这里的参数！其他都不用动 ----------------------
# 领域配置（比如你现在做Python入门，就填对应的参数）
DOMAIN_NAME = "python入门"  # 领域名称，用来区分不同技能的模型
STUDENT_NUM = 1000  # 训练用的学生数，冷启动模拟数据随便设，真实数据就填实际学生数
EXERCISE_NUM = 200  # 这个领域的题目总数
KNOWLEDGE_NUM = 50  # 这个领域的知识点总数
# 训练参数，小白不用改
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 没有GPU用CPU也能跑
# -----------------------------------------------------------------------------------

# ********************** 冷启动：模拟数据生成（没有真实数据就用这个）**********************
def generate_sim_data():
    """
    生成模拟的训练数据，小白冷启动专用
    生成3个核心文件：
    1. q_matrix.csv：Q矩阵，题目-知识点对应表
    2. train_log.csv：学生答题记录，学生ID-题目ID-答对/答错
    3. 模型保存文件夹
    """
    # 1. 生成Q矩阵：每道题随机考察1-3个知识点，符合真实场景
    q_matrix = np.zeros((EXERCISE_NUM, KNOWLEDGE_NUM))
    for exer_id in range(EXERCISE_NUM):
        # 随机选1-3个知识点作为这道题的考点
        knowledges = np.random.choice(KNOWLEDGE_NUM, size=np.random.randint(1, 4), replace=False)
        q_matrix[exer_id, knowledges] = 1
    # 保存Q矩阵，后续真实数据替换这个文件就行
    pd.DataFrame(q_matrix).to_csv(f"{DOMAIN_NAME}_q_matrix.csv", index=False, header=False)

    # 2. 生成模拟答题记录
    train_data = []
    # 随机生成学生的知识点掌握度
    student_mastery = np.random.rand(STUDENT_NUM, KNOWLEDGE_NUM)
    for student_id in range(STUDENT_NUM):
        # 每个学生随机做20-50道题
        exercise_ids = np.random.choice(EXERCISE_NUM, size=np.random.randint(20, 51), replace=False)
        for exer_id in exercise_ids:
            # 计算学生答对这道题的概率：掌握的知识点越多，概率越高
            related_knowledge = q_matrix[exer_id] == 1
            mastery_mean = student_mastery[student_id, related_knowledge].mean()
            correct_prob = 1 / (1 + np.exp(-1.7 * (mastery_mean - 0.5)))  # 对应IRT模型的逻辑
            is_correct = 1 if np.random.rand() < correct_prob else 0
            train_data.append([student_id, exer_id, is_correct])
    # 保存训练数据，后续真实数据替换这个文件就行
    pd.DataFrame(train_data, columns=["student_id", "exercise_id", "is_correct"]).to_csv(f"{DOMAIN_NAME}_train_log.csv", index=False)

    # 创建模型保存文件夹
    if not os.path.exists(f"{DOMAIN_NAME}_model"):
        os.makedirs(f"{DOMAIN_NAME}_model")

    print(f"【{DOMAIN_NAME}】模拟数据生成完成！")
    return q_matrix, train_data

# ********************** 模型训练主函数 **********************
def train_model():
    # 1. 生成/加载数据
    if not os.path.exists(f"{DOMAIN_NAME}_q_matrix.csv") or not os.path.exists(f"{DOMAIN_NAME}_train_log.csv"):
        q_matrix, train_data = generate_sim_data()
    else:
        q_matrix = pd.read_csv(f"{DOMAIN_NAME}_q_matrix.csv", header=None).values
        train_data = pd.read_csv(f"{DOMAIN_NAME}_train_log.csv").values

    # 转成PyTorch张量
    q_matrix = torch.FloatTensor(q_matrix).to(DEVICE)
    train_data = torch.LongTensor(train_data).to(DEVICE)

    # 2. 初始化模型、损失函数、优化器
    model = NeuralCDM(
        student_num=STUDENT_NUM,
        exercise_num=EXERCISE_NUM,
        knowledge_num=KNOWLEDGE_NUM
    ).to(DEVICE)
    criterion = nn.BCELoss()  # 二元交叉熵损失，对应0/1的答题对错
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. 训练循环
    print(f"【{DOMAIN_NAME}】开始训练模型！")
    for epoch in range(EPOCHS):
        model.train()
        # 打乱数据
        shuffle_idx = torch.randperm(len(train_data))
        train_data_shuffled = train_data[shuffle_idx]
        total_loss = 0

        # 分批训练
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data_shuffled[i:i+BATCH_SIZE]
            student_ids = batch[:, 0]
            exercise_ids = batch[:, 1]
            true_labels = batch[:, 2].float().unsqueeze(1)

            # 前向传播
            pred_labels, _ = model(student_ids, exercise_ids, q_matrix)
            # 计算损失
            loss = criterion(pred_labels, true_labels)
            total_loss += loss.item()

            # 反向传播优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印训练日志
        print(f"Epoch {epoch+1}/{EPOCHS}, 平均损失: {total_loss/len(train_data):.6f}")

    # 4. 保存训练好的模型和配置
    if not os.path.exists(f"{DOMAIN_NAME}_model"):
        os.makedirs(f"{DOMAIN_NAME}_model")
    torch.save(model.state_dict(), f"{DOMAIN_NAME}_model/model.pth")
    # 保存模型配置，后续诊断用
    model_config = {
        "student_num": STUDENT_NUM,
        "exercise_num": EXERCISE_NUM,
        "knowledge_num": KNOWLEDGE_NUM,
        "domain_name": DOMAIN_NAME
    }
    np.save(f"{DOMAIN_NAME}_model/config.npy", model_config)
    print(f"【{DOMAIN_NAME}】模型训练完成！已保存到 {DOMAIN_NAME}_model/ 文件夹")

if __name__ == "__main__":
    train_model()
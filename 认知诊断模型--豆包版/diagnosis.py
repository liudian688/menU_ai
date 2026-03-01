import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from NeuralCDM import NeuralCDM

# ********************** 新用户诊断核心函数 **********************
def new_user_diagnosis(domain_name, new_user_answer_log):
    """
    新用户诊断，完全不用重训模型！
    参数说明：
    domain_name: 要诊断的领域名称，比如"python入门"，和训练时的名称一致
    new_user_answer_log: 新用户的答题记录，格式是二维列表：[[题目ID, 是否答对(1/0)], ...]
    返回值：
    mastery_result: 字典，key=知识点ID，value=掌握度(0-1)
    """
    # 1. 加载训练好的模型配置和权重
    model_config = np.load(f"{domain_name}_model/config.npy", allow_pickle=True).item()
    q_matrix = pd.read_csv(f"{domain_name}_q_matrix.csv", header=None).values
    q_matrix = torch.FloatTensor(q_matrix)

    # 2. 初始化模型，加载预训练权重
    model = NeuralCDM(
        student_num=model_config["student_num"],
        exercise_num=model_config["exercise_num"],
        knowledge_num=model_config["knowledge_num"]
    )
    model.load_state_dict(torch.load(f"{domain_name}_model/model.pth", map_location="cpu"))
    model.eval()  # 固定模型所有参数！！！核心！！！不改动预训练的任何内容

    # 3. 提取新用户的答题数据
    exercise_ids = torch.LongTensor([log[0] for log in new_user_answer_log])
    true_labels = torch.FloatTensor([log[1] for log in new_user_answer_log]).unsqueeze(1)

    # ********************** 核心：只优化新用户的知识状态向量 **********************
    # 初始化新用户的知识状态向量，初始值0.5（中等掌握）
    new_user_mastery = torch.nn.Parameter(torch.sigmoid(torch.randn(1, model_config["knowledge_num"])))
    # 只优化这个向量，其他所有模型参数都固定
    optimizer = optim.Adam([new_user_mastery], lr=0.01)
    criterion = torch.nn.BCELoss()

    # 迭代优化，100次就够，毫秒级完成
    for step in range(100):
        optimizer.zero_grad()
        # 固定模型的其他参数，只用新用户的知识状态向量计算
        h_s = torch.sigmoid(new_user_mastery)
        h_diff = torch.sigmoid(model.exercise_diff_emb(exercise_ids))
        h_disc = torch.sigmoid(model.exercise_disc_emb(exercise_ids))
        q_e = q_matrix[exercise_ids]

        # 计算预测的答对概率
        x = q_e * (h_s - h_diff) * h_disc
        model.fc1.weight.data.clamp_(min=0)
        model.fc2.weight.data.clamp_(min=0)
        model.fc3.weight.data.clamp_(min=0)
        x = torch.relu(model.fc1(x))
        x = torch.relu(model.fc2(x))
        pred_labels = torch.sigmoid(model.fc3(x))

        # 计算损失，只优化新用户的知识状态
        loss = criterion(pred_labels, true_labels)
        loss.backward()
        optimizer.step()

    # 4. 整理诊断结果
    final_mastery = torch.sigmoid(new_user_mastery).detach().numpy()[0]
    mastery_result = {f"知识点{k}": round(float(final_mastery[k]), 4) for k in range(model_config["knowledge_num"])}

    print(f"【{domain_name}】新用户诊断完成！")
    return mastery_result

# ---------------------- 测试新用户诊断 ----------------------
if __name__ == "__main__":
    from test import generate_test_answer_log
    # 模拟新用户的答题记录：[[题目ID, 答对1/答错0], ...]
    test_answer_log = generate_test_answer_log()
    print(test_answer_log)

    # 调用诊断函数
    result = new_user_diagnosis(domain_name="python入门", new_user_answer_log=test_answer_log)
    # 打印诊断结果
    for knowledge, mastery in result.items():
        print(f"{knowledge} 掌握度：{mastery}")
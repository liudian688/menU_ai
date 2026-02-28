import torch
import torch.nn as nn


class KnowledgeBase(object):
    '''
    用户个性化记忆库
    存储用户的知识点掌握情况，完全独立于模型参数
    '''
    def __init__(self, knowledge_dim):
        self.knowledge_dim = knowledge_dim
        self.user_knowledge_states = {}  # user_id -> knowledge_state
        self.user_history = {}  # user_id -> list of (exer_id, score, knowledge_code)
    
    def get_user_state(self, user_id):
        '''获取用户的知识点掌握状态'''  
        if user_id not in self.user_knowledge_states:
            # 新用户初始化默认状态
            self.user_knowledge_states[user_id] = torch.zeros(self.knowledge_dim)
        return self.user_knowledge_states[user_id]
    
    def update_user_state(self, user_id, exer_id, score, knowledge_code, net=None):
        '''根据答题情况使用模型反向推断用户知识掌握状态'''  
        if user_id not in self.user_knowledge_states:
            self.user_knowledge_states[user_id] = torch.zeros(self.knowledge_dim)
        
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        # 记录答题历史
        self.user_history[user_id].append((exer_id, score, knowledge_code))
        
        # 如果有模型，使用认知诊断模型进行反向推断
        if net is not None:
            self._update_with_model(user_id, exer_id, score, knowledge_code, net)
        else:
            # 如果没有模型，使用简单的启发式更新
            self._update_heuristic(user_id, exer_id, score, knowledge_code)
    
    def _update_heuristic(self, user_id, exer_id, score, knowledge_code):
        '''使用启发式方法更新知识状态（简单版本）'''
        if score == 1:
            # 答对了，增强对应知识点的掌握度
            for kn in knowledge_code:
                self.user_knowledge_states[user_id][kn - 1] = min(1.0, self.user_knowledge_states[user_id][kn - 1] + 0.1)
        else:
            # 答错了，减弱对应知识点的掌握度
            for kn in knowledge_code:
                self.user_knowledge_states[user_id][kn - 1] = max(0.0, self.user_knowledge_states[user_id][kn - 1] - 0.05)
    
    def _update_with_model(self, user_id, exer_id, score, knowledge_code, net):
        '''使用认知诊断模型进行知识状态反向推断'''
        # 获取当前用户知识状态
        current_state = self.user_knowledge_states[user_id].clone()
        
        # 构建知识点嵌入
        kn_emb = torch.zeros(self.knowledge_dim)
        for kn in knowledge_code:
            if 1 <= kn <= self.knowledge_dim:
                kn_emb[kn-1] = 1.0
        
        # 使用模型进行前向预测
        with torch.no_grad():
            predicted_score = net(
                current_state.unsqueeze(0),
                torch.tensor([exer_id]),
                kn_emb.unsqueeze(0)
            ).item()
        
        # 计算预测与实际表现的差异
        prediction_error = abs(predicted_score - score)
        
        # 基于预测误差调整知识状态
        # 如果模型预测与实际表现差异较大，说明当前知识状态估计不准确
        # 需要根据实际表现进行更大程度的调整
        
        adjustment_factor = 0.2 + prediction_error * 0.3  # 误差越大，调整幅度越大
        
        if score == 1:
            # 答对了：增强相关知识点，但根据预测准确性调整幅度
            for kn in knowledge_code:
                if 1 <= kn <= self.knowledge_dim:
                    # 如果模型预测正确率较低但实际答对，说明掌握度可能被低估
                    if predicted_score < 0.5:
                        adjustment = adjustment_factor * (1.0 - current_state[kn-1])
                    else:
                        adjustment = adjustment_factor * 0.1
                    self.user_knowledge_states[user_id][kn-1] = min(1.0, current_state[kn-1] + adjustment)
        else:
            # 答错了：减弱相关知识点，但根据预测准确性调整幅度
            for kn in knowledge_code:
                if 1 <= kn <= self.knowledge_dim:
                    # 如果模型预测正确率较高但实际答错，说明掌握度可能被高估
                    if predicted_score > 0.5:
                        adjustment = adjustment_factor * current_state[kn-1]
                    else:
                        adjustment = adjustment_factor * 0.05
                    self.user_knowledge_states[user_id][kn-1] = max(0.0, current_state[kn-1] - adjustment)


class Net(nn.Module):
    '''
    NeuralCDM - 通用预训练底座
    固定主干参数，只负责通用判断能力
    '''
    def __init__(self, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # network structure - 只保留题目相关参数，移除学生嵌入
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_emb, exer_id, kn_emb):
        '''
        :param stu_emb: FloatTensor, 用户知识点掌握状态
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
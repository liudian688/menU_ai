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
        self.knowledge_statistics = {}  # user_id -> {knowledge_code: {'correct': int, 'total': int}}
    
    def get_user_state(self, user_id):
        '''获取用户的知识点掌握状态'''  
        if user_id not in self.user_knowledge_states:
            # 新用户初始化默认状态
            self.user_knowledge_states[user_id] = torch.zeros(self.knowledge_dim)
        return self.user_knowledge_states[user_id]
    
    def update_user_state(self, user_id, exer_id, score, knowledge_code):
        '''根据答题情况更新用户状态'''  
        if user_id not in self.user_knowledge_states:
            self.user_knowledge_states[user_id] = torch.zeros(self.knowledge_dim)
        
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        if user_id not in self.knowledge_statistics:
            self.knowledge_statistics[user_id] = {}
        
        # 记录答题历史
        self.user_history[user_id].append((exer_id, score, knowledge_code))
        
        # 更新知识点统计信息
        for kn in knowledge_code:
            if kn not in self.knowledge_statistics[user_id]:
                self.knowledge_statistics[user_id][kn] = {'correct': 0, 'total': 0}
            
            self.knowledge_statistics[user_id][kn]['total'] += 1
            if score == 1:
                self.knowledge_statistics[user_id][kn]['correct'] += 1
        
        # 基于答题次数和正确率重新计算知识点掌握度
        for kn in knowledge_code:
            if kn in self.knowledge_statistics[user_id]:
                stats = self.knowledge_statistics[user_id][kn]
                if stats['total'] > 0:
                    # 计算正确率作为掌握度
                    mastery_level = stats['correct'] / stats['total']
                    self.user_knowledge_states[user_id][kn - 1] = mastery_level
                else:
                    self.user_knowledge_states[user_id][kn - 1] = 0.0


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
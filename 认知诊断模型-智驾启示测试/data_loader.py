import json
import torch
import os


class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self):
        self.batch_size = 32
        self.ptr = 0
        self.data = []

        data_path = os.path.abspath(__file__)
        data_file = os.path.join(os.path.dirname(data_path), 'database_cx/data/math_train_set.json')
        config_path = os.path.abspath(__file__)
        config1_file = os.path.join(os.path.dirname(config_path), 'config1.txt')
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config1_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='validation'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if d_type == 'validation':
            file_path = os.path.abspath(__file__)
            data_file = os.path.join(os.path.dirname(file_path), 'database_cx/data/math_val_set.json')
        else:
            file_path = os.path.abspath(__file__)
            data_file = os.path.join(os.path.dirname(file_path), 'database_cx/data/math_test_set.json')
        config_path = os.path.abspath(__file__)
        config1_file = os.path.join(os.path.dirname(config_path), 'config1.txt')
        
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config1_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

import torch
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import ValTestDataLoader
from model import Net, KnowledgeBase


# can be changed according to config.txt
exer_n = 500
knowledge_n = 50
student_n = 1000


# 知识基地实例
knowledge_base = None


def test(epoch):
    global knowledge_base
    data_loader = ValTestDataLoader('test')
    net = Net(exer_n, knowledge_n)
    knowledge_base = KnowledgeBase(knowledge_n)
    device = torch.device('cpu')
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, 'model/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        # 获取批次数据
        batch_data = data_loader.data[data_loader.ptr]
        user_id = batch_data['user_id']
        logs = batch_data['logs']
        
        # 准备输入数据
        input_exer_ids = []
        input_knowledge_embs = []
        labels = []
        
        for log in logs:
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * knowledge_n
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            labels.append(log['score'])
        
        # 从知识基地获取用户状态
        stu_emb = knowledge_base.get_user_state(user_id).unsqueeze(0).repeat(len(input_exer_ids), 1).to(device)
        input_exer_ids = torch.LongTensor(input_exer_ids).to(device)
        input_knowledge_embs = torch.Tensor(input_knowledge_embs).to(device)
        labels = torch.LongTensor(labels).to(device)
        
        out_put = net(stu_emb, input_exer_ids, input_knowledge_embs)
        out_put = out_put.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and out_put[i] > 0.5) or (labels[i] == 0 and out_put[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += out_put.tolist()
        label_all += labels.tolist()
        
        data_loader.ptr += 1

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
    with open('result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch, accuracy, rmse, auc))


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def get_status():
    '''
    An example of getting student's knowledge status from knowledge base
    :return:
    '''
    global knowledge_base
    knowledge_base = KnowledgeBase(knowledge_n)
    with open('result/student_stat.txt', 'w', encoding='utf8') as output_file:
        for user_id in range(1, student_n + 1):
            # get knowledge status of student with user_id
            status = knowledge_base.get_user_state(user_id).tolist()
            output_file.write(str(status) + '\n')


def get_exer_params():
    '''
    An example of getting exercise's parameters (knowledge difficulty and exercise discrimination)
    :return:
    '''
    net = Net(exer_n, knowledge_n)
    load_snapshot(net, 'model/model_epoch12')    # load model
    net.eval()
    exer_params_dict = {}
    for exer_id in range(exer_n):
        # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
        k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id]))
        exer_params_dict[exer_id + 1] = (k_difficulty.tolist()[0], e_discrimination.tolist()[0])
    with open('result/exer_params.txt', 'w', encoding='utf8') as o_f:
        o_f.write(str(exer_params_dict))


if __name__ == '__main__':
    if (len(sys.argv) != 2) or (not sys.argv[1].isdigit()):
        print('command:\n\tpython predict.py {epoch}\nexample:\n\tpython predict.py 70')
        exit(1)

    # global student_n, exer_n, knowledge_n
    with open('config1.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    test(int(sys.argv[1]))

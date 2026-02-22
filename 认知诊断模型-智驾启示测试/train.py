import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net, KnowledgeBase


# can be changed according to config.txt
exer_n = 17746
knowledge_n = 123
student_n = 4163
# can be changed according to command parameter
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5


# 全局知识基地实例
knowledge_base = None


def train():
    global knowledge_base
    data_loader = TrainDataLoader()
    net = Net(exer_n, knowledge_n)
    knowledge_base = KnowledgeBase(knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            if input_stu_ids is None:
                break
            
            # 从数据文件中获取原始数据，用于更新知识基地
            batch_size = len(input_stu_ids)
            batch_data = data_loader.data[data_loader.ptr - batch_size : data_loader.ptr]
            
            # 准备用户状态
            stu_embs = []
            for i, stu_id in enumerate(input_stu_ids):
                # 从知识基地获取用户状态
                user_id = stu_id.item() + 1  # 恢复原始用户ID
                stu_emb = knowledge_base.get_user_state(user_id)
                stu_embs.append(stu_emb)
            stu_embs = torch.stack(stu_embs).to(device)
            
            input_exer_ids, input_knowledge_embs, labels = input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(stu_embs, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            # grad_penalty = 0
            loss = loss_function(torch.log(output), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0
            
            # 更新知识基地中的用户状态
            for i, log in enumerate(batch_data):
                user_id = log['user_id']
                exer_id = log['exer_id']
                score = log['score']
                knowledge_code = log['knowledge_code']
                knowledge_base.update_user_state(user_id, exer_id, score, knowledge_code)

        # validate and save current model every epoch
        rmse, auc = validate(net, epoch)
        save_snapshot(net, 'model/model_epoch' + str(epoch + 1))


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        # 获取批次数据
        batch_data = data_loader.data[data_loader.ptr]
        user_id = batch_data['user_id']
        logs = batch_data['logs']
        
        # 准备输入数据
        input_stu_ids = []
        input_exer_ids = []
        input_knowledge_embs = []
        labels = []
        
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * knowledge_n
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            labels.append(log['score'])
        
        # 从知识基地获取用户状态
        stu_emb = knowledge_base.get_user_state(user_id).unsqueeze(0).repeat(len(input_stu_ids), 1).to(device)
        input_exer_ids = torch.LongTensor(input_exer_ids).to(device)
        input_knowledge_embs = torch.Tensor(input_knowledge_embs).to(device)
        labels = torch.LongTensor(labels).to(device)
        
        output = net.forward(stu_emb, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()
        
        data_loader.ptr += 1

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    # global student_n, exer_n, knowledge_n, device
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    train()

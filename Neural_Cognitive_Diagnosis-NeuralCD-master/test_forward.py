#from model import Net
import train
train.train()
'''
for i in range(100):
    input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
    input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
    output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
    print('output=')
    print(output_1)
    print('labels=')
    print(labels)
    print('\n')
'''
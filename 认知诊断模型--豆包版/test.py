# lst=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# for i in range(50):
#     lst=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#     for j in range(4):
#         lst[i]=1
#         for k in lst:
#             print(k,end=',')
#         print()

def generate_test_answer_log(num_questions=20):
    lst=[]
    import random
    for i in range(num_questions):
        lst2=[]
        for j in range(1):
            lst2.append(random.randint(0,200))
            lst2.append(random.randint(0,1))
        lst.append(lst2)
    return lst

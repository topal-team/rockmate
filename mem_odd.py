import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("Test is based on GPU, make sure you have it!")
    #device = torch.device('cpu')
    
def case1():
    a = torch.ones([10,10,10,10],device = device)
    a.requires_grad_()
    b = torch.permute(a, [0,2,3,1])
    c = torch.ones([10,10,10,10],device=device)
    c.requires_grad_()

    mem_before = torch.cuda.memory_allocated()
    d = torch.matmul(b,c)
    print("Saved memory before:", torch.cuda.memory_allocated()- mem_before)
    print("Should be close to:", d.shape.numel()*4)
    
print("=========Case 1========")
case1()

def case2():
    a = torch.ones([10,10,10,10],device = device)
    a.requires_grad_()
    b = torch.permute(a, [0,1,3,2])
    c = torch.ones([10,10,10,10],device=device)
    c.requires_grad_()

    mem_before = torch.cuda.memory_allocated()
    d = torch.matmul(b,c)
    print("Saved memory before:", torch.cuda.memory_allocated()- mem_before)
    print("Should be close to:", d.shape.numel()*4)
    
print("=========Case 2========")
print("Compare to case 1, case 2 changed the order of permute")
case2()



def case3():
    a = torch.ones([10,10,10,10],device = device)
    a.requires_grad_()
    b = torch.permute(a, [0,2,3,1])
    c = torch.ones([10,10,10,10],device=device)
    # c.requires_grad_()

    mem_before = torch.cuda.memory_allocated()
    d = torch.matmul(b,c)
    print("Saved memory before:", torch.cuda.memory_allocated()- mem_before)
    print("Should be close to:", d.shape.numel()*4)
    
print("=========Case 3========")
print("Compare to case 1, case 3 does not require grad for the other tensor")
case3()


def case4():
    a = torch.ones([10,10,10],device = device)
    a.requires_grad_()
    b = torch.permute(a, [0,1,2])
    c = torch.ones([10,10,10,10],device=device)
    # c.requires_grad_()

    mem_before = torch.cuda.memory_allocated()
    d = torch.matmul(b,c)
    print("Saved memory before:", torch.cuda.memory_allocated()- mem_before)
    print("Should be close to:", d.shape.numel()*4)
    
print("=========Case 4========")
print("Compare to case 1, case 4 has 3-dim tensors. No permute order can reproduce mem difference") 
case4()

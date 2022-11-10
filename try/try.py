import torch
x=torch.rand(4,7, requires_grad=True)  # 4个样本，共7类
y=torch.LongTensor([1,3,5,0]) # 对应的标签
criterion = torch.nn.CrossEntropyLoss()  # pytroch库
out = criterion(x,y)
print(out)
# 自己实现
gt = torch.zeros(4,7).scatter(1, y.view(4,1),1)  # 生成one-hot标签，scatter的用法可参考jianshu.com/p/b4e9fd4048f4
loss = -(torch.log(torch.softmax(x, dim=-1)) * gt).sum() / 4  # 对样本求平均
print(loss)
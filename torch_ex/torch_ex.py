import torch


#######################
# 创建特殊的torch
#######################

# zeros函数用于创建特定形状的0数组，第一个参数可以是列表或者元组，是目标的维度，从外向里，第二个参数是0的类别
# temp = torch.zeros([2, 3, 5], dtype=torch.float32)
# print(temp)

# zeros_like函数用于创建和输入的tensor相同形状的tensor
# a = torch.tensor([[1, 1, 1],
#                   [2, 2, 2]])
# print(a)
# temp = torch.zeros_like(a)
# print(temp)

# ones函数用于创建特定形状的1数组，与上面相同
temp = torch.ones([2, 3, 5], dtype=torch.float32)
print(temp)

#######################
# torch形状操作
#######################

# squeeze函数用于在tensor中去掉所有size为1的维度，如果传入维度，则会只去掉指定的维度，如果size不为1，则不会产生效果
# 生成的torch与原torch共享内存
# temp = torch.zeros([1, 3, 5], dtype=torch.float32)
# print(temp)
# 可以使用下面三种表示方式，正数代表从前向后，从0开始，负数代表从后向前，从-1开始
# temp = temp.squeeze()
# temp = temp.squeeze(0)
# temp = temp.squeeze(-3)
# print(temp)

# unsqueeze函数用于在tensor中添加一个size为1的维度，传入的参数为添加后新维度所在的维度
# temp = torch.zeros([3, 5], dtype=torch.float32)
# print(temp)
# 该函数必须传入参数，也可以为负数，与上面相同
# temp = temp.unsqueeze(0)
# print(temp)







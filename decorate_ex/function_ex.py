# 有关函数名的一部分可以在线程池里面看到
# 嵌套函数是指在一个函数中定义的函数，该函数只有在其高阶函数中才能调用，外界无法调用
# 嵌套函数出现的场景常为封装和闭包，因为外界无法调用嵌套函数，从而增强其安全性
def outer(x, y, mod):
    # 这种时候经常会提示x和y在外部函数中存在，这其实就是闭包，闭包是指内部函数可以调用外部函数的值，而且当外部函数
    # 名被当做参数进行传递时，其内部的变量值不变，相当于内部函数的运行环境不发生变化
    # def add(x, y):
    #     return x+y

    def add():
        return x+y

    def subtraction():
        return x-y

    if mod == 0:
        return add()
    else:
        return subtraction()


# print(outer(2, 3, 0))
# print(outer(2, 3, 1))

# 闭包有两个注意事项，第一个就是内部函数可以引用外部函数的值，但是不能修改
# 第二个是循环不算外层函数
# result = []

# for i in range(3):
#     def mut(x):
#         return i*x
#     result.append(mut)
#
# 想象的输出结果是0 5 10，但输出的是10 10 10，原因就在于for循环没有保存环境，mut函数中的i只能取到最后一个i，即为10
# for j in range(3):
#     print(result[j](5))

# 改进就是在外面加上一层，以保存mut的环境
# for i in range(3):
#     def outer_for(i):
#         def mut(x):
#             return x*i
#         return mut
#     result.append(outer_for(i))
#
# for j in range(3):
#     print(result[j](5))


# 如果想在嵌套函数总修改外部函数的值，需要nonlocal进行声明，否则会报错
def outer_change():
    http = 10

    def change():
        nonlocal http
        http += 10

    change()
    print(http)


outer_change()

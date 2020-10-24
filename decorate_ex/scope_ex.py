# python的作用域为内置，全局，和局部，查找时由局部向外查找
# python中只有model，class，function才会产生域，if，for等语句不会，内部变量可以直接由外部访问
# 正常情况下，内部作用域只能引用全局变量，当内部作用域修改全局变量时，实际上是创建一个新的局部变量，全局变量没有发生变化

i = 0
if i == 1:
    print("i为1")
    j = 0
else:
    print("i不为1")
    j = 1
print(j)


# 当想在内部作用域中修改全局变量时，需要进行声明，如果不声明会报错，因为找不到局部变量i
# 嵌套函数修改请看function_ex文件
def add():
    global i
    i += 10


add()
print(i)



















# 迭代器与list的区别
# 内置有__iter__方法的对象都为可迭代对象，list就是可迭代对象，通过iter方法就可以得到迭代器对象
# 迭代器对象包含内置方法__next__，用于取下一个值，迭代器对象只能够遍历一次，它只记录当前的地址
# 通过next方法计算下一个地址，再取到内存中，所以迭代器对象无法回头，而且无法得到长度，想要得到长度就
# 要一直next直到抛出异常，这样就知道长度了

a = [1, 2, 3, 4]
a_iter = iter(a)
for i in a:
    print(i)

for i in a_iter:
    print(i)

for i in a:
    print(i)

# 这次就不会输出结果，因为地址已经指向末尾，要想输出只能重新用a再生成一个迭代器
for i in a_iter:
    print(i)

# list合并直接使用+号就可以
b = [5, 6, 7, 8]
print(a+b)











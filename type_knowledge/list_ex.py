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


# python的列表生成式是一个内置功能，可以通过简洁的代码写出for循环
# 主要分为三种
# 基本的列表生成式，就相当于将for循环需要的到的量放到了for的前面
c = [i for i in range(5)]
print(c)
# 除此之外，还可以进行嵌套循环，前面的for相当于内层循环
d = [i*j for j in range(5, 10) for i in range(1, 5)]
print(d)


# 第二种是带有if的，这种是可以由一个判定条件，当什么情况下会返回，if必须放在for的后面，下面就是获得偶数
e = [i for i in range(10) if i % 2 == 0]
print(e)


# 第三种是带有if else的，这种是有两种情况可以返回，此时if else必须放在for的前面，i前面是if的结果，else后面是else的结果
f = [i if i % 2 == 0 else i*i for i in range(10)]
print(f)
# 如果比以上三种情况还要复杂，就老老实实写for循环吧



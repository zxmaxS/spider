# 装饰器是对于python可以将函数作为变量传递而产生的，其思想主要是利用闭包，不想对一个函数进行修改但又想增加其功能性
# 常用场景为web运行某个函数时进行权限检测，或者进行日志编写
import time
from functools import wraps


# 装饰器就是定义一个函数，将传进函数封装到内置函数中，再把内置函数返回回去
def decorate(func):
    # 这个东西也是一个装饰器，这是一个python内置函数，作用是置换函数的一些属性，比如现在这个装饰器是将update赋值给hello
    # 调用hello.__name__时会输出update，这会产生一些无法预料的错误，所以就需要这一句话对返回的函数进行处理
    @wraps(func)
    # 下面的参数可以看thread_pool_ex中有解释，这里的意思就是将任意参数原封不动地传给被修饰的函数
    def update(*args, **kwargs):
        print(time.strftime("%H:%M:%S", time.localtime()))
        func(*args, **kwargs)
    return update


# @在python中只有两个用处，一个是类中定义静态函数@staticmethod，第二个就是装饰器的简化作用，下面的句子就相当于
# hello = decorate(hello)
# 只是少写了一行代码
@decorate
def hello():
    print("hello world")


# 这是一个具有参数的装饰器
def decorate_with_param(http='zx'):
    def decorate_inner(func):
        @wraps(func)
        def update(*args, **kwargs):
            print(http)
            print(time.strftime("%H:%M:%S", time.localtime()))
            func(*args, **kwargs)
        return update
    return decorate_inner


# 实际上就相当于
# hello_2 = decorate_with_param('temp')(hello_2)
# 就是先运行前面的函数，前面的函数会返回一个函数，然后将hello_2作为参数传入返回函数中再次返回一个函数赋值给hello_2
@decorate_with_param('temp')
def hello_2():
    print("hello world")


# 当存在多个装饰器时会进行嵌套，即下面的语句相当于
# hello_3 = decorate(decorate_with_param()(hello_3))
@decorate
@decorate_with_param()
def hello_3():
    print("hello world")


# 当你的装饰器内部需要调用多个函数，并且它可能还需要扩展其他功能，这时就可以设计一个装饰器类
# 当变为类后，就可以在内部设计多个方法，并且可以通过继承来实现类的拓展
class decorate_class:
    # 初始化函数就是装饰器传来的参数
    def __init__(self, http):
        self.http = http

    # __call__方法可以使一个类具有函数的性质，即在该类的实例后面放一对小括号，就会执行下面的函数
    def __call__(self, func):
        @wraps(func)
        def update(*args, **kwargs):
            self.print_time()
            print(self.http)
            func(*args, **kwargs)
        return update

    def print_time(self):
        print(time.strftime("%H:%M:%S", time.localtime()))


# 这相当于
# hello_4 = decorate_class('zx')(hello_4)
# 前面的decorate_class('zx')实际上是相当于创建一个实例，后面加上一对括号就是正常的装饰器
@decorate_class('zx')
def hello_4():
    print("hello world")


if __name__ == '__main__':
    # hello()
    # hello_2()
    # hello_3()
    hello_4()









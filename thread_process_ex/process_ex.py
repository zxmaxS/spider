# python的多进程是由multiprocessing模块实现的，由于windows上没有fork函数但是python是全平台的，所以有了
# 这个模块，在linux上可以直接使用fork进行多线程，不过由于封装较少，实现起来较为复杂，该模块与线程模块几乎相同
from multiprocessing import Process
from time import sleep
from functools import wraps


def func_decorate(func):
    @wraps(func)
    def update(num):
        print("进程{}开始运行".format(num))
        # 注意这里是调用传进来的参数，不是调用fun1，因为调用fun1会再次触发装饰器，形成无限循环
        func(num)
        print("进程{}结束运行".format(num))
    return update


@func_decorate
def fun1(num):
    sleep(num)


# 多进程与进程池必须在main函数中调用，否则会报错，原因在于由于没有fork，python的多线程在windows上需要进行模拟，而
# 子进程会import主进程文件，如果不放在main中，就会在import时重新再次运行创建进程函数，从而形成无限循环
if __name__ == '__main__':
    process_list = []

    for i in range(5):
        # 创建进程需要进行键值赋值，因为Process类的默认参数顺序不太对，具体什么也不太清楚，反正就键值赋值就行了
        # target就是执行函数名，args传入一个元组，也就是参数列表
        temp = Process(target=fun1, args=(i+1, ))
        # start函数与多线程一致
        temp.start()
        process_list.append(temp)

    for item in process_list:
        # join函数与多线程一致
        item.join()












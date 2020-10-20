# python的线程池是指当你有多个任务时不需要创建多个线程，而是将少量线程循环利用，让任务进行排队
# 从而节省线程的创建与销毁的开销，下面的类就是线程池
from concurrent.futures import ThreadPoolExecutor
from time import sleep


def task(name, delay, rounds):
    for i in range(delay):
        print("线程%d运行第%d次" % (name, i+1))
        sleep(rounds)


if __name__ == '__main__':
    # 这是创建线程池对象，传入的参数为同时执行的最大线程数，就是当任务数多于3个时后面的任务就会等待
    pool = ThreadPoolExecutor(3)
    for i in range(5):
        # submit函数就是将任务放入线程池，其中的第一个参数是函数名，千万不要后面带上括号加上参数，否则就只是正常
        # 地执行函数，没有用到线程池，其中task函数需要的参数直接在后面用,隔开就可以了，也可以写上参数名
        # 这个函数在文档中的示例为submit(fn, *args, **kwargs)，其中的*args会将你传入的没有参数名的参数
        # 整合成一个元组，而后面的**kwargs会将你传人的有参数名的参数整合成一个字典
        result1 = pool.submit(task, i+1, 2, rounds=5)
    # print(result1.done())

















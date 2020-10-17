# 目前的python实现多线程主要依靠threading模块
import threading
from time import sleep


# 自己创建的线程要继承threading类并重写run方法
class myself(threading.Thread):
    def __init__(self, name, counter):
        # 这是一种传统的调用父类方法方式
        threading.Thread.__init__(self)
        self.name = name
        self.counter = counter

    # run函数就是当你想让线程执行的函数
    def run(self):
        print("开始线程"+self.name)
        print_time(self.name, self.counter)
        print("结束线程"+self.name)


def print_time(name, delay):
    for i in range(5):
        print(name)
        sleep(delay)


thread1 = myself("第一个线程", 2)
thread2 = myself('第二个线程', 3)

# 当调用start函数后，线程就会运行你在run里面定义的函数，但是该方法是非阻塞的，也就是你调用之后不管线程是否运行
# 完毕，都向下执行，如果想要阻塞，要使用join函数
thread1.start()
thread2.start()

# join函数是一个阻塞函数，即执行到这等待调用的线程运行完才会跑下一行代码，这里就是等待线程1运行完才会跑下一行
# 即使线程二先跑完也会被阻塞，这个函数也可以传入参数，参数为等待的时间，如果你只想等这个线程2秒，就可以传入参数2
thread1.join()
thread2.join()


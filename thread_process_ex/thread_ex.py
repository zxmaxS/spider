# 目前的python实现多线程主要依靠threading模块
# 线程与进程的选择，当任务的相关性较高时使用线程，否则使用进程
# 集群是指多台服务器提供同一个服务，有一台服务器进行负载均衡，分布式就是一台服务器提供一种服务，类似于前后端分离
import threading
from time import sleep

# 所有子线程共享全局变量，当没有线程锁时就会发生混乱
http = 1
# 这就是线程锁，可以申请多个锁，操作系统中学过的锁
lock = threading.Lock()
# 这是线程中的独立变量，可以使用同一个全局对象为每个线程创建不同的字典，但是我没找到有什么用处
local = threading.local()


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
        # print_time(self.name, self.counter)
        # 这是第一种申请锁的方式，acquire函数就是申请锁，release函数就是释放锁
        lock.acquire()
        lock_test_add(self.counter)
        lock.release()
        # 虽然两个线程使用同一个local对象，但他们的name属性是不同的，每个线程之间相互独立
        local.name = self.name
        print(local.name)
        print("结束线程"+self.name)


class yourself(threading.Thread):
    def __init__(self, name, counter):
        threading.Thread.__init__(self)
        self.name = name
        self.counter = counter

    def run(self):
        print("开始线程"+self.name)
        # 这是第二种申请锁的方式，with函数可以看exercise文件
        with lock:
            lock_test_subtraction(self.counter)
        local.name = self.name
        print(local.name)
        print("结束线程"+self.name)


def lock_test_add(counter):
    global http
    for i in range(counter):
        http += 1
        print(http)


def lock_test_subtraction(counter):
    global http
    for i in range(counter):
        http -= 1
        print(http)


def print_time(name, delay):
    for i in range(5):
        print(name)
        sleep(delay)


thread1 = myself("第一个线程", 5)
thread2 = yourself('第二个线程', 6)
# thread2 = myself('第二个线程', 3)

# 默认情况下线程均为非守护线程，即使主线程结束了，主进程也会等待所有子线程运行完毕后自杀，当设置为守护线程时
# 当主线程运行完毕，没有阻塞，该子线程也立即结束，而不是等待运行完毕，而且设置必须在线程运行之前
# 守护线程的子线程也为守护线程
# thread2.setDaemon(True)

# 当调用start函数后，线程就会运行你在run里面定义的函数，但是该方法是非阻塞的，也就是你调用之后不管线程是否运行
# 完毕，都向下执行，如果想要阻塞，要使用join函数
thread1.start()
thread2.start()

# join函数是一个阻塞函数，即执行到这等待调用的线程运行完才会跑下一行代码，这里就是等待线程1运行完才会跑下一行
# 即使线程二先跑完也会被阻塞，这个函数也可以传入参数，参数为等待的时间，如果你只想等这个线程2秒，就可以传入参数2
thread1.join()
thread2.join()

# queue在线程中的应用主要是将变量存入队列中，queue自带锁机制，从而使得多线程同步，这里就不放演示了，想用的话自己看一看



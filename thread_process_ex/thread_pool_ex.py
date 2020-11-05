# python的线程池是指当你有多个任务时不需要创建多个线程，而是将少量线程循环利用，让任务进行排队
# 从而节省线程的创建与销毁的开销，下面的类就是线程池
from concurrent.futures import ThreadPoolExecutor
# 这个方法使用来判断进程是否完成，传入的是一个future对象数组，当没有任务完成时就会阻塞，有任务完成时就会返回完成的
# future对象
from concurrent.futures import as_completed
from concurrent.futures import wait
from time import sleep


def task(name, delay, rounds):
    for i in range(delay):
        print("线程%d运行第%d次" % (name, i+1))
        sleep(rounds)
    return name


# python多进程在windows下必须在下面的main中，原因可以在process_pool_ex中看，而线程池不需要
if __name__ == '__main__':
    # 这是创建线程池对象，传入的参数为同时执行的最大线程数，就是当任务数多于3个时后面的任务就会等待
    pool = ThreadPoolExecutor(3)
    parma1 = []
    parma2 = []
    parma3 = []
    for i in range(5):
        parma1.append(i+1)
        parma2.append(5-i)
        parma3.append(3)
    # map函数与内置库的map函数相同，第一个参数是函数名，剩下的参数为函数中的参数的list，返回的是一个迭代器
    # 与as_completed不同的是这个函数不是先运行完先输出，而是根据你输入参数的顺序，当第一个没有运行完，即使
    # 第二个运行完也不会输出，等待第一个运行完后输出
    # python的函数名如果不带括号，则只是一个存储函数开始地址的变量，如果带有括号才是执行的意思，所以这里传入函数名
    # 不能带括号，函数名不仅可以传递，还可以作为返回值，将其他函数名作为参数的函数称为高阶函数
    for item in pool.map(task, parma1, parma2, parma3):
        print(item)
    # result = []
    # for i in range(5):
    #     # submit函数就是将任务放入线程池，其中的第一个参数是函数名，千万不要后面带上括号加上参数，否则就只是正常
    #     # 地执行函数，没有用到线程池，其中task函数需要的参数直接在后面用,隔开就可以了，也可以写上参数名
    #     # 这个函数在文档中的示例为submit(fn, *args, **kwargs)，其中的*args会将你传入的没有参数名的参数
    #     # 整合成一个元组，而后面的**kwargs会将你传人的有参数名的参数整合成一个字典
    #     result.append(pool.submit(task, i+1, 5-i, rounds=3))
    # # wait函数有三个参数，第一个是任务的list，第二个是最长等待时间，第三个是停止阻塞条件，默认是list中所有任务完成
    # # 也可以换成其他的，自己查一查
    # wait(result)
    # print("所有线程运行完毕")
    # # 这样就可以使所有线程完成后主线程向下运行
    # for item in as_completed(result):
    #     print(item.result())
    # for item in result:
    #     # cancel函数可以取消还未进行运行的任务，比如第四个和第五个，取消成功返回true，失败返回false
    #     print(item.cancel())
    # for item in result:
    #     sleep(3)
    #     # submit函数会返回一个future对象，可以通过该对象获得线程的状态，下面是查看线程是否完成
    #     print(item.done())
    # for i in range(3):
    #     # result函数会获得函数返回值
    #     print(result[i].result())
    # 当不需要线程池后使用shutdown函数关闭线程池，关闭后的线程池不再接受新任务，但会把已接收的任务完成
    pool.shutdown()
















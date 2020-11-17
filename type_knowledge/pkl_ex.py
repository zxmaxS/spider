# pkl是python保存文件的一种后缀名，主要是依赖下面的这个包，这是将一个自定义的python对象序列化(变为字符串)
# 后存储到文件中，存储时是二进制模式，对于依赖外部状态的对象无法序列化，如线程等，但是有办法，如果需要可以去查
import pickle
import time


class Employee:
    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        self.print_time()
        print("总人数为%d" % Employee.empCount)

    def displayName(self):
        self.print_time()
        print("名字是%s" % self.name)
        print("薪水为%s" % self.salary)

    def print_time(self):
        print(time.strftime("%H:%M:%S", time.localtime()))

    def __del__(self):
        Employee.empCount -= 1
        print("%s已被销毁" % self.name)


if __name__ == '__main__':
    emp_1 = Employee('张三', 10000)
    # emp_2 = Employee('李四', 5000)
    # # 一定要以二进制格式打开
    # file = open('type_knowledge/data/test.pkl', 'wb')
    # # dump函数用于将一个对象存入文件中
    # # 可以在一个文件中存入多个对象
    # pickle.dump(emp_1, file)
    # pickle.dump(emp_2, file)
    # file.close()
    # del emp_1
    # del emp_2
    # # 读取文件需要rb
    # file = open('type_knowledge/data/test.pkl', 'rb')
    # # 一个文件中存在多个对象时可以依次取出，是队列，不是栈
    # emp_3 = pickle.load(file)
    # emp_4 = pickle.load(file)
    # # 导入的数据类变量会丢失
    # print(emp_3.empCount)
    # print(emp_4.empCount)
    # file.close()

    # dumps函数将一个对象转变为序列化字符串而不是存入文件中
    data = pickle.dumps(emp_1)
    print(data)
    emp_5 = pickle.loads(data)
    emp_5.displayName()
















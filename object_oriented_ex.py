# 类的定义就是class加上类名就可以了
class Employee:
    # 这是一个类变量，类变量与实例变量是不同的，它对于所有的对象使通用的，相当于一个全局变量
    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    @staticmethod
    def displayCount():
        print("总人数为%d" % Employee.empCount)

    # 类中的第一个参数为self，这个参数就代表调用这个函数的对象，当你的对象使用.调用函数时，自身也会被传入
    #
    def displayName(self):
        print("名字是%s" % self.name)
        print("薪水为%s" % self.salary)

    def __del__(self):
        Employee.empCount -= 1











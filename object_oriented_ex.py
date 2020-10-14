# 类的定义就是class加上类名就可以了
class Employee:
    # 这是一个类变量，类变量与实例变量是不同的，它对于所有的对象使通用的，相当于一个全局变量
    # 调用的时候使用类名调用
    empCount = 0

    # 这个函数就是构造函数，创建实例时传入的参数传进这里
    def __init__(self, name, salary):
        # 对于一个实例，刚被创建的时候并没有任何实例变量，只有从类中继承的类变量，这时通过.进行调用可以得到类变量的值
        # 可以通过.直接创建类变量，比如下面的name与salary，类中本来没有这两个变量，这里直接创建了这两个实例变量
        self.name = name
        self.salary = salary
        # 当创建的实例变量与类变量同名时，通过实例调用的变量为实例变量，与类变量没有任何关系
        self.empCount = 5
        print(Employee.empCount)
        print(self.empCount)
        Employee.empCount += 1
        # del函数用于删除实例变量，当删除后调用由于没有同名的实例变量，直接调用类变量
        del self.empCount
        print(self.empCount)

    @staticmethod
    def displayCount():
        print("总人数为%d" % Employee.empCount)

    # 类中的第一个参数为self，这个参数就代表调用这个函数的对象，当你的对象使用.调用函数时，自身也会被传入
    # 传入的主要作用是调用实例变量，相当于局部变量
    def displayName(self):
        print("名字是%s" % self.name)
        print("薪水为%s" % self.salary)

    def __del__(self):
        Employee.empCount -= 1


emp1 = Employee('无敌', 100000)








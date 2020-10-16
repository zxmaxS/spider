import copy


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
        # self.empCount = 5
        # print(Employee.empCount)
        # print(self.empCount)
        Employee.empCount += 1
        # del函数用于删除实例变量，当删除后调用由于没有同名的实例变量，直接调用类变量
        # del self.empCount
        # print(self.empCount)

    @staticmethod
    def displayCount():
        print("总人数为%d" % Employee.empCount)

    # 类中的第一个参数为self，这个参数就代表调用这个函数的对象，当你的对象使用.调用函数时，自身也会被传入
    # 传入的主要作用是调用实例变量，相当于局部变量
    def displayName(self):
        print("名字是%s" % self.name)
        print("薪水为%s" % self.salary)

    # 这是析构函数，在使用del方法删除对象时被调用，或者程序终止时调用
    def __del__(self):
        Employee.empCount -= 1
        print("%s已被销毁" % self.name)


# 类的实例化没有new关键词，直接向类名中传递参数
# print(Employee.empCount)
emp1 = Employee('无敌', 100000)
emp1.displayName()


# python中所有变量都是一个对象，对象分为可变对象与不可变对象，python中赋值对象总是建立引用值而不是复制对象
# 可变对象包含list,set,dict，不可变对象包括int,float,str,tuple
# 对于不可变对象，重新赋值并不是改变原有的数值，而是创建一个新的数值，使变量指向新值，如果旧值引用为0则被垃圾回收，如下
# 使变量名a指向内存中的1 int对象，如果内存中已经存在，则直接指向，否则创建1 int对象
# a = 1
# 使变量名a指向内存中的2 int对象，如果内存中已经存在，则直接指向，否则创建2 int对象，此时1 int对象仍然存在于内存之中
# 如果有其他变量的值为1，则继续存在，否则进行垃圾回收
# a = 2
# 下面也是使变量名b指向内存中的2 int对象，因为上面已经创建好了
# b = a


# 对于可变对象，变量的更改只是在原有地址的基础上进行修改，而不是创建一个新的对象，如下
# a = [0, [1, 2], 3]
# b = a
# print("a1", a)
# print("b1", b)
# a[0] = 8
# a[1][1] = 9
# print("a2", a)
# 可以发现改变a后b也会改变，原因就是a与b指的是同一个位置，两者只是个别名的关系
# print("b2", b)


# 如果想要新建一个可变对象，需要使用deepcopy函数
# a = [0, [1, 2], 3]
# b = copy.deepcopy(a)
# print("a1", a)
# print("b1", b)
# a[0] = 8
# a[1][1] = 9
# print("a2", a)
# 此时a与b指向的不是同一个对象，而是两个list对象
# print("b2", b)


# python的垃圾销毁机制是检测一个对象的引用数是否为0，如果为0则会删除这个对象的内存
# 当两个变量相互引用时，python回定期检测这种引用情况
# 下面的赋值就相当于创建了一个别名而不是一个新的Employee对象，所以empCount不会发生变化
emp2 = emp1
print(Employee.empCount)
# 当删除一个别名时，由于有另一个变量指向该对象，所以该对象仍然存在于内存之中，empCount不会发生变化
del emp1
print(Employee.empCount)
emp2.displayName()
# 这里删除之后该对象引用为0，从内存中被清除
del emp2
print(Employee.empCount)




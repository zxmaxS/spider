# 方法重写是指子类对父类的方法进行重新修改，参数相同
# 方法重载是指函数名相同但参数不同
# private只能类内调用，protect可以由继承类进行调用
class People:
    allPeople = 0
    # 前面有一个下划线的是protect
    _http = 0
    # 前面有两个下划线的是private
    __http = 0
    # 函数的private与protect和上面的定义方法相同

    def __init__(self, name, age):
        self.name = name
        self.age = age
        People.allPeople += 1

    def eat(self):
        print("吃饭")

    def sleep(self):
        print("睡觉")


# python中继承只需要在类名后面加上括号，括号内就是父类
class Student(People):
    # 子类的构造函数如果想继承父类，则需要不写或者显示地调用父类构造函数
    def __init__(self, name, age, school):
        # 这里就是显示地调用，如果没有下面的语句，则不会调用父类构造函数
        super().__init__(name, age)
        self.school = school

    def study(self):
        print("学习")

    # 这就是方法重写，当对象调用一个方法时首先在子类中进行寻找，如果子类中找不到的话再去父类中寻找
    def eat(self):
        print("在食堂吃饭")


s1 = Student("无敌", 18, "家里蹲大学")
s1.study()
s1.eat()

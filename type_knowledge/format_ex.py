# format函数是python2.6以上版本提出用来取代%规范字符串的一种方法
# 其中中括号代表一个位置，由后面中按位置进行传递

print("{} {}".format('hello', 'world'))


# 也可以自行指定位置，将后面作为一个list
print("{1} {0} {1}".format('hello', 'world'))


# 也可以将后面视为一个dict
print("{name} {age}".format(name='Jack', age=18))


# 也可以直接传入一个字典
temp = {
    'name': 'Jack',
    'age': 18
}
# 这里的**相当于解包功能，就是将大括号去掉，变成参数的形式，对于list也是类似，前面加上一个*，前置内容可以查看return_ex
print("{name} {age}".format(**temp))


# 传入list
temp_1 = ['Jack', '18']
temp_2 = ['hello', 'world']
# 可以这样将list传进去，在字符串中用下标调用，其中0是必要的和1就相当于位置后面的中括号是下标
print("{0[0]} {0[1]} {1[0]} {1[1]}".format(temp_1, temp_2))
# 也可以使用*进行解包，和上面dict类似
print("{1} {0} {1}".format(*temp_1))






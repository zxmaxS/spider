# format函数是python2.6以上版本提出用来取代%规范字符串的一种方法
# 其中中括号代表一个位置，由后面中按位置进行传递

print("{} {}".format('hello', 'world'))


# 也可以自行指定位置，将后面作为一个list
print("{1} {0} {1}".format('hello', 'world'))


# 也可以将后面视为一个dict
print("{name} {age}".format(name='Jack', age=18))








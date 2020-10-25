import re
# ?不是表示通配符0个或1个，而是表示?前的字符0个或1个
# [\s,;]+ 这个加号表示的不是前面一个\s后面就都是\s，而表示的是[\s,;][\s,;][\s,;]
# [\s\S]+ 比.匹配的更多，因为.不能匹配换行，[a-c]也可以写成[abc]
# ^代表匹配字符串开始的为，$代表匹配字符串结束的位置
# []代表一个可替换字符，而大括号则是代表前面字符出现的次数，如0{2}，就是匹配00，0{1,3}，匹配000


# 由于python字符串也用\转义，所以最好在前面加个r
rule = r'ABC\-001'
content = '010-12345'

# 最常用的匹配方法，匹配成功，返回match对象，否则返回none
re.match(rule, content)
# match是从头开始进行匹配的，如果想从任意位置开始匹配，使用search函数
re.search(rule, content)

# 当正常的split无法满足要求时，可以使用正则表达式来进行切分
re.split(rule, content)

# 提取字符串使用group方法，其中0代表本身，1和后面的数字表示你的第一个括号，第二个括号
rule = r'^(\d{3})-(\d{3,8})$'
result = re.match(rule, content)
print(result.group(0))
print(result.group(1))
print(result.group(2))


# 正则表达式默认是贪婪匹配，即尽量向前一个匹配，如果在某句后面加上?表示不贪婪匹配，则回当后面开始匹配时停止匹配
# 比如第一个就是贪婪匹配，当匹配到100时，10就会算到前面的.*通配符中而不会算到后面的\d+里面
# 而第二个就让.*非贪婪，10就不会被.*匹配掉而进入后面
content = 'Xiaoshuaib has 100 bananas'
rule = r"^Xi.*(\d+)\s.*s$"
result = re.match(rule, content)
print(result.group(1))
rule = r"^Xi.*?(\d+)\s.*s$"
result = re.match(rule, content)
print(result.group(1))


# re使用时会先编译在运行，如果一个表达式会用到多次，可以进行预编译
temp = re.compile(rule)
result = temp.match(content)

# 如果匹配的内容中有换行符
result = re.match(rule, content, re.S)


# 找出所有字符
result = re.findall(rule, content)


# 修改字符串
content = re.sub(r'\d+', '250', content)


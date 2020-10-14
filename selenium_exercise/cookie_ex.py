from selenium import webdriver

driver = webdriver.Chrome()
driver.get("http://www.baidu.com")

driver.maximize_window()


# 下面的方法是用于获得全部的cookie，得到的是一个字典的列表，每个字典代表一个键值对，其中的name属性为键，value属性为值
# 其余的5个变量均为无关变量，是用于判断访问cookie权限的字段
# cookie = driver.get_cookies()
# for item in cookie:
#     print("%s --> %s" % (item['name'], item['value']))

# 下面的方法是通过cookie名得到cookie，得到也是一个字典，从中获得value值就行了
# cookie = driver.get_cookie("PSTM")
# print(cookie)

# 下面的方法是添加cookie，每次只能传进一个字典，其中要包含name与value剩下的五个属性可以使用默认值
# driver.add_cookie({
#     'name': '张鑫',
#     'value': '20'
# })
# cookie = driver.get_cookies()
# for item in cookie:
#     print("%s --> %s" % (item['name'], item['value']))

# 删除cookie使用下面的方法
cookie = driver.get_cookies()
for item in cookie:
    print("%s --> %s" % (item['name'], item['value']))
driver.delete_cookie("PSTM")
cookie = driver.get_cookies()
for item in cookie:
    print("%s --> %s" % (item['name'], item['value']))

# 删除全部cookie
driver.delete_all_cookies()
cookie = driver.get_cookies()
print(cookie)



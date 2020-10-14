import urllib.request as req  # url库有四个模块。request是其中一个
import requests  # 这个库是在urllib的基础上扩展的

# 三个参数第一个url，第二个是传给网站的参数，第三个是响应时间
# 默认是get，传入参数就变成post
# response = req.urlopen("http://www.baidu.com")
# print(response.read().decode('utf-8'))


# 四个参数路由，参数，头部，以及方法
# response = req.Request("http://www.baidu.com")


# 其他使用方法都类似，有参数就直接在括号里面加，想获得目标数据也可以直接在response里调用方法或者属性
response = requests.get("http://www.baidu.com")
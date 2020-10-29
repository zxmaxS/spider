import urllib.request as req  # url库有四个模块。request是其中一个
import requests  # 这个库是在urllib的基础上扩展的

# 三个参数第一个url，第二个是传给网站的参数，第三个是响应时间
# 默认是get，传入参数就变成post
# response = req.urlopen("http://www.baidu.com")
# print(response.read().decode('utf-8'))


# 四个参数路由，参数，头部，以及方法
# response = req.Request("http://www.baidu.com")


# post就是把下面的get换成post
# response = requests.get("http://www.baidu.com")
# response = requests.post("http://www.baidu.com")


# get方法除了可以直接在url中加上?，也可以直接将参数封装到一个字典中，传入头部的方法也放在下面了
headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.102 Safari/537.36 '
    }
# response = requests.get('https://movie.douban.com/top250?start=0', headers=headers)
params = {
    'start': 0
}
response = requests.get('https://movie.douban.com/top250', params=params, headers=headers)

# 通过text方法可以获取网页源代码，网页传输过来的是01文件流，需要进行解码，response可以通过响应头判断网页的编码来进行解码
# 解码后才是正常代码，即Unicode编码
# print(response.text)
# 虽然两者的输出相同，但content获取的是byte数据，即用于获得图片流然后写入图片中，而text只能爬取图片地址
# print(response.content)
# 可以通过encoding属性查看网页的编码
print(response.encoding)
# 也可以人为设定其编码
# response.encoding = 'utf-8'


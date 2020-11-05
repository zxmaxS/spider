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
# params = {
#     'start': 0
# }
# response = requests.get('https://movie.douban.com/top250', params=params, headers=headers)

# 通过text方法可以获取网页源代码，网页传输过来的是01文件流，需要进行解码，response可以通过响应头判断网页的编码来进行解码
# 解码后才是正常代码，即Unicode编码
# print(response.text)
# 虽然两者的输出相同，但content获取的是byte数据，即用于获得图片流然后写入图片中，而text只能爬取图片地址
# print(response.content)
# 可以通过encoding属性查看网页的编码
# print(response.encoding)
# 也可以人为设定其编码
# response.encoding = 'utf-8'


# 接下来我用自己的一个本地网站测试一下文件上传，等我学完nginx再考虑移到线上吧
# 一个input标签发送一个文件时
# 使用files参数传入一个字典，字典的键就是input标签
# 的name，字典的值是一个元组，其中第一个是文件名，第二个是以rb方式打开的文件，因为要进行传输，所以需要使用二进制，第三个参
# 数是文件类型，如image/jpeg这种，第四个参数为文件的头部，这个需要啥自己再去看吧，其中第三个第四个参数可以缺省，使用默认值
# file = {
#     'avatar': ('comment.txt', open('result/bili_comment.txt', 'rb')),
#     'avatar_2': ('comment_comment.txt', open('result/bili_comment_comment.txt', 'rb'))
# }
# 字典还可以简化成，此时文件名会通过路径自动获取，如果不需重命名文件就可以这样
# file = {
#     'avatar': open('result/bili_comment.txt', 'rb'),
#     'avatar_2': open('result/bili_comment_comment.txt', 'rb')
# }
# 当在后面加上read函数后，会将字典的键作为文件名上传
# file = {
#     'avatar': open('result/bili_comment.txt', 'rb').read(),
#     'avatar_2': open('result/bili_comment_comment.txt', 'rb').read()
# }
# 使用元组时就是将键与值封装为一个元组，然后放入一个list中
# file = [
#     ('avatar', open('result/bili_comment.txt', 'rb')),
#     ('avatar_2', open('result/bili_comment_comment.txt', 'rb'))
# ]
# 当一个input传递多个文件时
# 多文件上传的字典模式没有搞懂，元组模式就是下面这样，第一个参数相同，至于为什么input标签的name有个[]可以看一看laravel
# 的文件上传，应该不久之后就会写
# file = [
#     ('avatar', open('result/bili_comment.txt', 'rb')),
#     ('avatar_2[]', open('result/bili_comment_comment.txt', 'rb')),
#     ('avatar_2[]', open('result/bilibili.csv', 'rb'))
# ]
# 除此之外，还可以自定义文件内容而不是打开文件
file = {
    'avatar': ('result.csv', 'some,data,to,send\nanother,row,to,send\n')
}

# response = requests.post('http://127.0.0.1:5689/file_update', files=file)
# print(response.status_code)




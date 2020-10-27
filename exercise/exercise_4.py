import requests
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import json


# 应该是ip被封了，过两天再试一试，虽然被封了，还是写一写新学到的内容，明天应该就是代理ip了，等明天再说
# 今天是第二天，我已经解封了，我减少了线程池的个数，并且增加了每次请求需要进行sleep，已经能够进行爬取了，但还是挺慢的，因为不敢
# 加速拍被封，必须要sleep一秒钟，爬一页需要至少20秒，线程数为3，要是想爬万古生香得80分钟，等我学了ip池再来改进吧
# 这里是将视频id独立出来，便于修改
oid = 56340700


def get_page():
    url = 'https://api.bilibili.com/x/v2/reply'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.121 Safari/537.36 '
    }
    # 首先是参数，B站的评论应该是用js进行请求的，请求出来直接就是json数据，当想要爬取一个网站时，先不要硬着看网页源代码，可以
    # 先去搜索一下有没有爬虫，一般性的网站网上都会有分析的例子，这样就不用自己一点点硬着分析，这个参数就是从网上找到的，大概就
    # 可以看懂，pn代表页数，oid代表视频的id，其他参数是ajax的，等我回头学一学javascript后再说
    parmas = {
        'jsonp': 'jsonp',
        'pn': 1,
        'type': 1,
        'oid': oid,
        'sort': 2
    }
    # requests库自带json解析，否则需要调用python内置json解析库
    # 内置json库只有两个方法，json.dumps()将一个对象封装成json格式，json.loads()将json对象解析为字典格式
    # 如除了用requests的json解析外还可以写为
    # html = json.loads(requests.get(url, headers=headers, params=parmas))
    html = requests.get(url, headers=headers, params=parmas).json()['data']['page']
    # 获得的数据中count为总评论数，size为每页的评论数
    count = html['count']
    size = html['size']
    # ceil函数为向上取整，得到float，floor函数为向下取整，也得到float
    return int(ceil(count/size))


def get_page_reply(id):
    url_2 = 'https://api.bilibili.com/x/v2/reply/reply'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.121 Safari/537.36 '
    }
    parmas_2 = {
        'jsonp': 'jsonp',
        'pn': 1,
        'type': 1,
        'oid': oid,
        'root': id,
        'ps': 10
    }
    html = requests.get(url_2, headers=headers, params=parmas_2).json()['data']['page']
    count = html['count']
    size = html['size']
    return int(ceil(count/size))


def main_reply(j, id):
    url_2 = 'https://api.bilibili.com/x/v2/reply/reply'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.121 Safari/537.36 '
    }
    parmas_2 = {
        'jsonp': 'jsonp',
        'pn': j,
        'type': 1,
        'oid': oid,
        'root': id,
        'ps': 10
    }
    html = requests.get(url_2, headers=headers, params=parmas_2).json()['data']['replies']
    sleep(1)
    if html is not None:
        reply_reply = []
        for item in html:
            reply_reply.append(item['content']['message'])
        return reply_reply
    else:
        return None


def main(i):
    url = 'https://api.bilibili.com/x/v2/reply'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.121 Safari/537.36 '
    }
    parmas = {
        'jsonp': 'jsonp',
        'pn': i,
        'type': 1,
        'oid': oid,
        'sort': 2
    }
    html = requests.get(url, headers=headers, params=parmas).json()['data']['replies']
    reply = []
    reply_reply = []
    for item in html:
        reply.append(item['content']['message'])
        page = get_page_reply(item['rpid'])
        for j in range(page):
            temp = main_reply(j, item['rpid'])
            if temp is not None:
                reply_reply += temp
    sleep(2)
    return reply, reply_reply


if __name__ == '__main__':
    pool = ThreadPoolExecutor(3)
    # 通过一次请求来计算页数
    page = get_page()
    count_num = 0
    # 打开txt与打开csv没啥区别，还可以像下面一样打开多个文件
    with open('result/bili_comment.txt', 'a', encoding='UTF-8') as file_1, open('result/bili_comment_comment.txt', 'a', encoding='UTF-8') as file_2:
        mission = []
        for i in range(page):
            mission.append(pool.submit(main, i))
        for item in as_completed(mission):
            # 这里是return_ex中的内容
            reply, reply_reply = item.result()
            print(count_num)
            count_num += 1
            for detail in reply:
                # 这里是format_ex中的内容
                file_1.write('{}\n'.format(detail))
            for detail in reply_reply:
                file_2.write('{}\n'.format(detail))


    # main(1)

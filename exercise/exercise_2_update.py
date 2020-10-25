import requests
from bs4 import BeautifulSoup
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time


def main(start):
    url = 'https://movie.douban.com/top250?start='+str(start)+'&filter='
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.121 Safari/537.36 '
    }
    html = requests.get(url, headers=headers)
    return parser_html(html.text)


def parser_html(html):
    result = []
    soup = BeautifulSoup(html, 'html.parser')
    name = soup.find_all('div', class_='hd')
    pic_rank = soup.find_all('div', class_='pic')
    score = soup.find_all('div', class_='star')
    director_introduction = soup.find_all('div', class_='bd')
    for i in range(len(name)):
        temp = {
            'name': name[i].a.get_text(),
            'picture': pic_rank[i].img['src'],
            'num': pic_rank[i].em.get_text(),
            'star': score[i].find('span', class_='rating_num').get_text(),
            'director': director_parser(director_introduction[i+1]),
            'introduction': introduction_parser(director_introduction[i+1])
        }
        result.append(temp)
    return result


def director_parser(director):
    temp = director.p.get_text()
    temp = temp.replace(u"\xa0", '')
    rule = r'导演:(.*)主演:'
    result = re.search(rule, temp)
    if result:
        return result.group(1).strip(' ')
    else:
        return temp.split('...')[0].strip()


def introduction_parser(intro):
    temp = intro.find_all('p')
    if len(temp) < 2:
        return '该电影没有评论'
    else:
        return temp[1].get_text()


if __name__ == '__main__':
    # main(0)
    t1 = time()
    with open('result/movie_update.csv', 'a', newline='', encoding='UTF-8') as f:
        fieldNames = ['name', 'picture', 'num', 'star', 'director', 'introduction']
        writer = csv.DictWriter(f, fieldnames=fieldNames)
        writer.writeheader()
        pool = ThreadPoolExecutor(3)
        mission = []
        for i in range(0, 249, 25):
            print(i)
            mission.append(pool.submit(main, i))
        for item in as_completed(mission):
            for temp in item.result():
                writer.writerow(temp)
        pool.shutdown()
    t2 = time()
    # 没用线程池大约用时2秒，用了之后大约1秒
    print(t2-t1)



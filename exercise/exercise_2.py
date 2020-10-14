from bs4 import BeautifulSoup
import requests
import re
import csv


def main(start):
    # 这里遇到了反爬虫机制，返回代码418，加个头部
    url = 'https://movie.douban.com/top250?start='+str(start)+'&filter='
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.102 Safari/537.36 '
    }
    html = requests.get(url, headers=headers)
    return html_parser(html.text)


def html_parser(html):
    result = []
    soup = BeautifulSoup(html, 'html.parser')
    name = soup.find_all('div', class_='hd')
    picture_num = soup.find_all('div', class_='pic')
    star = soup.find_all('div', class_='star')
    director_introduction = soup.find_all('div', class_='bd')
    # print(director_introduction[0].p[0].get_text())
    # print(soup)
    length = len(name)
    for i in range(length):
        temp = {
            'name': name[i].a.get_text().replace('\n', ''),
            'picture': picture_num[i].img['src'],
            'num': picture_num[i].em.get_text(),
            'star': star[i].find_all('span')[1].get_text(),
            # &nbsp是html中的占位符，可以达到多个连续空格的作用，在utf-8中使用\xa0代表，可以使用replace(u"\xa0", '')去掉
            'director': director_parser(director_introduction[i+1]),
            'introduction': introduction_parser(director_introduction[i+1])
        }
        result.append(temp)

    return result


def introduction_parser(intro):
    temp = intro.find_all('p')
    if len(temp) < 2:
        return '该电影没有评论'
    else:
        return temp[1].span.get_text()


def director_parser(director):
    temp = director.p.get_text()
    temp = temp.replace(u"\xa0", '')
    rule = r'导演:(.*)主'
    result = re.search(rule, temp)
    if result:
        return result.group(1).strip(' ')
    else:
        return temp.split('...')[0].strip()


if __name__ == '__main__':
    with open('movie.csv', 'a', newline='', encoding='UTF-8') as f:
        fieldNames = ['name', 'picture', 'num', 'star', 'director', 'introduction']
        writer = csv.DictWriter(f, fieldnames=fieldNames)
        writer.writeheader()
        for i in range(0, 249, 25):
            print(i)
            result = main(i)
            print(result[0])
            for item in result:
                writer.writerow(item)
    # main(0)
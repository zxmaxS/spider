import requests
from bs4 import BeautifulSoup
import re
import csv


def main(page):
    url = 'http://bang.dangdang.com/books/fivestars/01.00.00.00.00.00-recent30-0-0-1-' + str(page)
    html = request_book(url)
    # print(html)
    return parse_result(html)  # 解析过滤我们想要的信息


def parse_result(html):
    # # 进行创建soup对象，除了这种方法，还可以直接将文件对象传入库中构造对象
    # # 第二个参数是解析器名称，如果你不传，会自动选择最合适的，但会产生警告
    soup = BeautifulSoup(html, "html.parser")
    # # 对标签可以直接直接获得第一个标签
    # temp = soup.a
    # print(temp)
    # # 打印标签的属性
    # print(temp['href'])
    # # 打印标签的全部属性
    # print(temp.attrs)
    # # 获取单个标签包含的内容
    # print(temp.string)
    # # 获取标签以及所有子标签包含的内容
    # print(temp.get_text())
    # # 对于每一个节点都可以作为一个soup对象,可以继续向下迭代,可以跨层迭代
    # temp = soup.span
    # print(temp)
    # print(temp.a)
    # # 可以通过find_all函数找到所有匹配,返回的是一个列表
    # temp = soup.span
    # print(temp.find_all('a'))
    # # find_all函数有四个参数，如果不指定，标签的参数名是name，如果传入的参数名不是默认的四个
    # # 则会被认为是标签内的属性，此外class作为python的关键词，如果使用的话要在后面加_
    # print(temp.find_all(class_='login_link'))
    # # 当属性的值为True或False时代表是否包含该属性，True为包含，False为不包含
    # print(temp.find_all(class_=False))
    # # 当属性名为data-***时，上述方法可能报错，此时使用find_all的第二个参数attrs
    # # attrs传入一个字典，可以解决上述问题
    # # 第三个参数为string,表示标签中的内容，与attrs参数均可使用正则表达式
    # # 第四个参数为limit,即查询限制个数
    # # 上述方法会检测所有子孙节点，如果只想搜索直接子节点,可以使用参数recursive=False
    # # 可以通过contents获得一阶子节点列表,通过children返回的是一个对象,可以通过for循环进行输出
    # # 其中单个元素均为soup对象,除此之外,还可以获得多阶子节点
    # print(type(temp.contents[0]))
    # for child in temp.children:
    #     print(type(child))
    # # 通过parent获得父节点，通过parents递归获得所有父节点
    # print(temp.parent)
    # # 可以通过select方法通过css选择器进行查找，语义和css一致，搜索article标签下的ul标签中的li标签
    # print(soup.select('article ul li'))
    # # 通过类名查找，两行代码的结果一致，搜索class为thumb 的标签
    # soup.select('.thumb')
    # soup.select('[class~=thumb]')
    # # 通过id查找，搜索id为sponsor的标签
    # soup.select('#sponsor')
    # # 通过是否存在某个属性来查找，搜索具有id属性的li标签
    # soup.select('li[id]')
    # # 通过属性的值来查找查找，搜索id为sponsor的li标签
    # soup.select('li[id="sponsor"]')



    num = soup.find_all('div', class_=re.compile("list_num.*"))
    name = soup.find_all('div', class_="name")
    addr = soup.find_all('div', class_='pic')
    publisher = soup.find_all('div', class_='publisher_info')
    recommend = soup.find_all('span', class_='tuijian')
    star = soup.find_all('div', class_='biaosheng')
    price = soup.find_all('span', class_='price_n')
    result = []
    for i in range(len(num)):
        temp = {
            'num': num[i].string.replace('.', ''),
            'name': name[i].a['title'],
            'addr': addr[i].img['src'],
            'publisher': publisher[i].get_text(),
            'recommend': recommend[i].string,
            'star': star[i].span.string,
            'price': price[i].string
        }
        result.append(temp)
    # exit(132)
    return result


def request_book(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None


if __name__ == '__main__':
    with open('book.csv', 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['num', 'name', 'addr', 'publisher', 'recommend', 'star', 'price']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # 写入头部，即设置列名
        for i in range(1, 26):
            result = main(i)
            for item in result:
                writer.writerow(item)
    # main(1)

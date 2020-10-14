from selenium import webdriver
# 这是用于给创建谷歌对象传参数用的
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import csv
from time import sleep


def main():
    # PhantomJS是一个无痕浏览器，需要先在官网进行下载，然后将路径添加到环境变量里
    # 然后就可以不需打开界面就能使用selenium，但是这个已经不被selenium支持了，可以直接使用谷歌的无痕浏览
    option = Options()
    # 这个参数即代表无头运行，表示不需要图形化界面，其他有用的参数还有--disable-javascript，即禁用javascript
    # --incognito代表无痕浏览，就是本地不会留下访问记录，不是没有可视化界面
    # option除了可以添加参数外，还可以加载插件，因为每次打开的浏览器均为毫无任何设置与插件的浏览器
    # --start-maximized是当无头模式时设置窗口最大化，但是好像没啥卵用，只能直接设置窗口分辨率
    # 报错的原因很可能就是分辨率不够大，你想要找的元素没有显示出来，还可以去掉下拉框，但是我目前没有遇到这种应用场景
    # 分辨率设置就是window-size，后面的中间必须是x，不能是*，就是英文字母倒数第三个
    # 还有一些其他的参数，感觉不太重要，需要的时候再去看就应该可以了
    option.add_argument("--headless")
    option.add_argument("window-size=1920x1080")
    driver = webdriver.Chrome(options=option)
    # driver = webdriver.Chrome()
    # 当使用无头运行时最大化窗口的方法就没用了，只能通过option进行传参
    # driver.maximize_window()
    driver.get("https://www.bilibili.com/")

    driver.implicitly_wait(10)

    input = driver.find_element_by_class_name("nav-search-keyword")
    input.send_keys("蔡徐坤 篮球")
    input.submit()

    all_windows = driver.window_handles
    driver.switch_to.window(all_windows[1])

    last_page = driver.find_element_by_css_selector(".page-item.last")
    last_page_num = last_page.text

    fin_return = []
    for i in range(int(last_page_num)):
    # for i in range(10):
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        result = html_parser(soup)
        fin_return.extend(result)
        print(i)
        if i != int(last_page_num) - 1:
            next_page = driver.find_element_by_xpath("//li[@class='page-item next']/button")
            next_page.click()
            sleep(3)
    # 当使用无痕浏览器时别忘了进行关闭，否则会一直存在在内存中
    driver.quit()
    return fin_return


def html_parser(soup):
    result = []
    li = soup.find_all('li', class_='video-item matrix')
    for item in li:
        title = item.a['title']
        link = item.a['href']
        temp = []
        div = item.find('div', class_='tags')
        for test in div.children:
            temp.append(test.get_text().replace(' ', '').replace('\n', ''))
        temp_temp = {
            'title': title,
            'link': link,
            '播放量': temp[0],
            '弹幕数': temp[1],
            '上传时间': temp[2],
            'up主': temp[3]
        }
        result.append(temp_temp)
    return result


if __name__ == '__main__':
    with open('bilibili.csv', 'a', newline='', encoding='UTF-8') as f:
        fieldName = ['title', 'link', '播放量', '弹幕数', '上传时间', 'up主']
        writer = csv.DictWriter(f, fieldnames=fieldName)
        writer.writeheader()
        fin_result = main()
        # print(fin_result)
        for result in fin_result:
            writer.writerow(result)

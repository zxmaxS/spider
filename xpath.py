from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://www.baidu.com")
driver.maximize_window()

# /表示从根元素选取，//表示从相对位置选取，分隔符为/，下面就表示选取所有li标签下的a标签的第二个span标签，中括号中从1开始
# temp = driver.find_elements_by_xpath('//li/a/span[2]')
# for item in temp:
#     print(item.text)

# 表示查找class为title-content-title的span标签
temp = driver.find_elements_by_xpath('//li/a/span[@class="title-content-title"]')
for item in temp:
    print(item.text)


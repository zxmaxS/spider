from selenium import webdriver
from time import sleep
# 首先需要安装包，然后在https://sites.google.com/a/chromium.org/chromedriver下载对应谷歌版本的驱动
# 将驱动放入谷歌根目录，然后将谷歌根目录加入到环境变量path中

# 这是鼠标事件需要引入的类
from selenium.webdriver.common.action_chains import ActionChains

# 这是需要组合键时引入的类
from selenium.webdriver.common.keys import Keys


# 该对象就相当于一个浏览器对象
driver = webdriver.Chrome()
driver.get("https://www.baidu.com")

# 设置隐式元素等待，等待十秒响应
driver.implicitly_wait(10)


# 设置窗口大小
# driver.set_window_size(800, 800)


# 设置窗口为最大
driver.maximize_window()

# 获取当前页面标题
# title = driver.title
# print(title)

# 获取当前页面url
# url = driver.current_url
# print(url)


# 获得当前页面的html
# html = driver.page_source


# 可以通过id,name,css选择器,class等来选择元素，和下面类似，如果想选择多个，需要在方法中element后面加个s
# clear函数用于清除输入框中的内容，send_keys函数是用于模拟键盘输入，submit函数即为模拟回车，但用处远少于下面的
# send_keys函数除了可以输入文字，对于文件上传的input标签也可以将文件路径传入以进行文件上传
# click函数
# input = driver.find_element_by_css_selector('#kw')
# input.clear()
# input.send_keys("碧蓝航线")
# input.submit()


# size用于返回html元素的大小，如下面返回输入框的大小
# size = input.size
# print(size)


# text用于获取html元素的文本
# link = driver.find_element_by_class_name('title-content-title')
# text = link.text
# print(text)


# get_attribute()用于获取属性值，参数为属性名
# types = input.get_attribute('class')
# print(types)


# click函数即为模拟点击
# button = driver.find_element_by_css_selector('#su')
# button.click()


# back函数即为页面的回退按钮，forward函数即为前进按钮，refresh函数即为刷新按钮，但是最好加个sleep否则可能反应不过来
# sleep(2)
# driver.back()
# sleep(2)
# driver.forward()
# sleep(2)
# driver.refresh()

#######################
# 鼠标事件
#######################


# 首先将浏览器对象传入类中，得到action对象
# action = ActionChains(driver)
# 上述内容大多是单个操作，当代码跑到时立即执行，而事件可以先进行存储命令，当遇到执行命令时再执行

# move_to_element，鼠标悬停，参数为想要悬停的元素，perform函数就是执行函数
# above = driver.find_element_by_css_selector("#s-usersetting-top")
# action.move_to_element(above).perform()
# driver.find_element_by_link_text("搜索设置").click()

# content_click，鼠标右击，double_click，鼠标双击，下面的操作没有反应，但是我也没有找到更好的展示点
# action.context_click(above).perform()
# sleep(5)
# action.double_click(above).perform()

# drag_and_drop()鼠标拖拽，参数是两个元素，但是第一个是原位置，第二个是目标位置，但也没有找到合适的演示
# link = driver.find_element_by_css_selector(".title-content-title")
# inputs = driver.find_element_by_id("kw")
# action.drag_and_drop(link, inputs).perform()
# 当多个操作时就按照顺序在action后面加，最后加上一个perform

#######################
# 键盘事件
#######################

# 键盘事件主要用于模拟键盘的特殊输入按键
# 输入框输入内容
# driver.find_element_by_id("kw").send_keys("seleniumm")

# 删除多输入的一个 m
# driver.find_element_by_id("kw").send_keys(Keys.BACK_SPACE)

# 输入空格键+“教程”
# driver.find_element_by_id("kw").send_keys(Keys.SPACE)
# driver.find_element_by_id("kw").send_keys("教程")

# ctrl+a 全选输入框内容
# driver.find_element_by_id("kw").send_keys(Keys.CONTROL, 'a')

# ctrl+c 剪切输入框内容
# driver.find_element_by_id("kw").send_keys(Keys.CONTROL, 'c')

# ctrl+x 剪切输入框内容
# driver.find_element_by_id("kw").send_keys(Keys.CONTROL, 'x')

# ctrl+v 粘贴内容到输入框
# driver.find_element_by_id("kw").send_keys(Keys.CONTROL, 'v')

# 通过回车键来代替单击操作
# driver.find_element_by_id("su").send_keys(Keys.ENTER)

# 除此之外Keys.TAB代表tab键，Keys.ESCAPE代表esc键，Keys.F1代表F1键

# 设置元素等待，知道有这么个方法，防止网站反应时间过长导致无法查找到元素

#######################
# 定位一组元素
#######################

# 定位一组元素就是在正常的element后面加上s，查出来的是一个数组
# sleep(1)
# text = driver.find_elements_by_xpath('//div/h3/a')
# for item in text:
#     print(item.text)

#######################
# 切换标签页
#######################

# current_window_handle方法返回当前的标签页
# current_windows = driver.current_window_handle

# 这里我就遇到了由于反应过慢找不到元素的问题，所以最好每一步都sleep一下
# driver.find_element_by_link_text("登录").click()
# sleep(5)
# driver.find_element_by_xpath("//div[@class='pass-login-pop-form']/div[@class='tang-pass-footerBar']/a").click()

# 获得所有标签页，是一个数组
# all_windows = driver.window_handles
# for item in all_windows:
#     if item != current_windows:
        # 和跳转标签类似，这里跳转标签页
#         driver.switch_to.window(item)
#         sleep(2)
#         driver.find_element_by_name("userName").send_keys("username")
#         sleep(2)
#         driver.find_element_by_id("TANGRAM__PSP_4__password").send_keys("password")

#######################
# 处理alert警告框
#######################

# 处理alert警告框主要使用switch_to.alert.accept()函数接受
# temp = driver.find_element_by_id("s-usersetting-top")
# action = ActionChains(driver)
# sleep(2)
# action.move_to_element(temp).perform()
# sleep(2)
# driver.find_element_by_link_text("搜索设置").click()
# sleep(2)
# driver.find_element_by_link_text("保存设置").click()
# sleep(2)
# driver.switch_to.alert.accept()

#######################
# 处理滚动条
#######################

# selenium并没有内置的处理滚动条的方法，但可以调用js进行滚动操作
# driver.set_window_size(500, 500)
# inputs = driver.find_element_by_id('kw')
# inputs.send_keys("碧蓝航线")
# inputs.submit()
#
# sleep(2)
# js = 'window.scrollTo(100,200)'
# driver.execute_script(js)

# 获得窗口截屏，参数为保存位置
# driver.get_screenshot_as_file()








































































































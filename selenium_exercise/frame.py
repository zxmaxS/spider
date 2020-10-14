from selenium import webdriver

driver = webdriver.Chrome()
driver.get("http://www.126.com")
driver.maximize_window()

# frame,iframe表单是一种在一个html中嵌入另一个html的标签，相当于python的import
# 由于frame是从外部引入的，webdriver无法处理frame内的标签，只能使用switch_to.frame()函数
# frame标签的id一般是变化的，所以推荐使用xpath来进行定位元素
frame = driver.find_element_by_xpath("//div[@class='loginWrap']/div/iframe")
driver.switch_to.frame(frame)
driver.find_element_by_name("email").send_keys("username")
driver.find_element_by_name("password").send_keys("password")
driver.find_element_by_id("dologin").click()
# 进入多级表单时可以使用该下面的函数跳转到最外层
driver.switch_to.default_content()



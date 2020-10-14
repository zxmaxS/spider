# 网页中的下拉框有两种，一种是利用javascript实现的假下拉框，另一种则是select标签形成的
# 对于前一种，只需要正常的点击元素就可以，对于后一种，selenium有对应的库
from selenium import webdriver
from time import sleep
# 从select中引入select
from selenium.webdriver.support.select import Select

driver = webdriver.Chrome()
driver.get("http://sahitest.com/demo/selectTest.htm")

driver.maximize_window()
# temp = driver.find_element_by_id("s1Id")
# s1 = Select(temp)

# 选择选项有三种方法，index，value，visible_text
# index从0开始计算，value既是value属性，visible_text就是能看到的文字选项
# s1.select_by_index(1)
# sleep(2)
# s1.select_by_value("o2")
# sleep(2)
# s1.select_by_visible_text("o3")

# 当想看所有选项时
# option = s1.options
# for item in option:
#     print(item.text)

# 有了选择就会有取消选择在上面的方法前面加上de即可，但这种方法只对复选框有效，deselect_all()方法就是取消全部选择
temp = driver.find_element_by_id("s4Id")
s4 = Select(temp)
s4.select_by_index(1)
sleep(2)
s4.select_by_value("o2val")
sleep(2)

# 对于包含空格的选项，从第一个非空字符开始算起，对于&nbsp，每一个都相当于一个空格
s4.select_by_visible_text("With spaces")
sleep(2)
s4.select_by_visible_text("    With nbsp")
# sleep(2)

# 对于复选框，可以查看我已经选中的选项
# option = s4.all_selected_options
# for item in option:
#     print(item.text)

# 对于单选框和复选框查看默认值或者第一个选择的选项
option = s4.first_selected_option
print(option.text)

# s4.deselect_by_index(4)
# sleep(2)
# s4.deselect_all()


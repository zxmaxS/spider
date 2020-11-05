# ssh
ssh是一种加密登录的网路协议
### 密码登录
~~~
ssh 用户名@IP地址
~~~
上面是一种普通的登录方法，如果不是默认端口的话
~~~
ssh 用户名@IP地址 -p 端口号
~~~
上面的登录方法如果成功连接服务器，会让你输入密码，
密码正确后就可以直接登录
### 密钥登录
首先在客户端创建秘钥对，再把创建的公钥复制到服务器home
目录下，再将公钥字符串追加在家目录的.ssh目录的
authorized_keys(ssh认证文件)文件末尾。
##### 创建密钥
ssh加密方式有rsa和dsa两种，使用两种方式其中一种都可以。
本文使用rsa。首先打开本地终端，
执行ssh-keygen命令创建本地秘钥对
~~~
ssh-keygen -t rsa -C "xxxx@xx.xxx"
~~~
说明：
-t 指定秘钥类型(rsa,dsa两种）默认是rsa。可选
-C 注释文字，一般多用邮箱。可选

一路回车即可。其中两次是让你输入使用公钥的密码，
不输入即不使用密码。

之后会在home目录的隐藏文件夹.ssh(即~/.ssh)下生成
id_rsa私钥和id_rsa.pub公钥两个秘钥文件，如果是windows
系统，会在C盘的用户目录下生成.ssh文件夹

##### 复制公钥到服务器
使用scp将本机公钥复制到服务器家目录
~~~
scp -P 端口号 id_rsa.pud 用户名@IP地址:~/id_rsa.pub
~~~
其中若没有修改默认端口的话 -P 端口号 可以不写，
前面为本机公钥地址，后面为复制到服务端地址。

再用密码登录服务器，使用cat将复制过来的公钥字符串追加到
authorized_keys文件末尾
~~~
cat ~/id_rsa.pub >> ~/.ssh/authorized_keys
~~~
上面步骤都完成之后，在客户端再使用下面命令登录
~~~
ssh 用户名@服务器IP地址 -p 端口号
~~~
第一次登录需要输入密码，验证成功登录之后，
之后再使用该命令登录就不需要输入密码了。
### 快捷登录
我们可以在客户端.ssh文件下的config（在.ssh文件夹中，如果
没有就创建一个）配置文件中设置服务器别名，之后用别名登录。
~~~
Host shuang
HostName 47.95.252.50
User root
IdentitiesOnly yes
~~~
##### 其中Host是别名，HostName是IP地址，别弄反了
之后登录服务器就可以使用别名登录了
~~~
ssh shuang
~~~
 

 
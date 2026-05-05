# Pyhton基础
语法基础
字面量，变量，常量
Python全大写默认常量，但可修改
注释
# coding=utf-8    文件开头表明编码，python3默认，不用写
# ASCII码是英语国家用的
# ISO885-1 欧州
# GB2312 中国（只有7000左右常用的汉字和符号
# GBK 有两万多种
# uft-8全世界通用
单行、多行注释
# 单行
'''多行注释'''
数据类型 
整形、浮点、字符串、布尔
# Python中变量无类型，数据有类型
    #整形
    res_int=100
    res_float=100.99
        # 可分组，每三位为一组,不影响结果，方便看，python编译器自己会看
        res=100_000_000

        # 数字转字符串位数限制
        import sys
        res=9**999999999
        sys.set_int_max_str_digits(0) # 0表示数字转字符串不限制长度
        print(res) 
    
    # 精度控制
     %m.nd % intger_name
     # m表示对总体的控制，最小宽度，位数少于实际位数不起作用，正数右对齐，负数左对齐
     # n表示精度控制，对于浮点数表示小数点后得精度，对于字符串表示输出的最大宽度，对整数来说就是前补零
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
#浮点型，e必须有大小写都可以，+可写可不写
    speed_of_sound=3.4e+2 # 3.4*10^2
    speed_of_sound=3.4e2 # 3.4*10^2
    speed_of _light=2.998E+8 #光速
    
        # 负浮点数
         one_ml=1e-3 #1*10^-3   
    # 精度控制
    %m.ns
    # 1.m表示最小宽度，不够用空格不全，小于串的长度则不起作用
    # 2.m为正表示右对齐，m位负表示左对齐
    # 3.n表示精度，不够补零，截断会四舍五入
    

#字符串
     单引号，双引号不可直接换行，三引号可以直接换行
         
         #字符串格式化输出
         字符串在拼接时，只能是字符串相互拼接，不能直接用整形和浮点类型用+拼接，需要类型转换
         
         # 一般取类型的首字母
         # %s 字符串,%s也是万能的，Python会自己转
         # %i 整形，%d十进制整形
         # %f 浮点型
    # 三种写法，1.直接用加号进行拼接。2.使用占位符。3.使用f'string'
    # 这里只展示第三种，name,age都直接是变量名
    info=f'你好我叫{name},我今年{age}岁了' 
    
    # 字符串占位符
    %m.ns
    # 1.m表示最小宽度，不够用空格不全，小于串的长度则不起作用
    # 2.m为正表示右对齐，m位负表示左对齐
    # 3.n表示最多输出几位，n若是大于实际长度，则不起作用
        
        # eg
        weight="张三"
        print('我叫%1s' % weight) # 不起作用
        print('我叫%4s克' % weight) # 右对齐，空格补齐
        print('我叫%-4s克' % weight) # 左对齐，空格补齐
        weight="张三"
        print('我叫%.1s' % weight) # 只有一位
        print('我叫%.3s啊' % weight)  # 不起作用
        
    # python中字符串也可以用+=
    text='北京'
    text+='欢迎你'
    print(text)    

























转义字符
\'    代表一个单引号
\"    代表一个双引号
\\    代表一个反斜杠
\n    换行
\r    返回光标至首行，覆盖之前的输出
\f    换页
\v    垂直制表符
\t    水平制表符
    print('ab\tcd'.expandtabs(4)) # expandtabs空制表符长度
\b    删除一个字符再打印
\0    空字符 Null
\0oo    三位的八进制数表示的字符
\xXX    两位的十六进制数表示的字符

数据类型转换
# 转str
str(18)
# 转int,直接丢小数部分，
int(15.6) 
int('16.2') # 识别不了
# 转float
float('15.60')
float('17. 6') #识别不了
# 转bool
bool(1)
# 查看字符的Unicode编码
ord('a') 
# 将Unicode编码转为对应的字符
chr(98) 
算术运算符
+ # 加
- # 减
* # 乘
/ # 除
// # 取整除
% # 取模
** # 方


逻辑运算符
# 返回的不一定是布尔值，是某个参与计算的值本身
# 若参与运算的不是bool，python会自动转成bool

# 规则 and先看左边，左为假就直接返回左的结果，否则返回右边的结果
and  ==&&  

# 有一个true即可
# or也是先看左边，左为真就直接返回左的结果，否则返回右边的结果
or   ==||

not  ==!
进制转换
# python在打印时会自动转为十进制
# 二进制
num=0b101010
# 八进制
num=0o456255
# 十六进制
num=0xA526B

#十转二转为字符串
bin()
#十转八转为字符串
oct()
#十转十六转为字符串
hex()
# 将其他进制数转为十进制的int
value=int('010111',2)
value=int('035641',8)
value=int('0ab231',16)
流程控制语句
age = int(input("What is your age? "))
age = int(input("What is your age? "))
if age < 18:
    print("You are a minor.")
elif 18 <= age < 65:
    print("You are an adult.")
else:
    print("You are a senior.")

# for循环
# range左闭右开区间，默认从零开始
for i in range (1,10):
    print(i)
for j in range (10):
    print(j)
函数
定义格式
def function():
    print('hello')
传参方式
# 定义函数
def order(user,dish,num,time):
    print(f'新订单：{dish}，{num}份')
order('01_user','鱼香肉丝'，3,20260504)


# 函数传参可默认位置，也可以用关键字指定传参
# 二者也可以混用，位置参数要在关键字参数之前


# 限制传参
# /前只能用位置参数，*后只能用关键字参数
def order(user,dish,/,num,time):
order('01_user','辣子鸡',3,time='20260504')
参数默认值
# 直接在参数定义赋值默认值
参数默认值一定要放在最后，也就是参数默认值后面都要是参数默认值
# 设置默认值的参数也就成为了可选参数
def order(user,dish,num,time='2026050'):
    print(f'新订单：{dish}，{num}份')
可变参数，可变关键字参数
也是参数形式，可以和其它参数随意搭配使用，越复杂的就越靠后
位置参数》可变位置参数（*key）》关键字参数（key='')》默认参数(key='123')》可变关键字参数(**key)
在形参名前加*可接受多个，也就是数组（元组）
在形参前加*，就能接受多个key-value参数，就是字典
# 参数前加*就是接收多个
# 可变位置参数一定要在前
def order(user,**dish,-num,time='2026050'):
    print(f'新订单：{dish}，{num}份')
    
# 特殊值   其类型位NoneType
None 

# 取反是not,没有!











函数返回值

python
脚本语言，解释性，交互式，面向对象





函数
#求长度函数
len(a)
#不换行输出
print(a,end="")
def function(a,b){
    return a*b
}
读写文件
text="hello world"
fp=open("hello.txt",'w')  #w写，r读，a追加
fp.write(text)
fp.close()


fp1=open('hello.txt','r')
content=fp1.read()
content=fp.readline()#读一行
content=fp1.readlines()
类
class person:
    #默认属性，调用初始化修改
    name:'梨花'
    age:19
    #类实体初始化方法
    def __init__(name,age)
        self.name=name
        self.age=age
    def add(a,b)
        return a+b
输入
text=input("please input number")
元组和列表
#元组,元组不可更改，列表可以更改
a_tuple=(12,3,4,5)
another_tuple=2,3,4,5,6,78
a_list=[1,2,3,4,5]
for num in a_list 

#列表
a=[2,4,6,8,9]
a.append(0)
print(a)
#在1的位置上加个0
a.insert(1,0)
#去掉值为2的元素
a.remove(2)
#第三位到第六位不包括6
[起始:终止(不包括):步长]
a[3:6:2]

#传负值就是从后往前
#倒序排
a.sort(reverse=True)
字典
#key:value
d={'name':'111',
'age':{'虚岁':24,'周岁':'23'},
id:111}
print(d['age']['周岁'])
错误处理
try:
    file=open('www','r')
except Exception as e:
    print(r)
zip lambda
a=[1,2,3]
b=[4,5,6]
zip(a,b)
list(zip(a,b))
out:[(1,4),(2,5),(3,6)]

#定义运算方程的简易函数
fun2=lambda x,y:x+y

list(map(fun2,[1,3],[2,5]))
out:[3,8]
浅复制和深度复制
import copy

a=[1,2,[3,4]]
b=a
id(a)==id(b)
a和b是同一个地址，不同索引
copy.copy会拷贝第一层数据，更深层的数据不是深度复制
c=copy.copy(a)
copy.deepcopy是深度拷贝

pickle
import pickle
a_dict = {'da': 111, 2: [23,1,4], '23': {1:2,'d':'sad'}}
##fle = open('pickle_example.pickle', 'wb’)
##pickle.dump(a_dict, fle)
##fle.close()
with open('pickle_example.picke', 'rb') as fle
    a_dict1 =pickle.load(fle)
fle.close(print(a_dict1)
set找不同
c_list=['a','a','b','c','b','c','d']
print(set(c_list))
print(type(set(c_list)))
sentence='Welcome Back To This Tutirial'
print(set(c_list))


正则表达式
import re
#multiple patterns("run" or "ran")

#加r是表达式，[]内是多选
ptn = r"r[au]p"print(re.search(ptn,"dog runs to cat"))
None

print(re.search(r"r[Ã-z]n","dog runscat"))
print(re.search(r"r[a-z]n","dog runs to cat"))
print(re.search(r"r[0-9]n","dog r2ns to cat"))
print(re.search(r"r[0-9a-z]n"，"dog runs to cat"))
None
<sre.SRE Match object;span=(4，7)match='run'>
<sre.SRE Match object;span=(4，7)match='r2n'>
<sre.SRE Match object;span=(4，7)match='run'>

数字
#\d:数字形式
print(re.search(r'r\dn'"run r4n"))
#\D:字母形式
print(re.search(r"r\Dn","run r4n"))
sre.SRE Matchobject;span=(4，7)，match='r4n'>
sre.SRE Matchobject;span=(0，3)，match='run'>

#\s :any white space
print(re.search(r"r\sn","r\nn r4n"))
#S:opposite to \s,any non-white space
print(re.search(r"r\Sn","r\nn r4n"))
<sre.SRE Matchobject;span=(0，3)match='r\nn'>
<sre.SRE Match object;span=(4，7)match='r4n'>

#\w:[a-zA-20-9 ]
print(re.search(r"r\wn","r\nn r4n"))
#\W:opposite to \w
print(re.search(r"r\Wn","r\nn r4n"))
<sre.SRE Matchobject;span=(4，7),match='r4n'>
<sre.SRE Match object;span=(0，3)，match='r\nn'>

#\b :empty string(only at the start or end of the word)
print(re.search(r"\b runs b","dog run to cat"))
#\B:empty string(but not at the start or end of a word)
print(re.search(r"\B runs B","dog run to cat"))
None
sre.SRE Match object;span=(5,1l),match='runs

#\\:match \
print(re.search(r"runs\\","runs\ to me"))
# . :match anything(except \n)
print(re.search(r"r.n","r-ns to me"))
<sre.SRE Match object;span=(0，5)match='runs\\'>
<sre.SRE Match object;span=(0，3)，match='r-n'>

句尾句首
#^:match line beginning
print(re.search(r"^dog"," dogruns to cat"))
#$:match line ending
print(re.search(r"cat$""dog runs to cat"))
<sre.SRE Match object;span=(0，3)，match='dog'>
<sre.SRE Match object;span=(12，15)，match='cat'>


#?:may or may not occur
print(re.search(r"Mon(day)?"，"Monday"))
print(re.search(r"Mon(day)?","Mon"))
<sre.SRE Match object;span=(0，6)，match='Monday'>
<sre.SRE Match object;span=(0，3)，match='Mon'>


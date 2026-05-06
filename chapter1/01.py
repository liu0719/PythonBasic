# 定义函数
def order(user,dish,/,num,*,time):
    print(f'新订单：{dish}{num}份')
order('01_user','鱼香肉丝',3,time='20260504')

# 函数传参可默认位置，也可以用关键字指定传参
# 二者也可以混用，位置参数要在关键字参数之前

# 限制传参方式，/前只能用位置参数，*后只能用关键字参数
# /前只能用位置参数，*后只能用关键字参数
order('01_user','辣子鸡',33,time='20260504')
# 这是新增的，你看得见吗
def add(a,b):
    '''
    计算两个数相加的和
    :param a: 第一个数
    :param b: 第二个数
    :return: 两个数的和
    '''
    return a + b
print(add(1,2))
help(int) 
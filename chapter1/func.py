def demo(*args):
    print(args) 

l1 = [1,2,3,4,5]    
t1=(1,2,3,4,5)
demo(l1)
demo(t1)

# 这种写法相当于demo(1,2,3,4,5)，
demo(*l1)
demo(*t1)
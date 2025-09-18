# 加减乘除  传入  格式
#
# a = 3
# b = 2
# print(a+b)
# print(a-b)
# print(a*b)
# print(a/b)
# print(a//b)   #地板除
# print(a**b)

# #返回A的平方
# def myfun(A):
#     C = A**2
#     return C

# #返回A的B次方
# def myfun(A, B):
#     return A**B


#如果传入B，就返回A的B次方, 如果没传入B，返回A的平方
def myfun(A, B=2):
    return A**B
a = 3

print(myfun(a))
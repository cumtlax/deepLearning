#深度学习， 模型   东西 集成很多属性和自己的函数。 是这个的类的实例， 就可以用类的东西。
#__init__


class person():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def print_name(self):
        print(self.name)

    def print_age(self):
        print(self.age)


# laoli = person("ligekaoyan", 28) #实例化

# print(laoli.print_name())

#继承  本身是个人
class superman(person):
    def __init__(self, name, age):
        super(superman, self).__init__(name, age)
        self.fly_ = True
    def fly(self):
        if self.fly_ == True:
            print("我会飞！")



lige = superman("lige", "28")
lige.print_name()
print(lige.name)
lige.fly()










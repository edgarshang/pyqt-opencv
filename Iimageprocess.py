from abc import ABCMeta, abstractmethod

class Process:
    __metaclass__ = ABCMeta
 
    def __init__(self):
        pass
 
    @abstractmethod
    def imageprocess(self, info):
        print("hello, world") 
        pass

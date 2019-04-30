import sys
class baseMethod():

    def __init__(self,*args):
        self.mandatoryArgCheck(*args)

    def mandatoryArgCheck(*args):
        for arg in args:
            if(arg == None):
                raise Exception("A Mandatory arg is null")
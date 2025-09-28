import os
import time

import numpy as np


class Logger(object):

    def __init__(self,strftime=None,base_path=r"records/"):
        if not strftime:
            strftime=time.strftime("%Y%m%d%H%M%S",time.localtime()) 

        self.path = f"{base_path}/{strftime}_output.log"
    

    def log(self,data:dict,save=True):
        s=""
        for k in data:
            if k=="feature_selected":
                continue
            
            s+=f"{k}:{data[k]} "
        if save:
            with open(self.path,'a') as f:
                f.write(s+"\n")
        
        return s


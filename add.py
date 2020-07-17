'''
@Author: Magician
@Date: 2020-07-17 18:50:06
@LastEditors: HK
@LastEditTime: 2020-07-17 20:35:16
@Description: file content
@FilePath: \add\add.py
'''

import numpy as np

a = np.load('a.npy')
b = np.load('b.npy')
c = []


c = np.append(a,b)


np.save('merge.npy',c)

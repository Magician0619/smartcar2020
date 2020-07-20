'''
@Author: Magician
@Date: 2020-07-17 18:50:06
@LastEditors: HK
@LastEditTime: 2020-07-20 23:10:54
@Description: file content
@FilePath: \smartcar2020\add.py
'''

import numpy as np

#E:\智能车\人工智能组资料\人工智能创意赛复赛\数据集\赛道线\车道线0720merge\下午

a = np.load('E:/智能车/人工智能组资料/人工智能创意赛复赛/数据集/赛道线/车道线0720merge/下午/data.npy')
b = np.load('E:/智能车/人工智能组资料/人工智能创意赛复赛/数据集/赛道线/车道线0720merge/中午/data.npy')
c = []


c = np.append(a,b)


np.save('0720merge.npy',c)

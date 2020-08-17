import cv2

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
  file = open(filename,'a')
  for i in range(len(data)):
    s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
    s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符
    file.write(s)
  file.close()
  print("保存文件成功") 


a = cv2.imread  ("DFS/0.jpg")
cv2.imshow("djfks",a)
img_list = a.tolist()#将图片转换成数组列表格式
text_save("DFS/maze.txt",img_list)
print(img_list)
cv2.waitKey(0)
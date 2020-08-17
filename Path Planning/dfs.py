import cv2

MIN = 1


a = [[0 for col in range(500)] for row in range(500)]#迷宫最大数组
print("debug info:",type(a))
book = [[0 for col in range(500)] for row in range(500)]#标记数组

def dfs(start_x,start_y,end_x,end_y,migong_array,step):
    '''
    :param start_x: 起始横坐标
    :param start_y: 起始纵坐标
    :param end_x: 终点横坐标
    :param end_y: 终点纵坐标
    :param migong_array: 迷宫的数组
    :return:
    '''
    next_step = [[0,1],  #向右走
            [1,0],  #向下走
            [0,-1], #向左走
            [-1,0]  #向上走
            ]
    if (start_x == end_x and start_y == end_y):
        global MIN
        if(step < MIN):
            MIN = step
        return

    for i in range(len(next_step)):
        next_x = start_x + next_step[i][0]
        next_y = start_y + next_step[i][1]

        if(next_x < 0 or next_y < 0 or next_x > len(migong_array) or next_y > len(migong_array[0])):
            continue
        if(a[next_x][next_y] == 0 and book[next_x][next_y] == 0):
            book[next_x][next_y] = 1
            dfs(next_x,next_y,end_x,end_y,migong_array,step+1)
            book[next_x][next_y] = 0
    return

if __name__ == '__main__':
    start_x = 160
    start_y = 0
    end_x = 160
    end_y = 240

    img = cv2.imread("DFS/0.jpg")
    migong_array = img.tolist()

    # migong_array = [[0,0,1,0],[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]   #初始化迷宫

    for i in range(len(migong_array)):
        for j in range(len(migong_array[0])):
            a[i][j] = migong_array[i][j]  #将迷宫数组写入a中
    book[start_x][start_y] = 1  #将第一步标记为1，证明走过了。避免重复走

    dfs(start_x,start_y,end_x,end_y,migong_array,0)

    print('The min length of path is : {}'.format(MIN)) #输出为7，即最短路径为 7 
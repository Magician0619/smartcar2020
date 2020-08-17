# 邻接矩阵实现稠密图
class DenseGraph:
    def __init__(self, n, directed):
        self.__m = 0  # 边数
        self.__n = n  # 顶点个数
        self.__directed = directed  # 是否有向图
        self.g = []
        for i in range(n):
            self.g.append(list(False for vec in range(n)))
 
    def V(self):
        return self.__n
 
    def E(self):
        return self.__m
 
    def addEdge(self, v, w):
        assert 0 <= v < self.__n
        assert 0 <= w < self.__n
 
        # 若已经存在边
        if self.hasEdge(v, w):
            return
 
        self.g[v][w] = True
        if (not self.__directed):
            # 非有向图
            self.g[w][v] = True
        self.__m += 1
 
    def hasEdge(self, v, w):
        assert (v >= 0 and v < self.__n)
        assert (w >= 0 and w < self.__n)
        return self.g[v][w]
 
    # 打印邻接矩阵
    def show(self):
        for v in range(self.__n):
            print(('vertex%d: %s') % (v, str(self.g[v])))
 
    # 返回v节点的邻接节点
    def getAdj(self, v):
        retList = []
        adj_list = self.g[v]
        for i in range(len(adj_list)):
            if adj_list[i] == True:
                retList.append(i)
        return retList
 
 
 
 
# 稀疏图
class SparseGraph:
    def __init__(self, n, directed):
        self.__n = n
        self.__m = 0
        self.__directed = directed
        self.g = []
        for i in range(n):
            self.g.append([])
 
    def V(self):
        return self.__n
 
    def E(self):
        return self.__m
 
    def addEdge(self, v ,w):
        assert (v >= 0 and v < self.__n)
        assert (w >= 0 and w < self.__n)
 
        self.g[v].append(w)
        if (v != w and not self.__directed):
            self.g[w].append(v)
        self.__m += 1
 
    def hasEdge(self, v, w):
        assert (v >= 0 and v < self.__n)
        assert (w >= 0 and w < self.__n)
 
        # # 不包平行边处理 复杂度 O(N) 一般单独处理 此处注释
        # if self.hasEdge(v, w):
        #     return
        for i in range(self.__n):
            if w == self.g[v][i]:
                return True
            else:
                return False
 
    # 打印邻接表
    def show(self):
        for v in range(self.__n):
            print(('vertex%d: %s')% (v, str(self.g[v])))
 
    # 返回v节点的邻接节点
    def getAdj(self, v):
        return self.g[v]
 
 
# 读取文件生成图数据结构的类
class ReadgRraph:
    # 从文件filename中读取图的信息, 存储进图graph中
    def __init__(self, graph, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        f.close()
        assert len(lines) >= 1, '文件格式错误'
        self.V = int(lines[0].split(' ')[0])
        self.E = int(lines[0].split(' ')[1])
        edges = []
        for line in lines[1:]:
            edges.append(line.split(' '))
        assert self.V == graph.V(), '文件和图的顶点个数不一致'
 
        # 读取每一条边的信息
        for edge in edges:
            a = int(edge[0])
            b = int(edge[1])
            assert 0 <= a < self.V
            assert 0 <= a < self.V
            graph.addEdge(a, b)
 
 
# 图的深度优先遍历类
class Component:
    # 构造函数 求出无权图的联通分量
    def __init__(self, graph):
        self.__graph = graph
        self.__ccount = 0  # 记录连通分量的个数
        self.__visited = []  # 记录dfs过程中节点是否被访问
        self.__id = []  # 每个节点对应的连通分量的标记
        # 初始化两个指示数组
        for i in range(self.__graph.V()):
            self.__visited.append(False)
            self.__id.append(-1)
        # 求图的连通分量
        for i in range(self.__graph.V()):
            if not self.__visited[i]:
                self.__dfs(i)
                self.__ccount += 1
 
    # 图的深度优先遍历
    def __dfs(self, v):
        self.__visited[v] = True
        self.__id[v] = self.__ccount
        for i in self.__graph.getAdj(v):
            # i为v节点的临接点
            if not self.__visited[i]:
                self.__dfs(i)
 
    # 返回图的连通分量个数
    def count(self):
        return self.__ccount
 
    # 查询点v和是否联通
    def isConnect(self, v, w):
        assert v >= 0 and v < self.__graph.V()
        assert w >= 0 and w < self.__graph.V()
        return self.__id[v] == self.__id[w]
 
 
# 利用深度优先遍历进行图的寻路
class Path:
    # 构造函数 求出从s到任意一个点的路径
    def __init__(self, graph, s):
        assert 0 <= s < graph.V(), '源结点不在图中'
        self.__graph = graph
        self.__s = s  # 记录连通图的源节点
        self.__visited = []  # 记录dfs过程中节点是否被访问
        self.__from = []  # 每个结点从哪个结点过来
        # 初始化两个指示数组
        for i in range(self.__graph.V()):
            self.__visited.append(False)
            self.__from.append(-1)
        # 寻路算法
        self.__dfs(s)
 
    # 图的深度优先遍历
    def __dfs(self, v):
        self.__visited[v] = True
        for i in self.__graph.getAdj(v):
            # i为v节点的临接点
            if not self.__visited[i]:
                self.__from[i] = v
                self.__dfs(i)
 
    # 查询从s结点到w结点是否有路径
    def hasPath(self, w):
        assert 0 <= w < self.__graph.V(), 'w结点不在图中'
        return self.__visited[w]  # 如果该为True 证明从s访问过w
 
    # 查询从s到w的路径，返回一个存放路径的list
    def path(self, w):
        assert self.hasPath(w), '从源结点到该结点不存在路径'
        # 通过from数组你想查找从s到w的路径 存放到栈中 此处用list作为一个栈
        p = w
        stack = []
        while p != -1:
            stack.append(p)
            p = self.__from[p]
        stack.reverse()
        return stack
 
    def showPath(self, w):
        retVec = self.path(w)
        for i in range(len(retVec)-1):
            print(retVec[i], end='-')
        print(retVec[-1])
 
 
from queue import Queue
# 寻找无权图的最短路径
class ShortestPath:
 
    def __init__(self, graph, s):
        self.__graph = graph
        assert 0 <= s < self.__graph.V()
        self.__s = s
        self.__visited = []
        self.__from = []
        self.__ord = []  # 该点到源结点的最短路径长度
        for i in range(self.__graph.V()):
            self.__visited.append(False)
            self.__from.append(-1)
            self.__ord.append(-1)
 
        # 无向图最短路径算法，从s开始广度优先遍历整张图
        q = Queue()
        q.put(self.__s)
        self.__visited[self.__s] = True
        self.__ord[self.__s] = 0
        while not q.empty():
            v = q.get()  # 出队一个结点
            for i in self.__graph.getAdj(v):
                # i为结点v的邻接节点
                if not self.__visited[i]:
                    # 没有遍历该节点
                    q.put(i)
                    self.__visited[i] = True
                    self.__from[i] = v
                    self.__ord[i] = self.__ord[v] + 1
 
    def hasPath(self, w):
        assert 0 <= w < self.__graph.V()
        return self.__visited[w]
 
    def path(self, w):
        assert 0 <= w < self.__graph.V()
 
        stack = []
        p = w
        while p != -1:
            stack.append(p)
            p = self.__from[p]
        # 依此取出stack中的元素
        stack.reverse()
        return stack
 
    def showPath(self, w):
        retVec = self.path(w)
        for i in range(len(retVec)-1):
            print(retVec[i], end='-')
        print(retVec[-1])
 
    def length(self, w):
        assert 0 <= w < self.__graph.V()
        return self.__ord[w]
 
 
 
 
 
if __name__ == '__main__':
    filename = 'DFS/maze.txt'
    g1 = SparseGraph(13, False)
    readGraph1 = ReadgRraph(g1, filename)
    g1.show()
    component1 = Component(g1)
    print(component1.count())
 
    filename2 = 'DFS/testG.txt'
    g2 = SparseGraph(7, False)
    readGraph2 = ReadgRraph(g2, filename2)
    g2.show()
    component2 = Component(g2)
    print(component2.count())
    path2 = Path(g2, 0)
    path2.showPath(6)
 
 
    path3 = ShortestPath(g2, 0)
    path3.showPath(6)
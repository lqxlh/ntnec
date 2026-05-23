class BS:
    #初始化基站：编号、总频率、坐标X/Y、高度H
    def __init__(self, id, F_BS, X, Y, H):
        self.id = id             #编号
        self.F_BS = F_BS         #总频率
        self.res_F = self.F_BS   #剩余频率
        self.X = X
        self.Y = Y
        self.H = H

    # 核心计算：给一段任务，算传输时间T + 能耗E
    def computing(self, B, C, f_BS):
        T = B * C / f_BS                  #时间公式
        E = 1e-27 * pow(f_BS, 2) * B * C  #能耗公式
        return T, E

    # 重置：把剩余频率恢复成满的
    def reset(self):
        self.res_F = self.F_BS

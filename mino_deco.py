from gurobipy import *
import numpy as np

class MINO_DECO_2layer():
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
    
        self.slope_sm = [1/8, 1/4, 1/8]
        self.intercept_sm = [3/8, 1/2, 5/8]

    def build_MP(self, X, y, w_2, b_2):
        """
        w_2: [hidden_dim]
        b_2: [hidden_dim]

        """

        N = X.shape[0]

        self.model_MP = Model()

        self.z = self.model_MP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.y_pred = self.model_MP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.h = {}
        self.r_relu = {}
        for k in range(N):
            for i in range(self.hidden_dim):
                self.h[k, i] = self.model_MP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="h({%s},{%s})" % (k, i))
                self.r_relu[k, i] = self.model_MP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="r_relu({%s},{%s})" % (k, i))

        self.w_1 = {}
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                self.w_1[i, j] = self.model_MP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="W_1({%s},{%s})" % (i, j))
        self.b_1 = self.model_MP.addVars(self.hidden_dim, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.r = self.model_MP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.p_relu = {}
        self.q_relu = {}
        for k in range(N):
            for i in range(self.hidden_dim):
                for j in range(2):
                    self.p_relu[k, i, j] = self.model_MP.addVar(vtype=GRB.BINARY, name="p_relu({%s},{%s},{%s})" % (k, i, j))
                    self.q_relu[k, i, j] = self.model_MP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q_relu({%s},{%s},{%s})" % (k, i, j))
            
                self.model_MP.addConstr(quicksum(self.p_relu[k, i, j] for j in range(2)) == 1)
                self.model_MP.addConstr(quicksum(self.q_relu[k, i, j] for j in range(2)) == self.h[k, i])

                self.model_MP.addConstr(-100*self.p_relu[k, i, 0] <= self.q_relu[k, i, 0])
                self.model_MP.addConstr(self.q_relu[k, i, 0] <= 0)

                self.model_MP.addConstr(0 <= self.q_relu[k, i, 1])
                self.model_MP.addConstr(self.q_relu[k, i, 1] <= 100*self.p_relu[k, i, 1])

                self.model_MP.addConstr(self.r_relu[k, i] == self.q_relu[k, i, 1])

        self.p = {}
        self.q = {}
        for k in range(N):
            for i in range(5):
                self.p[k, i] = self.model_MP.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                self.q[k, i] = self.model_MP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
            
            self.model_MP.addConstr(quicksum(self.p[k, i] for i in range(5)) == 1)
            self.model_MP.addConstr(quicksum(self.q[k, i] for i in range(5)) == self.y_pred[k])

            self.model_MP.addConstr(-100*self.p[k, 0] <= self.q[k, 0])
            self.model_MP.addConstr(self.q[k, 0] <= -3*self.p[k, 0])

            self.model_MP.addConstr(-3*self.p[k, 1] <= self.q[k, 1])
            self.model_MP.addConstr(self.q[k, 1] <= -1*self.p[k, 1])

            self.model_MP.addConstr(-1*self.p[k, 2] <= self.q[k, 2])
            self.model_MP.addConstr(self.q[k, 2] <= self.p[k, 2])

            self.model_MP.addConstr(self.p[k, 3] <= self.q[k, 3])
            self.model_MP.addConstr(self.q[k, 3] <= 3*self.p[k, 3])

            self.model_MP.addConstr(3*self.p[k, 4] <= self.q[k, 4])
            self.model_MP.addConstr(self.q[k, 4] <= 100*self.p[k, 4])

            self.model_MP.addConstr(self.r[k] == self.slope_sm[0]*self.q[k, 1] + self.intercept_sm[0]*self.p[k, 1] + \
                                    self.slope_sm[1]*self.q[k, 2] + self.intercept_sm[1]*self.p[k, 2] + \
                                    self.slope_sm[2]*self.q[k, 3] + self.intercept_sm[2]*self.p[k, 3] + \
                                    self.p[k, 4]
                                )

        for k in range(N):
            self.model_MP.addConstr(self.y_pred[k]==quicksum(w_2[i]*self.r_relu[k, i] for i in range(self.hidden_dim)) + b_2)

            self.model_MP.addConstr(self.z[k]>=y[k]-self.r[k])
            self.model_MP.addConstr(self.z[k]>=-(y[k]-self.r[k]))

            for j in range(self.hidden_dim):
                self.model_MP.addConstr(self.h[k, j]==quicksum(self.w_1[i, j]*X[k, i] for i in range(self.input_dim)) + self.b_1[j])

        self.model_MP.update()
        self.model_MP.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model_MP.update()

    def build_SP(self, X, y, w_1, b_1):
        """
        w_1: [input_dim, hidden_dim]
        b_1: [hidden_dim]
        
        """

        N = X.shape[0]

        self.model_SP = Model()

        self.z = self.model_SP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.y_pred = self.model_SP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.h = {}
        self.r_relu = {}
        for k in range(N):
            for i in range(self.hidden_dim):
                self.h[k, i] = self.model_SP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="h({%s},{%s})" % (k, i))
                self.r_relu[k, i] = self.model_SP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="r_relu({%s},{%s})" % (k, i))

        self.w_2 = self.model_SP.addVars(self.hidden_dim, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.b_2 = self.model_SP.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.r = self.model_SP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.p_relu = {}
        self.q_relu = {}
        for k in range(N):
            for i in range(self.hidden_dim):
                for j in range(2):
                    self.p_relu[k, i, j] = self.model_SP.addVar(vtype=GRB.BINARY, name="p_relu({%s},{%s},{%s})" % (k, i, j))
                    self.q_relu[k, i, j] = self.model_SP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q_relu({%s},{%s},{%s})" % (k, i, j))
            
                self.model_SP.addConstr(quicksum(self.p_relu[k, i, j] for j in range(2)) == 1)
                self.model_SP.addConstr(quicksum(self.q_relu[k, i, j] for j in range(2)) == self.h[k, i])

                self.model_SP.addConstr(-100*self.p_relu[k, i, 0] <= self.q_relu[k, i, 0])
                self.model_SP.addConstr(self.q_relu[k, i, 0] <= 0)

                self.model_SP.addConstr(0 <= self.q_relu[k, i, 1])
                self.model_SP.addConstr(self.q_relu[k, i, 1] <= 100*self.p_relu[k, i, 1])

                self.model_SP.addConstr(self.r_relu[k, i] == self.q_relu[k, i, 1])

        self.p = {}
        self.q = {}
        for k in range(N):
            for i in range(5):
                self.p[k, i] = self.model_SP.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                self.q[k, i] = self.model_SP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
            
            self.model_SP.addConstr(quicksum(self.p[k, i] for i in range(5)) == 1)
            self.model_SP.addConstr(quicksum(self.q[k, i] for i in range(5)) == self.y_pred[k])

            self.model_SP.addConstr(-100*self.p[k, 0] <= self.q[k, 0])
            self.model_SP.addConstr(self.q[k, 0] <= -3*self.p[k, 0])

            self.model_SP.addConstr(-3*self.p[k, 1] <= self.q[k, 1])
            self.model_SP.addConstr(self.q[k, 1] <= -1*self.p[k, 1])

            self.model_SP.addConstr(-1*self.p[k, 2] <= self.q[k, 2])
            self.model_SP.addConstr(self.q[k, 2] <= self.p[k, 2])

            self.model_SP.addConstr(self.p[k, 3] <= self.q[k, 3])
            self.model_SP.addConstr(self.q[k, 3] <= 3*self.p[k, 3])

            self.model_SP.addConstr(3*self.p[k, 4] <= self.q[k, 4])
            self.model_SP.addConstr(self.q[k, 4] <= 100*self.p[k, 4])

            self.model_SP.addConstr(self.r[k] == self.slope_sm[0]*self.q[k, 1] + self.intercept_sm[0]*self.p[k, 1] + \
                                    self.slope_sm[1]*self.q[k, 2] + self.intercept_sm[1]*self.p[k, 2] + \
                                    self.slope_sm[2]*self.q[k, 3] + self.intercept_sm[2]*self.p[k, 3] + \
                                    self.p[k, 4]
                                )

        for k in range(N):
            self.model_SP.addConstr(self.y_pred[k]==quicksum(self.w_2[i]*self.r_relu[k, i] for i in range(self.hidden_dim)) + self.b_2)

            self.model_SP.addConstr(self.z[k]>=y[k]-self.r[k])
            self.model_SP.addConstr(self.z[k]>=-(y[k]-self.r[k]))

            for j in range(self.hidden_dim):
                self.model_SP.addConstr(self.h[k, j]==quicksum(w_1[i, j]*X[k, i] for i in range(self.input_dim)) + b_1[j])


        self.model_SP.update()
        self.model_SP.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model_SP.update()

    def fit(self):
        #self.model_MP.optimize()
        self.model_SP.optimize()


if __name__ == "__main__":
    X = np.load('toy_X.npy')
    y = np.load('toy_y.npy')
    N = X.shape[0]

    """
    w_2 = [0.2385, 1.7171]
    b_2 = -2.8286

    #mino = MINO_sigmoid_v2()
    mino = MINO_DECO_2layer()
    mino.build_MP(X, y, w_2, b_2)
    mino.fit()

    for i in range(mino.input_dim):
        print("optimal_w_1: ", [mino.w_1[i, j].X for j in range(mino.hidden_dim)])

    optimal_b_1 = [mino.b_1[i].X for i in range(mino.hidden_dim)]
    print("optimal_b_1: ", optimal_b_1)

    optimal_y_pred = [mino.y_pred[i].X for i in range(N)]
    print("optimal_y_pred: ", optimal_y_pred)

    optimal_r = [mino.r[i].X for i in range(N)]
    print("optimal_r: ", optimal_r)

    for k in range(N):
        print("optimal_h: ", [mino.h[k, i].X for i in range(mino.hidden_dim)])
        print("optimal_r_relu: ", [mino.r_relu[k, i].X for i in range(mino.hidden_dim)])
    """

    w_1_raw = [[-0.1458, 1.8124],
            [0.0463, -1.0703]]
    w_1 = {}
    for i in range(2):
        for j in range(2):
            w_1[i, j] = w_1_raw[i][j]

    b_1 = [-0.4035,  0.6006]

    mino = MINO_DECO_2layer()
    mino.build_SP(X, y, w_1, b_1)
    mino.fit()

    optimal_w_2 = [mino.w_2[i].X for i in range(mino.hidden_dim)]
    print("optimal_w_2: ", optimal_w_2)
    print("optimal_b_2: ", mino.b_2.X)
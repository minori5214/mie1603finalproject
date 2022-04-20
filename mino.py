from gurobipy import *
import numpy as np

class MINO():
    def __init__(self, input_dim=2, output_dim=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def build(self, X, y):
        N = X.shape[0]

        self.model = Model()

        self.z = self.model.addVars(N, vtype=GRB.CONTINUOUS)
        self.y_pred = self.model.addVars(N, lb=0, ub=1, vtype=GRB.CONTINUOUS)

        self.w = self.model.addVars(self.input_dim, vtype=GRB.CONTINUOUS)
        self.b = self.model.addVar(vtype=GRB.CONTINUOUS)

        for k in range(N):
            self.model.addConstr(self.y_pred[k]==quicksum(self.w[i]*X[k, i] for i in range(self.input_dim)) + self.b)

            self.model.addConstr(self.z[k]>=y[k]-self.y_pred[k])
            self.model.addConstr(self.z[k]>=-(y[k]-self.y_pred[k]))

        self.model.update()
        self.model.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model.update()
    
    def fit(self):
        self.model.optimize()

class MINO_sigmoid(MINO):
    def __init__(self, input_dim=2, output_dim=1):
        super(MINO_sigmoid, self).__init__(input_dim, output_dim)

    def build(self, X, y):
        N = X.shape[0]

        self.model = Model()

        self.z = self.model.addVars(N, vtype=GRB.CONTINUOUS)
        self.y_pred = self.model.addVars(N, vtype=GRB.CONTINUOUS)

        self.w = self.model.addVars(self.input_dim, vtype=GRB.CONTINUOUS)
        self.b = self.model.addVar(vtype=GRB.CONTINUOUS)

        self.r = self.model.addVars(N, vtype=GRB.CONTINUOUS)

        self.p = {}
        self.q = {}
        for k in range(N):
            for i in range(3):
                self.p[k, i] = self.model.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                self.q[k, i] = self.model.addVar(vtype=GRB.CONTINUOUS, name="q({%s},{%s})" % (k, i))
            
            self.model.addConstr(quicksum(self.p[k, i] for i in range(3)) == 1)
            self.model.addConstr(quicksum(self.q[k, i] for i in range(3)) == self.y_pred[k])

            self.model.addConstr(self.q[k, 0] <= -self.p[k, 0])
            self.model.addConstr(-self.p[k, 0] <= self.q[k, 1])
            self.model.addConstr(self.q[k, 1] <= self.p[k, 1])
            self.model.addConstr(self.p[k, 1] <= self.q[k, 2])

            self.model.addConstr(self.r[k] >= 3*self.q[k, 1] + self.p[k, 2])

        for k in range(N):
            self.model.addConstr(self.y_pred[k]==quicksum(self.w[i]*X[k, i] for i in range(self.input_dim)) + self.b)

            self.model.addConstr(self.z[k]>=y[k]-self.r[k])
            self.model.addConstr(self.z[k]>=-(y[k]-self.r[k]))


        self.model.update()
        self.model.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model.update()

if __name__ == '__main__':
    X = np.load('toy_X.npy')
    y = np.load('toy_y.npy')
    N = X.shape[0]

    mino = MINO_sigmoid()
    mino.build(X, y)
    mino.fit()

    optimal_y_pred = [mino.y_pred[i].X for i in range(N)]
    optimal_r = [mino.r[i].X for i in range(N)]
    optimal_w = [mino.w[i].X for i in range(mino.input_dim)]
    optimal_b = mino.b.X
    print(optimal_y_pred)
    print(optimal_r)
    print(optimal_w)
    print(optimal_b)
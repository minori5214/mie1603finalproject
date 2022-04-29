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
    
    def fit(self, time_limit=300):
        self.model.setParam('TimeLimit', time_limit)
        self.model.optimize()

class MINO_sigmoid(MINO):
    def __init__(self, input_dim=2, output_dim=1):
        super(MINO_sigmoid, self).__init__(input_dim, output_dim)

        self.slope = 1/4
        self.intercept = 1/2

    def build(self, X, y):
        N = X.shape[0]

        self.model = Model()

        self.z = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.y_pred = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.w = self.model.addVars(self.input_dim, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.b = self.model.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.r = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.d = self.model.addVar(vtype=GRB.BINARY)

        self.p = {}
        self.q = {}
        for k in range(N):
            for i in range(3):
                self.p[k, i] = self.model.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                self.q[k, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
            
            self.model.addConstr(quicksum(self.p[k, i] for i in range(3)) == 1)
            self.model.addConstr(quicksum(self.q[k, i] for i in range(3)) == self.y_pred[k])

            self.model.addConstr(-100*self.p[k, 0] <= self.q[k, 0])
            self.model.addConstr(self.q[k, 0] <= -2*self.p[k, 0])

            self.model.addConstr(-2*self.p[k, 1] <= self.q[k, 1])
            self.model.addConstr(self.q[k, 1] <= 2*self.p[k, 1])

            self.model.addConstr(2*self.p[k, 2] <= self.q[k, 2])
            self.model.addConstr(self.q[k, 2] <= 100*self.p[k, 2])

            self.model.addConstr(self.r[k] == self.slope*self.q[k, 1] + self.intercept*self.p[k, 1] + self.p[k, 2])

        for k in range(N):
            self.model.addConstr(self.y_pred[k]==quicksum(self.w[i]*X[k, i] for i in range(self.input_dim)) + self.b)

            self.model.addConstr(self.z[k]>=y[k]-self.r[k])
            self.model.addConstr(self.z[k]>=-(y[k]-self.r[k]))

            #self.model.addConstr(self.y_pred[k]<= -0.1 + 100*self.d)
            #self.model.addConstr(self.y_pred[k]>= 0.1 - (1-self.d)*100)
        
        #self.model.addConstr(self.w[0]==2.3)
        #self.model.addConstr(self.w[1]==-1.4)
        #self.model.addConstr(self.b==-2.1)
        #self.model.addConstr(self.p[0, 0]==1.0)
        #self.model.addConstr(self.q[0, 0]<=-1.0)


        self.model.update()
        self.model.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model.update()

class MINO_sigmoid_v2(MINO):
    def __init__(self, input_dim=2, output_dim=1):
        super(MINO_sigmoid_v2, self).__init__(input_dim, output_dim)

        self.slope = [1/8, 1/4, 1/8]
        self.intercept = [3/8, 1/2, 5/8]

    def build(self, X, y):
        N = X.shape[0]

        self.model = Model()

        self.z = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.y_pred = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.w = self.model.addVars(self.input_dim, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.b = self.model.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.r = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.d = self.model.addVar(vtype=GRB.BINARY)

        self.p = {}
        self.q = {}
        for k in range(N):
            for i in range(5):
                self.p[k, i] = self.model.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                self.q[k, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
            
            self.model.addConstr(quicksum(self.p[k, i] for i in range(5)) == 1)
            self.model.addConstr(quicksum(self.q[k, i] for i in range(5)) == self.y_pred[k])

            self.model.addConstr(-100*self.p[k, 0] <= self.q[k, 0])
            self.model.addConstr(self.q[k, 0] <= -3*self.p[k, 0])

            self.model.addConstr(-3*self.p[k, 1] <= self.q[k, 1])
            self.model.addConstr(self.q[k, 1] <= -1*self.p[k, 1])

            self.model.addConstr(-1*self.p[k, 2] <= self.q[k, 2])
            self.model.addConstr(self.q[k, 2] <= self.p[k, 2])

            self.model.addConstr(self.p[k, 3] <= self.q[k, 3])
            self.model.addConstr(self.q[k, 3] <= 3*self.p[k, 3])

            self.model.addConstr(3*self.p[k, 4] <= self.q[k, 4])
            self.model.addConstr(self.q[k, 4] <= 100*self.p[k, 4])

            self.model.addConstr(self.r[k] == self.slope[0]*self.q[k, 1] + self.intercept[0]*self.p[k, 1] + \
                                    self.slope[1]*self.q[k, 2] + self.intercept[1]*self.p[k, 2] + \
                                    self.slope[2]*self.q[k, 3] + self.intercept[2]*self.p[k, 3] + \
                                    self.p[k, 4]
                                )

        for k in range(N):
            self.model.addConstr(self.y_pred[k]==quicksum(self.w[i]*X[k, i] for i in range(self.input_dim)) + self.b)

            self.model.addConstr(self.z[k]>=y[k]-self.r[k])
            self.model.addConstr(self.z[k]>=-(y[k]-self.r[k]))

        #self.model.addConstr(self.w[0]==2.3)
        #self.model.addConstr(self.w[1]==-1.4)
        #self.model.addConstr(self.b==-2.1)
        #self.model.addConstr(self.p[5, 2]==1.0)
        #self.model.addConstr(self.q[0, 0]<=-1.0)


        self.model.update()
        self.model.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model.update()

class MINO_2layers_sigmoid_v2(MINO):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=2):
        super(MINO_2layers_sigmoid_v2, self).__init__(input_dim, output_dim)

        self.slope_sm = [1/8, 1/4, 1/8]
        self.intercept_sm = [3/8, 1/2, 5/8]

        self.hidden_dim = hidden_dim

    def build(self, X, y):
        N = X.shape[0]

        self.model = Model()

        self.z = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.y_pred = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.h = {}
        self.r_relu = {}
        for k in range(N):
            for i in range(self.hidden_dim):
                self.h[k, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="h({%s},{%s})" % (k, i))
                self.r_relu[k, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="r_relu({%s},{%s})" % (k, i))

        self.w_1 = {}
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                self.w_1[i, j] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="W_1({%s},{%s})" % (i, j))
        self.b_1 = self.model.addVars(self.hidden_dim, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.w_2 = self.model.addVars(self.hidden_dim, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.b_2 = self.model.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.r = self.model.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.p_relu = {}
        self.q_relu = {}
        for k in range(N):
            for i in range(self.hidden_dim):
                for j in range(2):
                    self.p_relu[k, i, j] = self.model.addVar(vtype=GRB.BINARY, name="p_relu({%s},{%s},{%s})" % (k, i, j))
                    self.q_relu[k, i, j] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q_relu({%s},{%s},{%s})" % (k, i, j))
            
                self.model.addConstr(quicksum(self.p_relu[k, i, j] for j in range(2)) == 1)
                self.model.addConstr(quicksum(self.q_relu[k, i, j] for j in range(2)) == self.h[k, i])

                self.model.addConstr(-100*self.p_relu[k, i, 0] <= self.q_relu[k, i, 0])
                self.model.addConstr(self.q_relu[k, i, 0] <= 0)

                self.model.addConstr(0 <= self.q_relu[k, i, 1])
                self.model.addConstr(self.q_relu[k, i, 1] <= 100*self.p_relu[k, i, 1])

                self.model.addConstr(self.r_relu[k, i] == self.q_relu[k, i, 1])

        self.p = {}
        self.q = {}
        for k in range(N):
            for i in range(5):
                self.p[k, i] = self.model.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                self.q[k, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
            
            self.model.addConstr(quicksum(self.p[k, i] for i in range(5)) == 1)
            self.model.addConstr(quicksum(self.q[k, i] for i in range(5)) == self.y_pred[k])

            self.model.addConstr(-100*self.p[k, 0] <= self.q[k, 0])
            self.model.addConstr(self.q[k, 0] <= -3*self.p[k, 0])

            self.model.addConstr(-3*self.p[k, 1] <= self.q[k, 1])
            self.model.addConstr(self.q[k, 1] <= -1*self.p[k, 1])

            self.model.addConstr(-1*self.p[k, 2] <= self.q[k, 2])
            self.model.addConstr(self.q[k, 2] <= self.p[k, 2])

            self.model.addConstr(self.p[k, 3] <= self.q[k, 3])
            self.model.addConstr(self.q[k, 3] <= 3*self.p[k, 3])

            self.model.addConstr(3*self.p[k, 4] <= self.q[k, 4])
            self.model.addConstr(self.q[k, 4] <= 100*self.p[k, 4])

            self.model.addConstr(self.r[k] == self.slope_sm[0]*self.q[k, 1] + self.intercept_sm[0]*self.p[k, 1] + \
                                    self.slope_sm[1]*self.q[k, 2] + self.intercept_sm[1]*self.p[k, 2] + \
                                    self.slope_sm[2]*self.q[k, 3] + self.intercept_sm[2]*self.p[k, 3] + \
                                    self.p[k, 4]
                                )

        for k in range(N):
            self.model.addConstr(self.y_pred[k]==quicksum(self.w_2[i]*self.h[k, i] for i in range(self.hidden_dim)) + self.b_2)

            self.model.addConstr(self.z[k]>=y[k]-self.r[k])
            self.model.addConstr(self.z[k]>=-(y[k]-self.r[k]))

            for j in range(self.hidden_dim):
                self.model.addConstr(self.h[k, j]==quicksum(self.w_1[i, j]*X[k, i] for i in range(self.input_dim)) + self.b_1[j])

        #self.model.addConstr(self.w[0]==2.3)
        #self.model.addConstr(self.w[1]==-1.4)
        #self.model.addConstr(self.b==-2.1)
        #self.model.addConstr(self.p[5, 2]==1.0)
        #self.model.addConstr(self.q[0, 0]<=-1.0)


        self.model.update()
        self.model.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model.update()

if __name__ == '__main__':
    #X = np.load('toy_X.npy')
    #y = np.load('toy_y.npy')

    X = np.load('titanic_X_train.npy')
    y = np.load('titanic_y_train.npy').reshape(-1)
    N = X.shape[0]

    #mino = MINO_sigmoid_v2()
    mino = MINO_sigmoid_v2(input_dim=X.shape[1])
    mino.build(X, y)
    mino.fit()

    optimal_y_pred = [mino.y_pred[i].X for i in range(N)]
    optimal_r = [mino.r[i].X for i in range(N)]
    optimal_z = [mino.z[i].X for i in range(N)]
    for k in range(N):
        optimal_p = [mino.p[k, i].X for i in range(5)]
        optimal_q = [mino.q[k, i].X for i in range(5)]
        print("p     : ", k, optimal_p)
        print("q     : ", k, optimal_q)

    optimal_w = [mino.w[i].X for i in range(mino.input_dim)]
    optimal_b = mino.b.X
    print("y_pred: ", optimal_y_pred)
    print("r     : ", optimal_r)
    print("z     : ", optimal_z)

    print("w     : ", optimal_w)
    print("b     : ", optimal_b)
    print(mino.model.objVal)
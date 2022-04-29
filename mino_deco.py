from gurobipy import *
import numpy as np

class MINO_DECO_2layer():
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=2, sigmoid='V2'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
            
        self.M = 1000 # maximum raw output before sigmoid
        self.sigmoid = sigmoid
        if sigmoid == 'V1':
            self.slope_sm = 1/4
            self.intercept_sm = 1/2
        elif sigmoid == 'V2':
            self.slope_sm = [1/8, 1/4, 1/8]
            self.intercept_sm = [3/8, 1/2, 5/8]
        else:
            raise NotImplementedError

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
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
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

                self.model_MP.addConstr(-1000*self.p_relu[k, i, 0] <= self.q_relu[k, i, 0])
                self.model_MP.addConstr(self.q_relu[k, i, 0] <= 0)

                self.model_MP.addConstr(0 <= self.q_relu[k, i, 1])
                self.model_MP.addConstr(self.q_relu[k, i, 1] <= 1000*self.p_relu[k, i, 1])

                self.model_MP.addConstr(self.r_relu[k, i] == self.q_relu[k, i, 1])

        if self.sigmoid == 'V1':
            self.p = {}
            self.q = {}
            for k in range(N):
                for i in range(3):
                    self.p[k, i] = self.model_MP.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                    self.q[k, i] = self.model_MP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
                
                self.model_MP.addConstr(quicksum(self.p[k, i] for i in range(3)) == 1)
                self.model_MP.addConstr(quicksum(self.q[k, i] for i in range(3)) == self.y_pred[k])

                self.model_MP.addConstr(-self.M*self.p[k, 0] <= self.q[k, 0])
                self.model_MP.addConstr(self.q[k, 0] <= -2*self.p[k, 0])

                self.model_MP.addConstr(-2*self.p[k, 1] <= self.q[k, 1])
                self.model_MP.addConstr(self.q[k, 1] <= 2*self.p[k, 1])

                self.model_MP.addConstr(2*self.p[k, 2] <= self.q[k, 2])
                self.model_MP.addConstr(self.q[k, 2] <= self.M*self.p[k, 2])

                self.model_MP.addConstr(self.r[k] == self.slope_sm*self.q[k, 1] + self.intercept_sm*self.p[k, 1] + self.p[k, 2])
        elif self.sigmoid == 'V2':
            self.p = {}
            self.q = {}
            for k in range(N):
                for i in range(5):
                    self.p[k, i] = self.model_MP.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                    self.q[k, i] = self.model_MP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
                
                self.model_MP.addConstr(quicksum(self.p[k, i] for i in range(5)) == 1)
                self.model_MP.addConstr(quicksum(self.q[k, i] for i in range(5)) == self.y_pred[k])

                self.model_MP.addConstr(-self.M*self.p[k, 0] <= self.q[k, 0])
                self.model_MP.addConstr(self.q[k, 0] <= -3*self.p[k, 0])

                self.model_MP.addConstr(-3*self.p[k, 1] <= self.q[k, 1])
                self.model_MP.addConstr(self.q[k, 1] <= -1*self.p[k, 1])

                self.model_MP.addConstr(-1*self.p[k, 2] <= self.q[k, 2])
                self.model_MP.addConstr(self.q[k, 2] <= self.p[k, 2])

                self.model_MP.addConstr(self.p[k, 3] <= self.q[k, 3])
                self.model_MP.addConstr(self.q[k, 3] <= 3*self.p[k, 3])

                self.model_MP.addConstr(3*self.p[k, 4] <= self.q[k, 4])
                self.model_MP.addConstr(self.q[k, 4] <= self.M*self.p[k, 4])

                self.model_MP.addConstr(self.r[k] == self.slope_sm[0]*self.q[k, 1] + self.intercept_sm[0]*self.p[k, 1] + \
                                        self.slope_sm[1]*self.q[k, 2] + self.intercept_sm[1]*self.p[k, 2] + \
                                        self.slope_sm[2]*self.q[k, 3] + self.intercept_sm[2]*self.p[k, 3] + \
                                        self.p[k, 4]
                                    )
        else:
            raise NotImplementedError

        for k in range(N):
            self.model_MP.addConstr(self.y_pred[k]==quicksum(w_2[i]*self.r_relu[k, i] for i in range(self.hidden_dim)) + b_2)

            self.model_MP.addConstr(self.z[k]>=y[k]-self.r[k])
            self.model_MP.addConstr(self.z[k]>=-(y[k]-self.r[k]))

            for j in range(self.hidden_dim):
                self.model_MP.addConstr(self.h[k, j]==quicksum(self.w_1[j, i]*X[k, i] for i in range(self.input_dim)) + self.b_1[j])

        self.model_MP.update()
        self.model_MP.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model_MP.update()

    def build_SP(self, X, y, w_1, b_1):
        """
        w_1: [input_dim, hidden_dim]
        b_1: [hidden_dim]
        
        """

        # Calculate h and r_relu
        _w_1 = []
        for i in range(self.hidden_dim):
            _w_1.append([w_1[i, j] for j in range(self.input_dim)])
        _w_1 = np.array(_w_1)
        _b_1 = np.array(b_1)

        h = np.dot(X, np.transpose(_w_1)) + _b_1
        print(h)
        r_relu = np.where(h < 0, 0, h)

        N = X.shape[0]

        self.model_SP = Model()

        self.z = self.model_SP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.y_pred = self.model_SP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.w_2 = self.model_SP.addVars(self.hidden_dim, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)
        self.b_2 = self.model_SP.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        self.r = self.model_SP.addVars(N, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

        if self.sigmoid == 'V1':
            self.p = {}
            self.q = {}
            for k in range(N):
                for i in range(3):
                    self.p[k, i] = self.model_SP.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                    self.q[k, i] = self.model_SP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
                
                self.model_SP.addConstr(quicksum(self.p[k, i] for i in range(3)) == 1)
                self.model_SP.addConstr(quicksum(self.q[k, i] for i in range(3)) == self.y_pred[k])

                self.model_SP.addConstr(-self.M*self.p[k, 0] <= self.q[k, 0])
                self.model_SP.addConstr(self.q[k, 0] <= -2*self.p[k, 0])

                self.model_SP.addConstr(-2*self.p[k, 1] <= self.q[k, 1])
                self.model_SP.addConstr(self.q[k, 1] <= 2*self.p[k, 1])

                self.model_SP.addConstr(2*self.p[k, 2] <= self.q[k, 2])
                self.model_SP.addConstr(self.q[k, 2] <= self.M*self.p[k, 2])

                self.model_SP.addConstr(self.r[k] == self.slope_sm*self.q[k, 1] + self.intercept_sm*self.p[k, 1] + self.p[k, 2])
        elif self.sigmoid == 'V2':
            self.p = {}
            self.q = {}
            for k in range(N):
                for i in range(5):
                    self.p[k, i] = self.model_SP.addVar(vtype=GRB.BINARY, name="p({%s},{%s})" % (k, i))
                    self.q[k, i] = self.model_SP.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name="q({%s},{%s})" % (k, i))
                
                self.model_SP.addConstr(quicksum(self.p[k, i] for i in range(5)) == 1)
                self.model_SP.addConstr(quicksum(self.q[k, i] for i in range(5)) == self.y_pred[k])

                self.model_SP.addConstr(-1000*self.p[k, 0] <= self.q[k, 0])
                self.model_SP.addConstr(self.q[k, 0] <= -3*self.p[k, 0])

                self.model_SP.addConstr(-3*self.p[k, 1] <= self.q[k, 1])
                self.model_SP.addConstr(self.q[k, 1] <= -1*self.p[k, 1])

                self.model_SP.addConstr(-1*self.p[k, 2] <= self.q[k, 2])
                self.model_SP.addConstr(self.q[k, 2] <= self.p[k, 2])

                self.model_SP.addConstr(self.p[k, 3] <= self.q[k, 3])
                self.model_SP.addConstr(self.q[k, 3] <= 3*self.p[k, 3])

                self.model_SP.addConstr(3*self.p[k, 4] <= self.q[k, 4])
                self.model_SP.addConstr(self.q[k, 4] <= 1000*self.p[k, 4])

                self.model_SP.addConstr(self.r[k] == self.slope_sm[0]*self.q[k, 1] + self.intercept_sm[0]*self.p[k, 1] + \
                                        self.slope_sm[1]*self.q[k, 2] + self.intercept_sm[1]*self.p[k, 2] + \
                                        self.slope_sm[2]*self.q[k, 3] + self.intercept_sm[2]*self.p[k, 3] + \
                                        self.p[k, 4]
                                    )
        else:
            raise NotImplementedError

        for k in range(N):
            self.model_SP.addConstr(self.y_pred[k]==quicksum(self.w_2[i]*r_relu[k, i] for i in range(self.hidden_dim)) + self.b_2)

            self.model_SP.addConstr(self.z[k]>=y[k]-self.r[k])
            self.model_SP.addConstr(self.z[k]>=-(y[k]-self.r[k]))


        self.model_SP.update()
        self.model_SP.setObjective(quicksum(self.z), GRB.MINIMIZE)
        self.model_SP.update()

    def optimize(self, problem='MP', time_limit=300):
        if problem == 'MP':
            if time_limit != None:
                self.model_MP.setParam('TimeLimit', time_limit)
            self.model_MP.optimize()

            w_1_opt = {}
            for i in range(self.hidden_dim):
                for j in range(self.input_dim):
                    w_1_opt[i, j] = self.w_1[i, j].X
            b_1_opt = [self.b_1[i].X for i in range(self.hidden_dim)]

            return w_1_opt, b_1_opt, self.model_MP.status

        elif problem == 'SP':
            if time_limit != None:
                self.model_SP.setParam('TimeLimit', time_limit)
            self.model_SP.optimize()

            w_2_opt = [self.w_2[i].X for i in range(self.hidden_dim)]
            b_2_opt = self.b_2.X

            return w_2_opt, b_2_opt, self.model_SP.status
    
    def fit(self, X, y, num_iter=10, time_limit=300, w_1=None, b_1=None):
        prev_loss = np.inf
        if w_1 is None or b_1 is None:
            w_1, b_1 = self.weight_initialize(self.input_dim, self.hidden_dim, method='Xavier')
        #w_1 = np.array([[-0.6342, -0.6831, -0.5635, -0.1115, -0.3090],
        #[ 0.7114, -3.0982,  0.3847, -0.0256,  0.1434],
        #[-0.0673, -0.2047,  0.1489, -0.0539, -0.2021],
        #[ 0.3233, -1.1153,  0.1048,  0.3855,  0.0764],
        #[-0.3925,  2.9765, -0.3265, -0.0108,  0.1546]])
        #b_1 = np.array([-0.2850,  0.3952, -0.3664, -0.0111,  0.1324])

        for e in range(num_iter):
            self.build_SP(X, y, w_1, b_1)
            _w_2, _b_2, status = self.optimize(problem='SP', time_limit=time_limit)
            loss = sum([self.z[i].X for i in range(N)])
            print("w_2: ", _w_2)
            print("b_2: ", _b_2)
            print("Epoch {}: loss: {}".format(e, loss))

            if loss < prev_loss:
                accept = 'accept'
                w_2, b_2 = _w_2, _b_2
                prev_loss = loss
            else:
                accept = 'reject'
            self.save_log(e, loss, 'SP', w_1, b_1, _w_2, _b_2, status, accept=accept)

            self.build_MP(X, y, w_2, b_2)
            _w_1, _b_1, status = self.optimize(problem='MP', time_limit=time_limit)
            loss = sum([self.z[i].X for i in range(N)])
            print("w_1: ", _w_1)
            print("b_1: ", _b_1)
            print("Epoch {}: loss: {}".format(e, loss))

            if loss < prev_loss:
                accept = 'accept'
                w_1, b_1 = _w_1, _b_1
                prev_loss = loss
            else:
                accept = 'reject'
            self.save_log(e, loss, 'MP', _w_1, _b_1, w_2, b_2, status, accept=accept)

            if loss < 0.00001:
                print("Early stop")
                break



    def weight_initialize(self, input_dim, output_dim, method='Xavier', seed=0):
        if method == 'Xavier':
            import math
            #np.random.seed(seed)
            scale = 1/max(1., (input_dim+output_dim)/2.)
            limit = math.sqrt(3.0 * scale)

            w = np.random.uniform(-limit, limit, size=(output_dim, input_dim))
            w_dict = {}
            for i in range(output_dim):
                for j in range(input_dim):
                    w_dict[i, j] = w[i, j]

            b = [0.0]*output_dim
        else:
            raise NotImplementedError
        
        return w_dict, b

    def save_log(self, epoch, loss, problem, w_1, b_1, w_2, b_2, status, accept='accepted', filename='log.txt'):
        _w_1 = []
        for i in range(self.hidden_dim):
            _w_1.append([w_1[i, j] for j in range(self.input_dim)])

        with open(filename, 'a', encoding='UTF-8') as f:
            f.write("Epoch: {}, loss: {}, accept: {}, status: {}, {}\n".format(epoch, loss, accept, status, problem))
            f.write("w_1\n")
            f.write(str(_w_1))
            f.write("\nb_1\n")
            f.write(str(b_1))
            f.write("\nw_2\n")
            f.write(str(w_2))
            f.write("\nb_2\n")
            f.write(str(b_2))
            f.write("\n")

if __name__ == "__main__":
    X = np.load('titanic_X_train.npy')
    y = np.load('titanic_y_train.npy').reshape(-1)
    print(y.shape)
    N = X.shape[0]
    mino = MINO_DECO_2layer(input_dim=5, output_dim=1, hidden_dim=5, sigmoid='V1')
    mino.fit(X, y, num_iter=10, time_limit=600)

    #X = np.load('toy_X.npy')
    #y = np.load('toy_y.npy').reshape(-1)
    #print(y.shape)
    #N = X.shape[0]
    #mino = MINO_DECO_2layer(input_dim=2, output_dim=1, hidden_dim=2, sigmoid='V1')
    #mino.fit(X, y, num_iter=10, time_limit=30)
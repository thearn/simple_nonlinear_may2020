import numpy as np
from numpy import sin, cos, sqrt, exp
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from schedule import Schedule



class Distances(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_v', types=int, default=4)

    def setup(self):
        nn = self.options['num_nodes']
        nr = self.options['num_v']

        self.add_input('t',
               val=np.zeros(nn), units='min')

        # States
        self.add_input('X',
               val=np.zeros((nn, nr)))
        self.add_input('Y',
               val=np.zeros((nn, nr)))

        # schedule based scaling
        self.add_input('c_schedule',
               val=np.ones((nn, nr)))

        # aggregated distance constraint vector?
        self.add_output('dist', val=0.0)


        self.declare_partials('dist', ['X', 'Y', 'c_schedule'])


    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nv = self.options['num_v']

        X = inputs['X']
        Y = inputs['Y']
        c = inputs['c_schedule']
        t = inputs['t']

        dist_defect = 0.0
        dd2 = 0.0
        a = 15.0
        d_min = 12.0
        tol = 1e10
        outputs['dist'] = 0.0

        xe, ye = [], []

        self.dx = np.zeros(X.shape)
        self.dy = np.zeros(Y.shape)
        self.dc = np.zeros(c.shape)

        for i in range(nv):
            for k in range(i + 1, nv):
                x1, y1 = X[:,i], Y[:,i]
                x2, y2 = X[:,k], Y[:,k]

                d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                c1 = c[:, i]
                c2 = c[:, k]
                d_close = np.exp(-a*(-d + d_min * c1 * c2))
                d_close[np.where(d_close > tol)] = tol

                y = 1 / (1 + d_close)

                outputs['dist'] += y.sum()


                dx = -a*(x1 - x2)*d_close/((1 + d_close)**2*d)
                dy = -a*(y1 - y2)*d_close/((1 + d_close)**2*d)

                self.dx[:, i] += dx
                self.dx[:, k] += -dx

                self.dy[:, i] += dy
                self.dy[:, k] += -dy

                self.dc[:, i] += a*c2*d_min*d_close/(1 + d_close)**2
                self.dc[:, k] += a*c1*d_min*d_close/(1 + d_close)**2


                # for j in range(nn):
                #     if y[j] >= 0.9:
                #         dd2 += 1
                #         xe.append(x1[j])
                #         ye.append(y1[j])
                #         xe.append(x2[j])
                #         ye.append(y2[j])

        # print(outputs['dist'], dd2)
        # plt.figure()
        # plt.plot(X, Y)
        # plt.scatter(xe, ye)
        # plt.show()
        # quit()

    def compute_partials(self, params, jacobian):

        jacobian['dist', 'X'] = self.dx
        jacobian['dist', 'Y'] = self.dy
        jacobian['dist', 'c_schedule'] = self.dc

        # jacobian['dist', 'x1'] = -a*(x1 - x2)*np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2)))/((1 + np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2))))**2*sqrt((x1 - x2)**2 + (y1 - y2)**2))
        # jacobian['dist', 'x2'] = -a*(-x1 + x2)*np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2)))/((1 + np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2))))**2*sqrt((x1 - x2)**2 + (y1 - y2)**2))
        # jacobian['dist', 'y1'] = -a*(y1 - y2)*np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2)))/((1 + np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2))))**2*sqrt((x1 - x2)**2 + (y1 - y2)**2))
        # jacobian['dist', 'y2'] = -a*(-y1 + y2)*np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2)))/((1 + np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2))))**2*sqrt((x1 - x2)**2 + (y1 - y2)**2))
        # jacobian['dist', 'c1'] = a*c2*d_min*np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2)))/(1 + np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2))))**2
        # jacobian['dist', 'c2'] = a*c1*d_min*np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2)))/(1 + np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2))))**2
        # jacobian['dist', 'd_min'] = a*c1*c2*np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2)))/(1 + np.exp(-a*(c1*c2*d_min - sqrt((x1 - x2)**2 + (y1 - y2)**2))))**2


if __name__ == '__main__':
    
    np.random.seed(0)

    nn = 30
    nv = 25

    p = om.Problem()
    p.model = om.Group()

    p.model.add_subsystem('inputs_t', om.IndepVarComp('t', val=np.zeros(nn)), promotes=['*'])
    p.model.add_subsystem('inputs_x', om.IndepVarComp('X', val=np.zeros((nn, nv))), promotes=['*'])
    p.model.add_subsystem('inputs_y', om.IndepVarComp('Y', val=np.zeros((nn, nv))), promotes=['*'])

    p.model.add_subsystem('test', Schedule(num_nodes=nn, num_v=nv), promotes=['*'])
    p.model.add_subsystem('distance', Distances(num_nodes=nn, num_v=nv), promotes=['*'])
    p.setup()

    p['t_start'] = np.random.uniform(0, 40, nv)
    p['t_end'] = np.random.uniform(50, 100, nv)
    p['t'] = np.linspace(0, 100, nn)


    theta = np.linspace(0, 2*np.pi, nv + 1)[:nv]

    r = 100.0
    x_start = r * np.cos(theta)
    y_start = r * np.sin(theta)  

    k = 3
    theta2 = theta - np.pi + np.random.uniform(-np.pi/k, np.pi/k, nv)
    x_end = r * np.cos(theta2)
    y_end = r * np.sin(theta2)

    X = np.zeros((nn, nv))
    Y = np.zeros((nn, nv))

    print(x_start)
    for i in range(nv):
        X[:, i] = np.linspace(x_start[i], x_end[i], nn)
        Y[:, i] = np.linspace(y_start[i], y_end[i], nn)

    p['X'] = X
    p['Y'] = Y


    p.run_model()

    c = p['c_schedule']
    #plt.figure()
    #plt.plot(p['t'], p['c_schedule'])


    print(p['dist'])


    #plt.show()


    nn = 10
    nv = 5

    p = om.Problem()
    p.model = om.Group()
    p.model.add_subsystem('distance', Distances(num_nodes=nn, num_v=nv), promotes=['*'])
    p.setup(force_alloc_complex=True)

    np.random.seed(0)

    p['X'] = np.random.uniform(-10, 10, (nn, nv))
    p['Y'] = np.random.uniform(-10, 10, (nn, nv))
    p['c_schedule'] = np.ones((nn, nv))
    p['t'] = np.linspace(0, 100, nn)

    p.run_model()
    check_partials_data = p.check_partials(compact_print=True, method='cs')
    # plot in non-binary mode
    om.partial_deriv_plot('dist', 'X', check_partials_data, binary = False)


    # publish a paper on an application of an MDO technique applied to airspace operations in a gradient based optimization context





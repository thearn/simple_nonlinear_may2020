import numpy as np
from numpy import sin, cos, sqrt, exp, tanh
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import pickle

def make_filter(pts, start, stop, alpha, beta, gamma):
    x,y,t = pts
    startx, starty, startt = start
    gx, gy, gt = gamma
    stopx, stopy, stopt = stop
    ax, ay, at = alpha

    y1 = (0.25 - 0.25 * np.tanh(ax*(-stopx + x)))*(np.tanh(ax*(-startx + x)) + 1)
    x0 = startx + (stopx - startx)/2.0
    y2 = ((x - x0)/gx)**2

    y1 *= (0.25 - 0.25 * np.tanh(ay*(-stopy + y)))*(np.tanh(ay*(-starty + y)) + 1)
    y0 = starty + (stopy - starty)/2.0
    y2 += ((y - y0)/gy)**2

    y1 *= (0.25 - 0.25 * np.tanh(at*(-stopt + t)))*(np.tanh(at*(-startt + t)) + 1)
    t0 = startt + (stopt - startt)/2.0
    y2 += ((t - t0)/gt)**2


    # --------------------

    y2 = 1.0 / (1 + y2)
    y = beta * y1 + (1 - beta) * y2

    dfdx = ax*beta*(0.25 - 0.25*tanh(at*(-stopt + t)))*(0.25 - 0.25*tanh(ax*(-stopx + x)))*(0.25 - 0.25*tanh(ay*(-stopy + y)))*(1 - tanh(ax*(-startx + x))**2)*(tanh(at*(-startt + t)) + 1)*(tanh(ay*(-starty + y)) + 1) - 0.25*ax*beta*(0.25 - 0.25*tanh(at*(-stopt + t)))*(0.25 - 0.25*tanh(ay*(-stopy + y)))*(1 - tanh(ax*(-stopx + x))**2)*(tanh(at*(-startt + t)) + 1)*(tanh(ax*(-startx + x)) + 1)*(tanh(ay*(-starty + y)) + 1) - 1.0*(1 - beta)*(-1.0*startx - 1.0*stopx + 2*x)/(gx**2*(1 + (-0.5*starty - 0.5*stopy + y)**2/gy**2 + (-0.5*startx - 0.5*stopx + x)**2/gx**2 + (-0.5*startt - 0.5*stopt + t)**2/gt**2)**2)

    dfdy = ay*beta*(0.25 - 0.25*tanh(at*(-stopt + t)))*(0.25 - 0.25*tanh(ax*(-stopx + x)))*(0.25 - 0.25*tanh(ay*(-stopy + y)))*(1 - tanh(ay*(-starty + y))**2)*(tanh(at*(-startt + t)) + 1)*(tanh(ax*(-startx + x)) + 1) - 0.25*ay*beta*(0.25 - 0.25*tanh(at*(-stopt + t)))*(0.25 - 0.25*tanh(ax*(-stopx + x)))*(1 - tanh(ay*(-stopy + y))**2)*(tanh(at*(-startt + t)) + 1)*(tanh(ax*(-startx + x)) + 1)*(tanh(ay*(-starty + y)) + 1) - 1.0*(1 - beta)*(-1.0*starty - 1.0*stopy + 2*y)/(gy**2*(1 + (-0.5*starty - 0.5*stopy + y)**2/gy**2 + (-0.5*startx - 0.5*stopx + x)**2/gx**2 + (-0.5*startt - 0.5*stopt + t)**2/gt**2)**2)

    dfdt = at*beta*(0.25 - 0.25*tanh(at*(-stopt + t)))*(0.25 - 0.25*tanh(ax*(-stopx + x)))*(0.25 - 0.25*tanh(ay*(-stopy + y)))*(1 - tanh(at*(-startt + t))**2)*(tanh(ax*(-startx + x)) + 1)*(tanh(ay*(-starty + y)) + 1) - 0.25*at*beta*(0.25 - 0.25*tanh(ax*(-stopx + x)))*(0.25 - 0.25*tanh(ay*(-stopy + y)))*(1 - tanh(at*(-stopt + t))**2)*(tanh(at*(-startt + t)) + 1)*(tanh(ax*(-startx + x)) + 1)*(tanh(ay*(-starty + y)) + 1) - 1.0*(1 - beta)*(-1.0*startt - 1.0*stopt + 2*t)/(gt**2*(1 + (-0.5*starty - 0.5*stopy + y)**2/gy**2 + (-0.5*startx - 0.5*stopx + x)**2/gx**2 + (-0.5*startt - 0.5*stopt + t)**2/gt**2)**2)

    return y, [dfdx, dfdy, dfdt]



class Density(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_v', types=int, default=4)
        self.options.declare('beta', types=float, default=0.995)

        self.mx = 45
        self.mt = 45

    def setup(self):
        nn = self.options['num_nodes']
        nr = self.options['num_v']
        beta = self.options['beta']

        self.add_input('t',
               val=np.zeros(nn), units='min')

        # States
        self.add_input('X',
               val=np.zeros((nn, nr)), units='m')
        self.add_input('Y',
               val=np.zeros((nn, nr)), units='m')

        # schedule based scaling
        self.add_input('c_schedule',
               val=np.zeros((nn, nr)))

        # aggregated distance constraint vector?
        self.add_output('density', val=0.0)


        self.declare_partials('density', ['X', 'Y', 'c_schedule'])

        lat_sep = 2.75 / 2.0
        temp_sep = 1. / 2.0

        x, y, t = np.meshgrid(np.linspace(-nn, nn, self.mx), np.linspace(-nn, nn, self.mx), np.linspace(-nn, nn, self.mt))

        self.c, self.df = make_filter([x, y, t], 
                            [-lat_sep, -lat_sep, -temp_sep], 
                            [lat_sep, lat_sep, temp_sep], 
                            [20.0,20.0,20.0], 
                            beta, 
                            3*[2.0]) 

        self.space = np.linspace(-100, 100, self.mx)
        self.tm = np.linspace(0, 10, self.mt)


        Z = np.max(self.df[2], axis=-1)
        plt.figure()
        plt.imshow(Z)
        plt.colorbar()


    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nr = self.options['num_v']
        beta = self.options['beta']

        X = inputs['X']
        Y = inputs['Y']
        t = inputs['t']

        Z = np.zeros((self.mx, self.mx, self.mt))
        for i in range(nr):
            x = X[:, i]
            y = Y[:, i]

            C = np.zeros((self.mx, self.mx, self.mt))
            k1 = np.digitize(x, self.space)
            k2 = np.digitize(y, self.space)
            k3 = np.digitize(t, self.tm)

            for k in range(1, nn - 1):

                A = np.roll(self.c, (k1[k] - self.mx//2, k2[k] - self.mx//2, k3[k] - self.mt//2), axis=(1,0,2))
                C = C + A - C * A

            Z = Z + C

        Z = np.max(Z, axis=-1)
        plt.figure()
        plt.imshow(Z)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    from schedule import Schedule
    np.random.seed(0)
    p = om.Problem()
    p.model = om.Group()
    nv = 50
    n = 30

    p.model.add_subsystem('schedule', Schedule(num_nodes=n, num_v=nv), promotes=['*'])
    p.model.add_subsystem('test', Density(num_nodes=n, num_v=nv), promotes=['*'])
    p.model.add_subsystem('inputs', om.IndepVarComp('t', np.zeros(n), units='min'), promotes=['*'])
    p.setup(force_alloc_complex=True)

    p['t_start'] = np.random.uniform(0, 4, nv)
    p['t_end'] = np.random.uniform(5, 10, nv)
    p['t'] = np.linspace(0, 10, n)

    p['X'] = np.random.uniform(-100, 100, (n, nv))
    p['Y'] = np.random.uniform(-100, 100, (n, nv))

    p.run_model()

    #check_partials_data = p.check_partials(compact_print=True, method='cs')







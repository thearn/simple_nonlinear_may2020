import numpy as np
from numpy import sin, cos, sqrt, exp
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt


class DeMux(om.ExplicitComponent):
    """ Trajectory vector ODE demuxing component
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('nv', types=int)


    def setup(self):
        nn = self.options['num_nodes']
        nv = self.options['nv']

        self.add_input('X',
               val=np.zeros((nn, nv)), units='m')
        self.add_input('Y',
               val=np.zeros((nn, nv)), units='m')


        for i in range(nv):
            self.add_output('x_%i' % i, val=np.zeros(nn))
            self.add_output('y_%i' % i, val=np.zeros(nn))

        arange0 = np.arange(nn, dtype=int)
        arange1 = np.arange(nn * nv, dtype=int)
        arange2 = []
        for i in range(nn):
            arange2 += nv * [i]

        for i in range(nv):
            self.declare_partials('x_%i' % i, 'X', rows=arange2, cols=arange1)
            self.declare_partials('y_%i' % i, 'Y', rows=arange2, cols=arange1)

    def compute(self, inputs, outputs):
        nv = self.options['nv']
        X = inputs['X']
        Y = inputs['Y']

        for i in range(nv):
            outputs['x_%i' % i] = X[:, i]
            outputs['y_%i' % i] = Y[:, i] 

    def compute_partials(self, params, jacobian):
        nn = self.options['num_nodes']
        nv = self.options['nv']
        for i in range(nv):
            deriv = np.zeros((nn, nv))
            deriv[:,i] = np.ones(nn)
            jacobian['x_%i' % i, 'X'] = deriv.flatten()
            jacobian['y_%i' % i, 'Y'] = deriv.flatten()




if __name__ == '__main__':
    
    np.random.seed(0)

    nn = 10
    nv = 7

    p = om.Problem()
    p.model = om.Group()

    p.model.add_subsystem('distance', DeMux(num_nodes=nn, nv=nv), 
                          promotes=['*'])
    p.setup()

    p['X'] = np.random.uniform(-50, 50, (nn, nv))
    p['Y'] = np.random.uniform(-50, 50, (nn, nv))
    p.run_model()

    check_partials_data = p.check_partials(compact_print=True)

    om.partial_deriv_plot('x_0', 'X', check_partials_data, binary = False)



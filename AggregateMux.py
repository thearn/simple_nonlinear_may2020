import numpy as np
from numpy import sin, cos, sqrt, exp
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt


class AggregateMux(om.ExplicitComponent):
    """ Trajectory vector ODE AggregateMuxing component
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('nc', types=int)


    def setup(self):
        nn = self.options['num_nodes']
        nc = self.options['nc']

        for i in range(nc):
            self.add_input('dist_%i' % i, val=np.zeros(nn))

        # aggregated distance constraint vector
        self.add_output('dist', val=np.zeros(nn))
        self.add_output('dist_good', val=np.zeros(nn))

        arange0 = np.arange(nn, dtype=int)
        arange1 = np.arange(nn * nc, dtype=int)
        arange2 = []
        for i in range(nn):
            arange2 += nc * [i]

        for i in range(nc):
            self.declare_partials('dist', 'dist_%i' % i, rows=arange0, cols=arange0)
            self.declare_partials('dist_good', 'dist_%i' % i, rows=arange0, cols=arange0)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nc = self.options['nc']
        
        dist = np.zeros(nn)
        dist_good = np.zeros(nn)

        self.jac = {}
        self.jac_good = {}
        for i in range(nc):
            name  = 'dist_%i' % i
            x = inputs[name]
            dx = np.zeros(nn)
            dx_good = np.zeros(nn)

            idx = np.where(x > 0)
            idx_good = np.where(x <= 0)

            dist[idx] += x[idx]
            dist_good[idx_good] += x[idx_good]

            dx[idx] = 1.0
            dx_good[idx_good] = 1.0
            self.jac['dist', name] = dx
            self.jac_good['dist_good', name] = dx_good

        #print(dist.max())
        outputs['dist'] = dist
        outputs['dist_good'] = dist_good

    def compute_partials(self, params, jacobian):
        nn = self.options['num_nodes']
        nc = self.options['nc']

        for pair in self.jac:
            jacobian[pair] = self.jac[pair]

        for pair in self.jac_good:
            jacobian[pair] = self.jac_good[pair]


if __name__ == '__main__':
    
    np.random.seed(0)

    nn = 10
    nc = 7

    p = om.Problem()
    p.model = om.Group()

    p.model.add_subsystem('mux', AggregateMux(num_nodes=nn, nc=nc), 
                          promotes=['*'])
    p.setup()

    for i in range(nc):
        p['dist_%i' % i] = np.random.uniform(-50, 50, nn)
    p.run_model()

    check_partials_data = p.check_partials(compact_print=True)

    om.partial_deriv_plot('dist', 'dist_0', check_partials_data, binary = False)



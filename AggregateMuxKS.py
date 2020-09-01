import numpy as np
from numpy import sin, cos, sqrt, exp
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt


class AggregateMuxKS(om.ExplicitComponent):
    """ Trajectory vector ODE AggregateMuxKSing component
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
        self.add_output('distks', val=np.zeros((nn, nc)))

        arange0 = np.arange(nn, dtype=int)
        arange1 = np.arange(nn * nc, dtype=int)
        arange2 = []
        for i in range(nn):
            arange2 += nc * [i]

        for i in range(nc):
            self.declare_partials('distks', 'dist_%i' % i, rows=arange1, cols=arange2)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nc = self.options['nc']
        
        dist = np.zeros((nn, nc))

        self.jac = {}
        for i in range(nc):
            name = 'dist_%i' % i
            outputs['distks'][:, i] = inputs[name]
            
            dx = np.zeros((nn, nc))
            dx[:, i] = 1.0
            self.jac['distks', name] = dx


    def compute_partials(self, params, jacobian):
        nn = self.options['num_nodes']
        nc = self.options['nc']

        for name in self.jac:
            jacobian[name] = self.jac[name].flatten()


if __name__ == '__main__':
    
    np.random.seed(0)

    nn = 10
    nc = 7

    p = om.Problem()
    p.model = om.Group()

    p.model.add_subsystem('mux', AggregateMuxKS(num_nodes=nn, nc=nc), 
                          promotes=['*'])
    p.setup()

    for i in range(nc):
        p['dist_%i' % i] = np.random.uniform(-50, 50, nn)
    p.run_model()

    check_partials_data = p.check_partials(compact_print=True)

    om.partial_deriv_plot('distks', 'dist_0', check_partials_data, binary = False)



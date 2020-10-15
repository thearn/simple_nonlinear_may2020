import numpy as np
from numpy import sin, cos, sqrt, exp
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from schedule import Schedule


"distance from set of dynamic states to a single X,Y trajectory"


class SingleDistance(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_v', types=int, default=4)

    def setup(self):
        nn = self.options['num_nodes']
        nv = self.options['num_v']

        # States
        self.add_input('X',
               val=np.zeros((nn, nv)))
        self.add_input('Y',
               val=np.zeros((nn, nv)))

        self.add_input('fixed_x',
               val=np.zeros(nn))
        self.add_input('fixed_y',
               val=np.zeros(nn))

        self.add_output('dist_to_fixed',
               val=np.zeros((nn, nv)))

        arange1 = np.arange(nn * nv, dtype=int)
        arange2 = []
        for i in range(nn):
            arange2 += nv * [i]
        arange3 = []
        for i in range(nn):
            arange3.extend(range(nv))

        self.declare_partials('dist_to_fixed', 'X', rows=arange1, cols=arange1)
        self.declare_partials('dist_to_fixed', 'Y', rows=arange1, cols=arange1)

        self.declare_partials('dist_to_fixed', 'fixed_x', rows=arange1, cols=arange2)
        self.declare_partials('dist_to_fixed', 'fixed_y', rows=arange1, cols=arange2)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nv = self.options['num_v']

        xx = inputs['fixed_x']
        yy = inputs['fixed_y']


        self.dx = np.apply_along_axis(lambda _x: _x - xx, 0, inputs['X'])
        self.dy = np.apply_along_axis(lambda _y: _y - yy, 0, inputs['Y'])
        self.dist = np.sqrt(self.dx**2 + self.dy**2)

        outputs['dist_to_fixed'] = self.dist


    def compute_partials(self, params, jacobian):
        jacobian['dist_to_fixed', 'X'] = (self.dx / self.dist).flatten()
        jacobian['dist_to_fixed', 'Y'] = (self.dy / self.dist).flatten()

        jacobian['dist_to_fixed', 'fixed_x'] = (-self.dx / self.dist).flatten()
        jacobian['dist_to_fixed', 'fixed_y'] = (-self.dy / self.dist).flatten()


if __name__ == '__main__':
    
    np.random.seed(0)

    nn = 10
    nv = 5

    p = om.Problem()
    p.model = om.Group()
    p.model.add_subsystem('distance', SingleDistance(num_nodes=nn, num_v=nv), promotes=['*'])

    np.random.seed(0)

    p.setup()

    p['X'] = np.random.uniform(-10, 10, (nn, nv))
    p['Y'] = np.random.uniform(-10, 10, (nn, nv))

    p['fixed_x'] = np.random.uniform(-10, 10, nn)
    p['fixed_y'] = np.random.uniform(-10, 10, nn)

    p.run_model()
    check_partials_data = p.check_partials(compact_print=True)
    
    # plot in non-binary mode
    om.partial_deriv_plot('dist_to_fixed', 'fixed_x', check_partials_data, binary = False)




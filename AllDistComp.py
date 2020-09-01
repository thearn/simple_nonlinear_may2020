import numpy as np
from numpy import sin, cos, sqrt, exp
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt


class AllDistComp(om.ExplicitComponent):
    """ For computing complete pairwise distance constraints.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('limit', types=float, default=1.0)


    def setup(self):
        nn = self.options['num_nodes']

        # States
        self.add_input('x1',
               val=np.zeros(nn))
        self.add_input('y1',
               val=np.zeros(nn))

        self.add_input('x2',
               val=np.zeros(nn))
        self.add_input('y2',
               val=np.zeros(nn))

        self.add_output('dist', val=np.zeros(nn))

        arange1 = np.arange(nn, dtype=int)

        self.declare_partials('dist', ['x1', 'y1', 'x2', 'y2'], rows=arange1, cols=arange1)

    def compute(self, inputs, outputs):
        # compute distance for each pair of trajectories

        nn = self.options['num_nodes']
        limit = self.options['limit']

        x1 = inputs['x1']
        y1 = inputs['y1']

        x2 = inputs['x2']
        y2 = inputs['y2']

        diff_x = x1 - x2
        diff_y = y1 - y2
        dist = np.sqrt(diff_x**2 + diff_y**2)

        dist[np.where(dist < 1e-10)] = 1e-10

        self.dx = diff_x/dist/limit
        self.dy = diff_y/dist/limit

        outputs['dist'] = (limit - dist)/limit

    def compute_partials(self, params, jacobian):

        jacobian['dist', 'x1'] = -self.dx
        jacobian['dist', 'x2'] = self.dx
        jacobian['dist', 'y1'] = -self.dy 
        jacobian['dist', 'y2'] = self.dy



if __name__ == '__main__':
    
    np.random.seed(0)

    nn = 7

    p = om.Problem()
    p.model = om.Group()

    p.model.add_subsystem('distance', AllDistComp(num_nodes=nn, 
                                                   limit=50.0), 
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    p['x1'] = np.random.uniform(-50, 50, nn)
    p['y1'] = np.random.uniform(-50, 50, nn)
    p['x2'] = np.random.uniform(-50, 50, nn)
    p['y2'] = np.random.uniform(-50, 50, nn)
    p.run_model()

    check_partials_data = p.check_partials(compact_print=True, method='cs')
    print()
    print(p['dist'])

    om.partial_deriv_plot('dist', 'x1', check_partials_data, binary = False)



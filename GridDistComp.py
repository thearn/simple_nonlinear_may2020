import numpy as np
from numpy import sin, cos, sqrt, exp
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from recursive_grid import min_dist_grid, min_dist_brute


class GridDistComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_v', types=int, default=4)
        self.options.declare('limit', types=float, default=1.0)
        self.options.declare('method', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        nv = self.options['num_v']

        self.min_dist_method = min_dist_grid
        if self.options['method'] == 0:
            self.min_dist_method = min_dist_brute

        # States
        self.add_input('X',
               val=np.zeros((nn, nv)))
        self.add_input('Y',
               val=np.zeros((nn, nv)))

        # aggregated distance constraint vector
        self.add_output('dist', val=np.zeros(nn))
        self.add_output('dist_good', val=np.zeros(nn))

        arange1 = np.arange(nn * nv, dtype=int)
        arange2 = []
        for i in range(nn):
            arange2 += nv * [i]

        self.declare_partials('dist', ['X', 'Y'], rows=arange2, cols=arange1)
        self.declare_partials('dist_good', ['X', 'Y'], rows=arange2, cols=arange1)


    def compute(self, inputs, outputs):
        """
        ZERO OUT OUTPUTS@!!
        """
        nn = self.options['num_nodes']
        nv = self.options['num_v']
        limit = self.options['limit']

        X = inputs['X']
        Y = inputs['Y']

        self.dx = 0*X.copy()
        self.dy = 0*Y.copy()

        self.dx_good = 0*X.copy()
        self.dy_good = 0*Y.copy()

        outputs['dist'] = np.zeros(nn)
        outputs['dist_good'] = np.zeros(nn)

        gap = 0#int(0.1*nn) + 1
        for i in range(gap, nn - gap):
            x_sub = X[i]
            y_sub = Y[i]

            min_d, pts, pts_bad = self.min_dist_method(x_sub, y_sub, limit=limit)

            for a, b in pts_bad:
                d = pts_bad[a, b][0]
                gap = (limit - d)/limit
                if gap > 0.0:
                    outputs['dist'][i] += gap
                    self.dx[i, a] += -pts_bad[a, b][1]/limit
                    self.dx[i, b] += pts_bad[a, b][1]/limit
                    self.dy[i, a] += -pts_bad[a, b][2]/limit
                    self.dy[i, b] += pts_bad[a, b][2]/limit
                else:
                    outputs['dist_good'][i] += gap
                    self.dx_good[i, a] += -pts_bad[a, b][1]/limit
                    self.dx_good[i, b] += pts_bad[a, b][1]/limit
                    self.dy_good[i, a] += -pts_bad[a, b][2]/limit
                    self.dy_good[i, b] += pts_bad[a, b][2]/limit

        # print()
        # print("#",outputs['dist'])
        # print("#",outputs['dist_good'])

    def compute_partials(self, params, jacobian):

        jacobian['dist', 'X'] = self.dx.flatten()
        jacobian['dist', 'Y'] = self.dy.flatten()

        jacobian['dist_good', 'X'] = self.dx_good.flatten()
        jacobian['dist_good', 'Y'] = self.dy_good.flatten()

if __name__ == '__main__':
    
    np.random.seed(0)

    nn = 3
    nv = 10

    p = om.Problem()
    p.model = om.Group()

    p.model.add_subsystem('distance', GridDistComp(num_nodes=nn, 
                                                   num_v=nv,
                                                   limit=20.0), 
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    p['X'] = np.random.uniform(-50, 50, (nn, nv))
    p['Y'] = np.random.uniform(-50, 50, (nn, nv))

    p.run_model()

    check_partials_data = p.check_partials(compact_print=True, method='cs')

    print(p['dist'])

    om.partial_deriv_plot('dist', 'Y', check_partials_data, binary = False)



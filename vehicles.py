import numpy as np
from numpy import sin, cos
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import time

class Vehicles(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_v', types=int, default=4)

    def setup(self):
        nn = self.options['num_nodes']
        nr = self.options['num_v']

        self.add_input('c_schedule',
               val=np.ones((nn, nr)), units=None)

        # States
        self.add_input('X',
               val=np.zeros((nn, nr)), units='m')
        self.add_input('Y',
               val=np.zeros((nn, nr)), units='m')

        self.add_input('theta',
               val=np.zeros((nn, nr)), units='rad')

        # controls
        self.add_input('Vx',
                       val=np.zeros((nn, nr)), units='m/s')

        self.add_input('Vy',
                       val=np.zeros((nn, nr)), units='m/s')


        # SOC
        self.add_output('X_dot',
               val=np.zeros((nn, nr)), units='m/min')
        self.add_output('Y_dot',
               val=np.zeros((nn, nr)), units='m/min')


        # impulse
        self.add_output('sq_thrust',
               val=np.zeros(nn))


        arange1 = np.arange(nn * nr, dtype=int)
        arange2 = []
        for i in range(nn):
            arange2 += nr * [i]

        self.declare_partials('X_dot', ['Vx'], rows=arange1, cols=arange1)
        self.declare_partials('Y_dot', ['Vy'], rows=arange1, cols=arange1)

        self.declare_partials('sq_thrust', ['Vx', 'Vy'], rows=arange2, cols=arange1)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        X = inputs['X']
        Y = inputs['Y']
        Vx = inputs['Vx']
        Vy = inputs['Vy']

        outputs['X_dot'] = Vx
        outputs['Y_dot'] = Vy

        sq_thrust = np.sum(Vx**2 + Vy**2, axis=1)

        outputs['sq_thrust'] = sq_thrust

        #print(sq_thrust.sum())



    def compute_partials(self, inputs, jacobian):
        nn = self.options['num_nodes']
        X = inputs['X']
        Y = inputs['Y']
        Vx = inputs['Vx']
        Vy = inputs['Vy']

        jacobian['X_dot', 'Vx'] = np.ones(X.shape).flatten()
        #jacobian['X_dot', 'theta'] = (-V * sin(theta)).flatten()

        jacobian['Y_dot', 'Vy'] = np.ones(Y.shape).flatten()
        #jacobian['Y_dot', 'theta'] = (V*cos(theta)).flatten()

        jacobian['sq_thrust', 'Vx'] = (2*Vx).flatten()
        jacobian['sq_thrust', 'Vy'] = (2*Vy).flatten()

if __name__ == '__main__':
    from schedule import Schedule
    np.random.seed(0)
    p = om.Problem()
    p.model = om.Group()
    nv = 4
    n = 30

    p.model.add_subsystem('vehicles', Vehicles(num_nodes=n, num_v=nv), promotes=['*'])
    p.setup(force_alloc_complex=True)

    p['X'] = np.random.uniform(-1000, 1000, (n, nv))
    p['Y'] = np.random.uniform(-1000, 1000, (n, nv))
    #p['theta'] = np.random.uniform(-1000, 1000, (n, nv))
    p['Vx'] = np.random.uniform(-10, 10, (n, nv))
    p['Vy'] = np.random.uniform(-10, 10, (n, nv))
    #p['theta_dot'] = np.random.uniform(-1000, 1000, (n, nv))

    p.run_model()
    check_partials_data = p.check_partials(compact_print=True, method='cs')

    # plot in non-binary mode
    #om.partial_deriv_plot('Vy_dot', 't', check_partials_data, binary = False)
    #quit()

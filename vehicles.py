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

        # self.add_input('t',
        #        val=np.zeros(nn))

        self.add_input('c_schedule',
               val=np.zeros((nn, nr)), units=None)

        # States
        self.add_input('X',
               val=np.zeros((nn, nr)), units='m')
        self.add_input('Y',
               val=np.zeros((nn, nr)), units='m')

        self.add_input('Vx',
               val=np.zeros((nn, nr)), units='m/min')
        self.add_input('Vy',
               val=np.zeros((nn, nr)), units='m/min')

        self.add_input('theta',
               val=np.zeros((nn, nr)), units='rad')

        # controls
        self.add_input('theta_dot',
               val=np.zeros((nn, nr)), units='rad/min')

        self.add_input('thrust',
               val=np.zeros((nn, nr)), units='N')

        # SOC
        self.add_output('X_dot',
               val=np.zeros((nn, nr)), units='m/min')
        self.add_output('Y_dot',
               val=np.zeros((nn, nr)), units='m/min')
        self.add_output('Vx_dot',
               val=np.zeros((nn, nr)), units='m/min**2')
        self.add_output('Vy_dot',
               val=np.zeros((nn, nr)), units='m/min**2')

        # impulse
        self.add_output('sq_thrust',
               val=np.zeros(nn), units='N**2')


        arange1 = np.arange(nn * nr, dtype=int)
        arange2 = []
        for i in range(nn):
            arange2 += nr * [i]

        self.declare_partials('X_dot', ['Vx', 'theta'], rows=arange1, cols=arange1)
        self.declare_partials('Y_dot', ['Vy', 'theta'], rows=arange1, cols=arange1)
        
        self.declare_partials('Vx_dot', ['theta', 'thrust', 'c_schedule', 'Vx'], rows=arange1, cols=arange1)
        #self.declare_partials('Vx_dot', ['t'], rows=arange1, cols=arange2)
        
        self.declare_partials('Vy_dot', ['theta', 'thrust', 'c_schedule', 'Vy'], rows=arange1, cols=arange1)
        #self.declare_partials('Vy_dot', ['t'], rows=arange1, cols=arange2)

        self.declare_partials('sq_thrust', 'thrust', rows=arange2, cols=arange1)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        X = inputs['X']
        Y = inputs['Y']
        Vx = inputs['Vx']
        Vy = inputs['Vy']
        theta = inputs['theta']
        thrust = inputs['thrust']
        #t = inputs['t']
        c_schedule = inputs['c_schedule']

        self.c_thrust = (thrust*c_schedule)

        outputs['X_dot'] = cos(theta) * Vx
        outputs['Y_dot'] = sin(theta) * Vy

        self.a = 0.3
        outputs['Vx_dot'] = cos(theta) * (self.c_thrust - self.a*Vx)
        outputs['Vy_dot'] = sin(theta) * (self.c_thrust - self.a*Vy)

        sq_thrust = np.sum(thrust**2, axis=1)

        outputs['sq_thrust'] = sq_thrust



    def compute_partials(self, inputs, jacobian):
        nn = self.options['num_nodes']
        X = inputs['X']
        Y = inputs['Y']
        Vx = inputs['Vx']
        Vy = inputs['Vy']
        theta = inputs['theta']
        thrust = inputs['thrust']
        c_schedule = inputs['c_schedule']

        jacobian['X_dot', 'Vx'] = (cos(theta)).flatten()
        jacobian['X_dot', 'theta'] = (-Vx*sin(theta)).flatten()

        jacobian['Y_dot', 'Vy'] = (sin(theta)).flatten()
        jacobian['Y_dot', 'theta'] = (Vy*cos(theta)).flatten()

        jacobian['Vx_dot', 'theta'] = (-(-Vx*self.a + c_schedule*thrust)*sin(theta)).flatten()
        jacobian['Vx_dot', 'Vx'] = (-self.a*cos(theta)).flatten()
        #jacobian['Vx_dot', 'a'] = (-Vx*cos(theta)).flatten()
        jacobian['Vx_dot', 'thrust'] = (c_schedule*cos(theta)).flatten()
        jacobian['Vx_dot', 'c_schedule'] = (thrust*cos(theta)).flatten()

        jacobian['Vy_dot', 'theta'] = ((-Vy*self.a + c_schedule*thrust)*cos(theta)).flatten()
        jacobian['Vy_dot', 'Vy'] = (-self.a*sin(theta)).flatten()
        #jacobian['Vy_dot', 'a'] = (-Vy*sin(theta)).flatten()
        jacobian['Vy_dot', 'thrust'] = (c_schedule*sin(theta)).flatten()
        jacobian['Vy_dot', 'c_schedule'] = (thrust*sin(theta)).flatten()

        jacobian['sq_thrust', 'thrust'] = (2*thrust).flatten()

if __name__ == '__main__':
    from schedule import Schedule
    np.random.seed(0)
    p = om.Problem()
    p.model = om.Group()
    nv = 4
    n = 30

    p.model.add_subsystem('schedule', Schedule(num_nodes=n, num_v=nv), promotes=['*'])
    p.model.add_subsystem('vehicles', Vehicles(num_nodes=n, num_v=nv), promotes=['*'])
    p.setup(force_alloc_complex=True)

    p['t_start'] = np.random.uniform(0, 4, nv)
    p['t_end'] = np.random.uniform(5, 10, nv)
    p['t'] = np.linspace(0,10,n)

    p['X'] = np.random.uniform(-1000, 1000, (n, nv))
    p['Y'] = np.random.uniform(-1000, 1000, (n, nv))
    p['Vx'] = np.random.uniform(-1000, 1000, (n, nv))
    p['Vy'] = np.random.uniform(-1000, 1000, (n, nv))
    p['theta'] = np.random.uniform(-1000, 1000, (n, nv))
    p['thrust'] = np.random.uniform(-10, 10, (n, nv))
    p['theta_dot'] = np.random.uniform(-1000, 1000, (n, nv))

    p.run_model()
    check_partials_data = p.check_partials(compact_print=True, method='cs')

    # plot in non-binary mode
    #om.partial_deriv_plot('Vy_dot', 't', check_partials_data, binary = False)
    #quit()

import numpy as np 
import openmdao.api as om
import dymos as dm

from schedule import Schedule
from vehicles import Vehicles

import matplotlib.pyplot as plt



class Airspace(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_vehicles', types=int, default=4)

    def setup(self):
        nn = self.options['num_nodes']
        nv = self.options['num_vehicles']

        self.add_subsystem('schedule', Schedule(num_nodes=nn, num_v=nv), promotes=['*'])
        self.add_subsystem('vehicles', Vehicles(num_nodes=nn, num_v=nv), promotes=['*'])

nv = 5
ns = 30
t_duration = 200.0


p = om.Problem(model=om.Group())
traj = dm.Trajectory()

p.model.add_subsystem('traj', subsys=traj)
p.model.linear_solver = om.DirectSolver()

gl = dm.GaussLobatto(num_segments=ns, order=3)
nn = gl.grid_data.num_nodes

phase = dm.Phase(ode_class=Airspace,
                 ode_init_kwargs={'num_vehicles' : nv},
                 transcription=gl)

traj.add_phase(name='phase0', phase=phase)


phase.set_time_options(fix_initial=True, fix_duration=True, targets=['t'], units='min')

traj.add_phase(name='phase0', phase=phase)


ds = 1e-2
phase.add_state('X', 
                fix_initial=True, 
                shape=(nv,),
                rate_source='X_dot', 
                targets='X',
                units='m', 
                lower=-1000.0,
                upper=1000.0, 
                defect_scaler=ds)

phase.add_state('Y', 
                fix_initial=True, 
                shape=(nv,),
                rate_source='Y_dot', 
                targets='Y',
                units='m', 
                lower=-1000.0,
                upper=1000.0, 
                defect_scaler=ds)

phase.add_state('Vx', 
                fix_initial=True, 
                shape=(nv,),
                rate_source='Vx_dot', 
                targets='Vx', 
                lower=-10.0,
                upper=10.0, 
                units='m/s',
                defect_scaler=ds)

phase.add_state('Vy', 
                fix_initial=True, 
                shape=(nv,),
                rate_source='Vy_dot', 
                targets='Vy', 
                lower=-10.0,
                upper=10.0, 
                units='m/s',
                defect_scaler=ds)

phase.add_state('theta', 
                fix_initial=True, 
                shape=(nv,),
                rate_source='theta_dot', 
                targets='theta', 
                lower=0.0,
                upper=2*np.pi, 
                units='rad',
                defect_scaler=ds)

phase.add_state('E', 
                fix_initial=True, 
                rate_source='sq_thrust', 
                lower=0.0,
                units='min*N**2',
                defect_scaler=ds)


phase.add_input_parameter('t_start', 
                              targets='t_start', 
                              dynamic=False, 
                              shape=(nv,),
                              units='min',
                              val=np.random.uniform(0, 50, nv))

phase.add_input_parameter('t_end', 
                              targets='t_end', 
                              dynamic=False,
                              shape=(nv,), 
                              units='min',
                              val=np.random.uniform(150, 199, nv))


p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.options['print_results'] = False
p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
# p.driver.opt_settings['mu_init'] = 1.0E-2
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['linear_solver'] = 'mumps'
p.driver.opt_settings['max_iter'] = 500

phase.add_objective('E', loc='final', scaler=1)

phase.add_control('thrust', targets=['thrust'], shape=(nv,), lower=-10, upper=10, opt=True)
phase.add_control('theta_dot', targets=['theta_dot'], shape=(nv,), lower=-0.4, upper=0.4, opt=True)


p.driver.declare_coloring() 
p.setup(check=True)

p.set_val('traj.phase0.t_initial', 0)
p.set_val('traj.phase0.t_duration', t_duration)


theta = np.linspace(0, 2*np.pi, nv + 1)[:nv]

r = 100.0
x_start = r * np.cos(theta)
y_start = r * np.sin(theta)  

k = 1.0
theta2 = theta - np.pi + np.random.uniform(-np.pi/k, np.pi/k)
x_end = r * np.cos(theta2)
y_end = r * np.sin(theta2)

# for i in range(nv):
#     plt.plot([x_start[i], x_end[i]], [y_start[i], y_end[i]])
# plt.show()


p.set_val('traj.phase0.states:X', phase.interpolate(ys=[x_start, x_end], nodes='state_input'))
p.set_val('traj.phase0.states:X', phase.interpolate(ys=[y_start, y_end], nodes='state_input'))

z = np.zeros(nv)
p.set_val('traj.phase0.states:Vx', phase.interpolate(ys=[z, z], nodes='state_input'))
p.set_val('traj.phase0.states:Vy', phase.interpolate(ys=[z, z], nodes='state_input'))


p.run_driver()








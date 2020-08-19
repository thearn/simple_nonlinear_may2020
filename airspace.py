import numpy as np 
import openmdao.api as om
import dymos as dm

from schedule import Schedule
from vehicles import Vehicles
#from distance import Distances
from GridDistComp import GridDistComp

import matplotlib.pyplot as plt
import time


import pickle

np.random.seed(0)


limit=150.0

class Airspace(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_vehicles', types=int, default=4)

    def setup(self):
        nn = self.options['num_nodes']
        nv = self.options['num_vehicles']

        self.add_subsystem('vehicles', Vehicles(num_nodes=nn, 
                                                num_v=nv), 
                                       promotes=['*'])
        self.add_subsystem('distances', GridDistComp(num_nodes=nn, 
                                                     num_v=nv, 
                                                     limit=limit), 
                                        promotes=['*'])

nv = 15
ns = 25

p = om.Problem(model=om.Group())
traj = dm.Trajectory()

p.model.add_subsystem('traj', subsys=traj)
p.model.linear_solver = om.DirectSolver()

gl = dm.GaussLobatto(num_segments=ns, order=3)
#gl = dm.Radau(num_segments=ns, order=3)
nn = gl.grid_data.num_nodes

phase = dm.Phase(ode_class=Airspace,
                 ode_init_kwargs={'num_vehicles' : nv},
                 transcription=gl)

traj.add_phase(name='phase0', phase=phase)


phase.set_time_options(fix_initial=True, fix_duration=False, units='s')

traj.add_phase(name='phase0', phase=phase)

ds=1e-3
phase.add_state('X', 
                fix_initial=True,
                fix_final=True, 
                shape=(nv,),
                rate_source='X_dot', 
                targets='X',
                units='m', 
                lower=-1200.0,
                upper=1200.0, 
                defect_scaler=ds)

phase.add_state('Y', 
                fix_initial=True,
                fix_final=True, 
                shape=(nv,),
                rate_source='Y_dot', 
                targets='Y',
                units='m', 
                lower=-1200.0,
                upper=1200.0, 
                defect_scaler=ds)

# phase.add_state('E', 
#                 rate_source='sq_thrust', 
#                 fix_initial=False)


p.driver = om.pyOptSparseDriver()
# -------------------------
p.driver.options['optimizer'] = 'IPOPT'
p.driver.options['print_results'] = False
p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
# p.driver.opt_settings['mu_init'] = 1.0E-2
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['linear_solver'] = 'mumps'
p.driver.opt_settings['max_iter'] = 1500

# --------------------------

# p.driver.options['optimizer'] = 'SNOPT'
# p.driver.options['print_results'] = False
# p.driver.opt_settings['Major iterations limit'] = 1000000
# p.driver.opt_settings['Minor iterations limit'] = 1000000
# p.driver.opt_settings['Iterations limit'] = 1000000
# p.driver.opt_settings['iSumm'] = 6
# p.driver.opt_settings['Verify level'] = 0  # if you set this to 3 it checks partials, if you set it ot zero, ot doesn't check partials

# #p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-5
# p.driver.opt_settings['Major optimality tolerance'] = 1.0E-4

# --------------------------


phase.add_objective('time', loc='final')
#phase.add_objective('E', loc='final')

phase.add_control('Vx', targets=['Vx'], shape=(nv,), lower=-25, upper=25, units='m/s', ref=25.0, opt=True)
phase.add_control('Vy', targets=['Vy'], shape=(nv,), lower=-25, upper=25, units='m/s', ref=25.0, opt=True)

#phase.add_polynomial_control('V', targets='V', shape=(nv,), lower=-5, upper=5, units='m/s', opt=True, order=10)

phase.add_timeseries_output('dist')

p.model.add_constraint('traj.phase0.rhs_disc.dist', upper=0.0, scaler=100.0)

p.driver.declare_coloring() 

p.setup(check=True)

p.set_val('traj.phase0.t_initial', 0.0)
p.set_val('traj.phase0.t_duration', 100.0)


theta = np.linspace(0, 2*np.pi, nv + 1)[:nv]

r = 1000.0
x_start = r * np.cos(theta)
y_start = r * np.sin(theta)  

k = 12.0
theta2 = theta - np.pi + np.random.uniform(-np.pi/k, np.pi/k, nv)
x_end = r * np.cos(theta2)
y_end = r * np.sin(theta2)


th_start = np.random.uniform(0.0, 2*np.pi, nv)
th_end = np.random.uniform(0.0, 2*np.pi, nv)
p.set_val('traj.phase0.states:X', phase.interpolate(ys=[x_start, x_end], nodes='state_input'))
p.set_val('traj.phase0.states:Y', phase.interpolate(ys=[y_start, y_end], nodes='state_input'))
# p.set_val('traj.phase0.states:theta', phase.interpolate(ys=[th_start, th_end], nodes='state_input'))


t = time.time()
p.run_driver()

p.check_partials(compact_print=True)

print(time.time() - t, "seconds")

sim_out = p#traj.simulate()

t = sim_out.get_val('traj.phase0.timeseries.time')

dist = sim_out.get_val('traj.phase0.timeseries.dist')

for i in range(len(t)):
    print(i, t[i], dist[i])

X = sim_out.get_val('traj.phase0.timeseries.states:X')
Y = sim_out.get_val('traj.phase0.timeseries.states:Y')

#E = sim_out.get_val('traj.phase0.timeseries.states:E')

try:
    Vx = sim_out.get_val('traj.phase0.timeseries.controls:Vx')
except:
    Vx = sim_out.get_val('traj.phase0.timeseries.polynomial_controls:Vx')

try:
    Vy = sim_out.get_val('traj.phase0.timeseries.controls:Vy')
except:
    Vy = sim_out.get_val('traj.phase0.timeseries.polynomial_controls:Vy')
#print(E)

data = [t, X, Y, Vx, Vy, x_start, x_end, y_start, y_end, limit]
with open('flight.dat', 'wb') as f:
    pickle.dump(data, f)



plt.figure()
plt.subplot(211)
plt.plot(t, Vx)
plt.subplot(212)
plt.plot(t, Vy)

plt.figure()

for i in range(nv):
    plt.scatter(x_start, y_start)
    plt.scatter(x_end, y_end)
    x = X[:, i]
    y = Y[:, i]
    plt.plot(x, y)
plt.show()






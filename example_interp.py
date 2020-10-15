import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil, os
import openmdao.api as om
import dymos as dm

from schedule import Schedule
from vehicles import Vehicles
#from distance import Distances
from GridDistComp import GridDistComp

from AllDistComp import AllDistComp
from DeMux import DeMux
from AggregateMux import AggregateMux
from AggregateMuxKS import AggregateMuxKS

from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
from single_distance import SingleDistance
import time
import random
np.random.seed(0)

old_ns = 50
idx = random.sample(range(100), old_ns)
idx.sort()

theta = 0.8 * np.pi / 4
r = 1000.0
x_start = r*np.cos(theta)
y_start = r*np.sin(theta)

x_end = r*np.cos(np.pi + theta)
y_end = r*np.sin(np.pi + theta)

old_X = np.linspace(x_start, x_end, 100)[idx]
old_Y = np.linspace(y_start, y_end, 100)[idx]
old_t = np.linspace(0, 4700.0, 100)[idx]

# --------------


nv = 2
ns = 25 
limit = 100.0

class Airspace(om.Group):
        def initialize(self):
            self.options.declare('num_nodes', types=int)
            self.options.declare('nk', types=int)
            self.options.declare('num_vehicles', types=int, default=4)
            self.options.declare('separation', types=str, default='grid')


        def setup(self):
            nn = self.options['num_nodes']
            nv = self.options['num_vehicles']
            separation = self.options['separation']

            """Add in the old trajectory as a meta model
            """

            mm = MetaModelStructuredComp(method='slinear', vec_size=nn,
                                         extrapolate=True)
            mm.add_input('t', val=np.zeros(nn), training_data=old_t)
            mm.add_output('interp_x', val=np.zeros(nn), training_data=old_X)
            mm.add_output('interp_y', val=np.zeros(nn), training_data=old_Y)
            self.add_subsystem('mm', mm, promotes=['*'])

            # now add in trajectories to be solved with dymos
            self.add_subsystem('vehicles', Vehicles(num_nodes=nn, 
                                                    num_v=nv), 
                                           promotes=['*'])

            # add in distance calcs for solved trajectories
            self.add_subsystem('distances1', GridDistComp(num_nodes=nn, 
                                                         num_v=nv, 
                                                         limit=limit), 
                                            promotes=['*'])

            # add in distance calcs for solved trajectories to the fixed ones
            self.add_subsystem('distances2', SingleDistance(num_nodes=nn, 
                                                         num_v=nv), 
                                            promotes=['*'])
            self.connect('interp_x', 'fixed_x') 
            self.connect('interp_y', 'fixed_y')            


p = om.Problem(model=om.Group())
traj = dm.Trajectory()

p.model.add_subsystem('traj', subsys=traj)
p.model.linear_solver = om.DirectSolver()

gl = dm.GaussLobatto(num_segments=ns)
#gl = dm.Radau(num_segments=ns, order=3)
nn = gl.grid_data.num_nodes

phase = dm.Phase(ode_class=Airspace,
                 ode_init_kwargs={'num_vehicles' : nv, 'nk' : nn},
                 transcription=gl)

traj.add_phase(name='phase0', phase=phase)


phase.set_time_options(fix_initial=True, fix_duration=False, units='s',
                       targets=['t'])

traj.add_phase(name='phase0', phase=phase)

ds=1e-1
phase.add_state('X', 
                fix_initial=True,
                fix_final=True, 
                shape=(nv,),
                rate_source='X_dot', 
                targets='X',
                units='m', 
                lower=-1050.0,
                upper=1050.0, 
                defect_scaler=ds)

phase.add_state('Y', 
                fix_initial=True,
                fix_final=True, 
                shape=(nv,),
                rate_source='Y_dot', 
                targets='Y',
                units='m', 
                lower=-1050.0,
                upper=1050.0, 
                defect_scaler=ds)  



p.driver = om.pyOptSparseDriver()

p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['print_results'] = False
p.driver.opt_settings['Major iterations limit'] = 1000000
p.driver.opt_settings['Minor iterations limit'] = 1000000
p.driver.opt_settings['Iterations limit'] = 1000000
p.driver.opt_settings['iSumm'] = 6
p.driver.opt_settings['Verify level'] = -1  # if you set this to 3 it checks partials, if you set it ot zero, ot doesn't check partials

p.driver.opt_settings['Major optimality tolerance'] = 1.0E-16
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
p.driver.opt_settings['LU singularity tolerance'] = 1.0E-6



phase.add_objective('time', loc='final', scaler=1e10)
#phase.add_objective('E', loc='final')

phase.add_control('Vx', targets=['Vx'], shape=(nv,), lower=-25, upper=25, units='m/s', ref=25.0, opt=True)
phase.add_control('Vy', targets=['Vy'], shape=(nv,), lower=-25, upper=25, units='m/s', ref=25.0, opt=True)

phase.add_timeseries_output('dist', output_name='separation_constraint')

phase.add_timeseries_output('interp_x', output_name='interp_x')
phase.add_timeseries_output('interp_y', output_name='interp_y')

# distance constraints between solved trajectories
p.model.add_constraint('traj.phase0.rhs_disc.dist', equals=0.0, scaler=1e7)
p.model.add_constraint('traj.phase0.rhs_disc.dist_good', upper=0.0)

# distance constraint between solved and fixed/interpolated trajectory
p.model.add_constraint('traj.phase0.rhs_disc.dist_to_fixed', lower=limit)
p.driver.declare_coloring() 

p.setup(check=True)

p.set_val('traj.phase0.t_initial', 0.0)
p.set_val('traj.phase0.t_duration', 4e3)

theta = np.linspace(0, 2*np.pi, int(2*nv))[:-1]
#np.random.shuffle(theta)

theta_start = theta[:nv]
theta_end = np.roll(theta, nv//2)[:nv]
#theta2 = x_end = np.roll(x_port, nv//2)[:nv]


r = 1000
x_start = r * np.cos(theta_start)
y_start = r * np.sin(theta_start)  

theta_end = theta_start - np.pi + np.random.uniform(-np.pi/12, np.pi/12, nv)
x_end = r * np.cos(theta_end)
y_end = r * np.sin(theta_end)


p.set_val('traj.phase0.states:X', phase.interpolate(ys=[x_start, x_end], nodes='state_input'))
p.set_val('traj.phase0.states:Y', phase.interpolate(ys=[y_start, y_end], nodes='state_input'))

t = time.time()
p.run_model()

# save coloring file by nn, nv, 
p.run_driver()

print(time.time() - t, "seconds")

sim_out = traj.simulate()

t = sim_out.get_val('traj.phase0.timeseries.time')

print(t.max(), old_t.max())

separation_constraint = sim_out.get_val('traj.phase0.timeseries.separation_constraint')
print("max separation viol. :", np.max(separation_constraint))


X = sim_out.get_val('traj.phase0.timeseries.states:X')
Y = sim_out.get_val('traj.phase0.timeseries.states:Y')


interp_x = sim_out.get_val('traj.phase0.timeseries.interp_x')
interp_y = sim_out.get_val('traj.phase0.timeseries.interp_y')

try:
    Vx = sim_out.get_val('traj.phase0.timeseries.controls:Vx')
except:
    Vx = sim_out.get_val('traj.phase0.timeseries.polynomial_controls:Vx')

try:
    Vy = sim_out.get_val('traj.phase0.timeseries.controls:Vy')
except:
    Vy = sim_out.get_val('traj.phase0.timeseries.polynomial_controls:Vy')

# plt.figure()
# plt.subplot(211)
# plt.plot(t, Vx)
# plt.ylabel('Vx')
# plt.subplot(212)
# plt.xlabel('time')
# plt.ylabel('Vy')
# plt.plot(t, Vy)

plt.figure()


circle = plt.Circle((0, 0), 1000, fill=False)
plt.gca().add_artist(circle)
for i in range(nv):
    plt.scatter(x_start, y_start)
    plt.scatter(x_end, y_end)
    x = X[:, i]
    y = Y[:, i]
    plt.plot(x, y)

plt.plot(interp_x, interp_y, 'k--')

plt.xlim(-1100, 1100)
plt.ylim(-1100, 1100)
plt.show()


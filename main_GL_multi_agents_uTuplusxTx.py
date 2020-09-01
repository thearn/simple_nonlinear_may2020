# import libraries
import matplotlib.pyplot as plt
from matplotlib.patches import *
import sys
import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot
import time 
import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
import dymos as dm
from dymos.examples.plotting import plot_results 

# ###############################################
# import options and token
n_segments = 25
# N_trajectories = 2 # number of dynamic agents
N_coord = 2 # number of coordinates
number_static_obstacles = 3
number_agents = 8

# Define lower and upper constraints
v_lower = 15.
v_upper = 25.

theta_lower = -25.
theta_upper = 25.

# Define static obstacles
x0_so = np.array([18., 15, 18]) 
y0_so = np.array([16., 15, 18])
so_radius = np.array([3.0, 3.0, 2.0]) 
 
# Define agents
x0_agents = np.array([0.,100.,0.,100.,
                      0.,100, 50., 50.])
xm_agents = np.array([50.,50.,50.,50.,
                      50.,50.,50.,50.])
xf_agents = np.array([100.,0.,100.,0.,
                      100.,0.,50.,50.])

y0_agents = np.array([0.,0.,100.,100., 
                      50.,50.,0.,100.])
ym_agents = np.array([25.,85.,70.,15., 
                      60.,40.,60.,40.])
yf_agents = np.array([100.,100.,0.,0., 
                      50.,50.,100.,0.])

theta_init_agents = np.array([45.,135.,-45.,-135.,
                              0.,-180.,90.,-90.])

x_lower = min(np.concatenate((x0_agents, xf_agents), 
	                          axis=0))-10
x_upper = max(np.concatenate((x0_agents, xf_agents), 
	                          axis=0))+10

y_lower = min(np.concatenate((y0_agents, yf_agents), 
	                          axis=0))-10
y_upper = max(np.concatenate((y0_agents, yf_agents), 
	                          axis=0))+10

 
time_duration_lower = 0.5
time_duration_upper = 20.
time_duration_guess = 100.*np.sqrt(2)/v_upper

agent_radius = np.array([3.0, 3.0, 3.0, 3.0, 
                         3.0, 3.0,3.0,3.0]) 

obj_scaler = 50.

flag_continuity_v = True
flag_rate_continuity_v = False

flag_continuity_theta = True
flag_rate_continuity_theta = False

# ## tokens
token_optimal_solution = 1
token_gif = 1
 
class MinControlSubsystems(om.Group):
    """ 
    Notes:
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_static_obstacles', types=int)
        self.options.declare('num_agents', types=int)     


    def setup(self):
        nn = self.options['num_nodes']  
        nso = self.options['num_static_obstacles']   
        nagents = self.options['num_agents']     

        # subsystem: 2D motion
        for i in range(nagents): 

            self.add_subsystem(name='agent%i_2DmotionODE'%i, 
                               subsys = TwoDimensionalMotionODE(num_nodes=nn)) 
                
            for j in range(nso):

                name_soa_ij = 'agent%i_avoids_static_obstacle%i'% (i,j) 

                self.add_subsystem(name=name_soa_ij,
                                   subsys = StaticObstacleAvoidance(num_nodes=nn))
                # promotes_inputs=['x', 'y'])
            
            set_dynamic_avoidance = np.arange(i+1, nagents)
 
            for k in set_dynamic_avoidance:
                
                # print('agents_inner', i, k)

                name_aaa_ik = 'agent%i_avoids_agent%i'% (i,k)  

                self.add_subsystem(name=name_aaa_ik,
                                   subsys = DynamicObstacleAvoidance(num_nodes=nn))
        
        self.add_subsystem(name='penalize_control', 
                           subsys = PenalizeControl(num_nodes=nn, 
                                                    num_agents=nagents))

# #####################################################    
# #################################################### 
# #################################################### 

class PenalizeControl(om.ExplicitComponent):
    """
    Notes:
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_agents', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']       
        num_agents = self.options['num_agents']       
        
        for i in range(num_agents):
            # input: v
            self.add_input('agent%i_v'%i, 
                       val = np.zeros(num_nodes),
                       desc = 'agent%i_v'%i,
                       units = 'm/s')

            # input: theta
            self.add_input('agent%i_theta'%i, 
                           val = np.zeros(num_nodes),
                           desc = 'agent%i_theta'%i,
                           units = 'deg') 
             
            self.add_input('agent%i_x'%i, 
                       val = np.zeros(num_nodes),
                       desc = 'agent%i_x'%i,
                       units = 'm')

            # input: theta
            self.add_input('agent%i_y'%i, 
                           val = np.zeros(num_nodes),
                           desc = 'agent%i_y'%i,
                           units = 'm') 

        # outputs: d
        self.add_output('penalty_dot',
                        val = np.zeros(num_nodes), 
                        desc='control-penalty-at-a-time',
                        units='1/s')
        
        # Partials
        row_col = np.arange(num_nodes)
        
        for i in range(num_agents):

            self.declare_partials(of='penalty_dot', 
                                  wrt='agent%i_theta'%i, 
                                  rows = row_col,
                                  cols = row_col)

            self.declare_partials(of='penalty_dot', 
                                  wrt='agent%i_v'%i, 
                                  rows = row_col,
                                  cols = row_col)    

            self.declare_partials(of='penalty_dot', 
                                  wrt='agent%i_x'%i, 
                                  rows = row_col,
                                  cols = row_col)

            self.declare_partials(of='penalty_dot', 
                                  wrt='agent%i_y'%i, 
                                  rows = row_col,
                                  cols = row_col)    

    def compute(self, inputs, outputs):       
        num_agents = self.options['num_agents']  
        sum_ux_sq = 0.

        for i in range(num_agents):

            theta_i = inputs['agent%i_theta'%i]
            v_i = inputs['agent%i_v'%i] 
            x_i = inputs['agent%i_x'%i]
            y_i = inputs['agent%i_y'%i]
            sum_ux_sq = sum_ux_sq + theta_i**2 \
                                  + v_i**2 \
                                  + (x_i-xf_agents[i]) ** 2 \
                                  + (y_i-yf_agents[i]) ** 2

        outputs['penalty_dot'] = sum_ux_sq
    

    def compute_partials(self, inputs, partials):
        num_agents = self.options['num_agents']  

        for i in range(num_agents):

            theta_i = inputs['agent%i_theta'%i]
            v_i = inputs['agent%i_v'%i] 
            x_i = inputs['agent%i_x'%i]
            y_i = inputs['agent%i_y'%i]
            # sum_u_sq = sum_u_sq + theta_i**2 + v_i**2

            partials['penalty_dot', 'agent%i_theta'%i] = \
                                        2 * theta_i 
            partials['penalty_dot', 'agent%i_v'%i] = \
                                        2 * v_i 
            partials['penalty_dot', 'agent%i_x'%i] = \
                                        2 * (x_i-xf_agents[i])
            partials['penalty_dot', 'agent%i_y'%i] = \
                                        2 * (y_i-yf_agents[i])               
# ##################################################### 
# #################################################### 
# #################################################### 

class TwoDimensionalMotionODE(om.ExplicitComponent): 

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes'] 

        self.add_input('v', val=np.zeros(nn), desc='velocity', 
                       units='m/s') 

        self.add_input('theta', val=np.ones(nn), desc='angle', 
                        units='rad')

        self.add_input('theta_init', val=np.ones(nn), desc='initial angle', 
                        units='rad')        
        # #############
        # outputs
        # #############
        self.add_output('xdot', val=np.zeros(nn), 
                        desc='velocity component in x', units='m/s')

        self.add_output('ydot', val=np.zeros(nn), 
                        desc='velocity component in y', units='m/s') 

        # Setup partials
        arange = np.arange(self.options['num_nodes']) 

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta_init', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta_init', rows=arange, cols=arange) 

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        theta_init = inputs['theta_init']

        cos_the = np.cos(theta_init+theta)
        sin_the = np.sin(theta_init+theta)

        v = inputs['v'] 

        outputs['xdot'] = v * cos_the
        outputs['ydot'] = v * sin_the 

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        theta_init = inputs['theta_init']

        cos_the = np.cos(theta_init+theta)
        sin_the = np.sin(theta_init+theta)
        
        cthe__thei = -sin_the
        sthe__thei = cos_the
        cthe__the = -sin_the
        sthe__the = cos_the 
        v = inputs['v']  

        jacobian['xdot', 'v'] = cos_the
        jacobian['xdot', 'theta'] = v * cthe__the
        jacobian['xdot', 'theta_init'] = v * cthe__thei

        jacobian['ydot', 'v'] = sin_the
        jacobian['ydot', 'theta'] = v * sthe__the
        jacobian['ydot', 'theta_init'] = v * sthe__thei 

# ##################################################### 
# #################################################### 
# #################################################### 

class DynamicObstacleAvoidance(om.ExplicitComponent):
    """
    Notes:
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']   
        
        # input: x-dyn.obstacle
        self.add_input(name='x_obs', 
                       val = 0.0 * np.ones(num_nodes), 
                       desc='x-coordinate of the dynamic obstacle',
                       units='m')

        # inputs: y-dyn.obstacle
        self.add_input(name='y_obs',
                       val = 0.0 * np.ones(num_nodes),
                       desc='y-coordinate of the dynamic obstacle',
                       units='m')

        # input: x
        self.add_input(name='x', 
                       val = np.zeros(num_nodes),
                       desc='x-coordinate',
                       units='m')

        # inputs: y
        self.add_input(name='y',
                       val = np.zeros(num_nodes),  
                       desc='y-coordinate',
                       units='m')

        # outputs: d
        self.add_output(name='distance',
                        val = np.zeros(num_nodes), 
                        desc='distance from obstacle',
                        units='m**2')

        # Partials
        row_col = np.arange(num_nodes)

        self.declare_partials(of='distance', wrt='x', 
                              rows = row_col,
                              cols = row_col)

        self.declare_partials(of='distance', wrt='y', 
                              rows = row_col,
                              cols = row_col)

        self.declare_partials(of='distance', wrt='x_obs', 
                              rows = row_col,
                              cols = row_col)

        self.declare_partials(of='distance', wrt='y_obs', 
                              rows = row_col,
                              cols = row_col)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        x_obs = inputs['x_obs']
        y_obs = inputs['y_obs']

        outputs['distance'] = (x-x_obs)**2 + (y-y_obs)**2 
    
    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        x_obs = inputs['x_obs']
        y_obs = inputs['y_obs']
     
        # d = np.sqrt( (x-5)**2 + (y-5)**2 )

        partials['distance', 'x'] = 2 * (x-x_obs) 
        partials['distance', 'y'] = 2 * (y-y_obs)
        partials['distance', 'x_obs'] = -2 * (x-x_obs) 
        partials['distance', 'y_obs'] = -2 * (y-y_obs)

# ###################################################### 
# ####################################################  

class StaticObstacleAvoidance(om.ExplicitComponent):
    """
    Notes:
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']       

        # input: x
        self.add_input(name='x', 
                       val = np.zeros(num_nodes),
                       desc='x-coordinate',
                       units='m')

        # inputs: y
        self.add_input(name='y',
                       val = np.zeros(num_nodes),  
                       desc='y-coordinate',
                       units='m')

        # input: x
        self.add_input(name='x0', 
                       val = np.zeros(num_nodes),
                       desc='x0-coordinate obstacle',
                       units='m')

        # inputs: y
        self.add_input(name='y0',
                       val = np.zeros(num_nodes),
                       desc='y-coordinate obstacle',
                       units='m')

        
        # outputs: d
        self.add_output(name='distance',
                        val = np.zeros(num_nodes), 
                        desc='distance from obstacle',
                        units='m**2')

        # Partials
        row_col = np.arange(num_nodes)

        self.declare_partials(of='distance', wrt='x', 
                              rows = row_col,
                              cols = row_col)

        self.declare_partials(of='distance', wrt='y', 
                              rows = row_col,
                              cols = row_col)

        self.declare_partials(of='distance', wrt='x0', 
                              rows = row_col,
                              cols = row_col)

        self.declare_partials(of='distance', wrt='y0', 
                              rows = row_col,
                              cols = row_col)      

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        x0 = inputs['x0']
        y0 = inputs['y0']
        outputs['distance'] = (x-x0)**2 + (y-y0)**2 
    
    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        x0 = inputs['x0']
        y0 = inputs['y0']

        partials['distance', 'x'] = 2 * (x-x0) 
        partials['distance', 'y'] = 2 * (y-y0)

        partials['distance', 'x0'] = -2 * (x-x0) 
        partials['distance', 'y0'] = -2 * (y-y0)

# #################################################### 

# #####################################################
p = om.Problem(model=om.Group())
# p.driver = om.ScipyOptimizeDriver()
p.driver = om.pyOptSparseDriver()


p.driver.declare_coloring() 
#p.driver.use_fixed_coloring()

# p.driver.options['optimizer'] = 'IPOPT'
# p.driver.options['print_results'] = False
# p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
# # p.driver.opt_settings['mu_init'] = 1.0E-2
# #p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
# p.driver.opt_settings['print_level'] = 5
# p.driver.opt_settings['linear_solver'] = 'mumps'
# p.driver.opt_settings['max_iter'] = 15000


p.driver.options['optimizer'] = 'SNOPT' # 'SLSQP'
p.driver.opt_settings['Major feasibility tolerance'] = 1e-9
p.driver.options['print_results'] = False
p.driver.opt_settings['Major iterations limit'] = 1000000
p.driver.opt_settings['Minor iterations limit'] = 1000000
p.driver.opt_settings['Iterations limit'] = 1000000
p.driver.opt_settings['iSumm'] = 6
# p.driver.options['user_terminate_signal'] = signal.SIGUSR2

token_record=False
if token_record:
    p.driver.add_recorder(om.SqliteRecorder('twodim%d.sql' % n_segments))  
    p.driver.recording_options['includes'] = ['*']
    p.driver.recording_options['record_constraints'] = True
    p.driver.recording_options['record_inputs'] = True
    p.driver.recording_options['record_objectives'] = True
    p.driver.recording_options['record_desvars'] = True
    p.driver.recording_options['record_responses'] = True
    p.driver.recording_options['record_derivatives'] = True

# add trajectory
traj = p.model.add_subsystem('traj', dm.Trajectory())

phase = traj.add_phase('phase0', dm.Phase(ode_class=MinControlSubsystems,
                                          ode_init_kwargs={'num_static_obstacles': number_static_obstacles, 
                                                           'num_agents': number_agents}, 
                                          transcription=dm.GaussLobatto(num_segments=n_segments) 
                                          ) 
                       )

# #########################3# #########################
phase.set_time_options(initial_bounds=(0, 0), 
                       duration_bounds=(time_duration_lower, time_duration_upper),
                       units = 's') #targets = 'time')

# ###################################################
for i in range(number_agents):
    
    target_x, target_y = [], []
    target_x_aa, target_y_aa = [], []

    for j in range(number_static_obstacles):

        target_x__ij = 'agent%i_avoids_static_obstacle%i.x'% (i,j)
        target_x.append(target_x__ij)

        target_y__ij = 'agent%i_avoids_static_obstacle%i.y'% (i,j)
        target_y.append(target_y__ij)

    set_dynamic_avoidance = np.arange(i+1, number_agents)
    
    for k in set_dynamic_avoidance:

        target_x_ij = 'agent%i_avoids_agent%i.x'% (i,k)
        target_x.append(target_x_ij)

        target_y_ij = 'agent%i_avoids_agent%i.y'% (i,k)
        target_y.append(target_y_ij)
    
    set_dynamic_avoidance = np.arange(0, i)
    
    for k in set_dynamic_avoidance:

        target_x_ij = 'agent%i_avoids_agent%i.x_obs'% (k,i)
        target_x.append(target_x_ij)

        target_y_ij = 'agent%i_avoids_agent%i.y_obs'% (k,i)
        target_y.append(target_y_ij)

    target_x.append('penalize_control.agent%i_x'%i)
    target_y.append('penalize_control.agent%i_y'%i)

    phase.add_state('agent%i_x'%i, 
                    rate_source='agent%i_2DmotionODE.xdot'%i,
                    targets=target_x, 
                    units='m', 
                    lower = x_lower, upper = x_upper,
                    fix_initial=True, fix_final=True, solve_segments=False)        
    
    phase.add_state('agent%i_y'%i, 
                    rate_source='agent%i_2DmotionODE.ydot'%i,
                    targets=target_y,
                    units='m',
                    lower = y_lower, upper = y_upper,
                    fix_initial=True, fix_final=True, solve_segments=False)

    # name_target_v = 'agent%i_v'%i
    target_v = [] 
    target_v.append('agent%i_2DmotionODE.v'%i)
    target_v.append('penalize_control.agent%i_v'%i)

    phase.add_control('agent%i_v'%i, 
                      targets= target_v,
                      continuity=flag_continuity_v, 
                      rate_continuity=flag_rate_continuity_v, 
                      units='m/s', 
                      lower=v_lower, upper=v_upper)

    target_theta = [] 
    target_theta.append('agent%i_2DmotionODE.theta'%i)
    target_theta.append('penalize_control.agent%i_theta'%i)

    phase.add_control('agent%i_theta'%i, 
                      targets=target_theta,
                      continuity=flag_continuity_theta, 
                      rate_continuity=flag_rate_continuity_theta,
                      units='deg', 
                      lower=theta_lower, upper=theta_upper)

    phase.add_input_parameter('agent%i_theta_init'%i, 
                              targets='agent%i_2DmotionODE.theta_init'%i,
                              units='deg')


phase.add_state('control_penalty', 
                 rate_source='penalize_control.penalty_dot',  
                 units=None, #lower = x_lower, upper = x_upper,
                 fix_initial=True, fix_final=False, solve_segments=False) 

# ################################################################
# Static Obstacle 
for j in range(number_static_obstacles):

    target_x0 = []
    target_y0 = []

    for i in range(number_agents):  

        target_x0_ij = 'agent%i_avoids_static_obstacle%i.x0'% (i,j)
        target_x0.append(target_x0_ij)
        target_y0_ij = 'agent%i_avoids_static_obstacle%i.y0'% (i,j)
        target_y0.append(target_y0_ij)
    
    #
    phase.add_input_parameter('x0_so%i'%j,   
                                  units='m', targets=target_x0)
    # 
    phase.add_input_parameter('y0_so%i'%j,   
                                  units='m', targets=target_y0)
 

assumptions =p.model.add_subsystem('assumptions', om.IndepVarComp())

for i in range(number_static_obstacles):
    assumptions.add_output('x0_so%i'%i, val=x0_so[i], units='m')
    assumptions.add_output('y0_so%i'%i, val=y0_so[i], units='m')
    p.model.connect('assumptions.x0_so%i'%i, 'traj.phase0.input_parameters:x0_so%i'%i)
    p.model.connect('assumptions.y0_so%i'%i, 'traj.phase0.input_parameters:y0_so%i'%i)

for i in range(number_agents):
    assumptions.add_output('agent%i_theta_init'%i, val=theta_init_agents[i], units='deg') 
    p.model.connect('assumptions.agent%i_theta_init'%i, 'traj.phase0.input_parameters:agent%i_theta_init'%i) 

 
# ##########################################################################

for i in range(number_agents):

    for j in range(number_static_obstacles):

        phase.add_path_constraint('agent%i_avoids_static_obstacle%i.distance'% (i,j), 
                                  constraint_name = 'static%i%i_distance'% (i,j),
                                  units='m**2',
                                  lower=(so_radius[j]+agent_radius[i])**2, 
                                  scaler=1.)
             
        phase.add_timeseries_output('agent%i_avoids_static_obstacle%i.distance'% (i,j), 
                                     output_name = 'static%i%i_distance'% (i,j),
                                     units='m**2')

    set_dynamic_avoidance = np.arange(i+1, number_agents)
 
    for k in set_dynamic_avoidance: 
        
        phase.add_path_constraint('agent%i_avoids_agent%i.distance'% (i,k), 
                                  constraint_name = 'dynamic%i%i_distance'% (i,k),
                                  units='m**2',
                                  lower=(agent_radius[k]+agent_radius[i])**2,  
                                  scaler=1.)
        
        phase.add_timeseries_output('agent%i_avoids_agent%i.distance'% (i,k), 
                                     output_name = 'dynamic%i%i_distance'% (i,k),
                                     units='m**2')
 
# ###################################################
phase.add_objective('control_penalty', 
                    loc='final', 
                    scaler=obj_scaler) 

p.setup() 
 
for i in range(number_agents):
    # ###
    p['traj.phase0.controls:agent%i_theta'%i] = \
                phase.interpolate(ys=[theta_lower, theta_upper], nodes='control_input')
    # ###
    p['traj.phase0.controls:agent%i_v'%i] = \
                phase.interpolate(ys=[v_lower, v_upper], nodes='control_input')
    # ###
    p['traj.phase0.states:agent%i_x'%i] = \
                phase.interpolate(xs=[0, 1, 2.], ys=[x0_agents[i], xm_agents[i], xf_agents[i]], nodes='state_input')
    # ###
    p['traj.phase0.states:agent%i_y'%i] = \
                phase.interpolate(xs=[0, 1, 2.],  ys=[y0_agents[i], ym_agents[i], yf_agents[i]], nodes='state_input')  

# ###              
p['traj.phase0.t_duration'] = time_duration_guess
p['traj.phase0.t_initial'] = 0.0

start_time = time.time()
p.run_driver()
print("%s sec" % (time.time() - start_time))

print(p.get_val('traj.phase0.timeseries.time')[-1])
start_time = time.time()
exp_out = traj.simulate()
print("%s sec" % (time.time() - start_time))

# ###################################################
#  Figures for Optimal Solutions
# ###################################################

if token_optimal_solution:
    
    # ###################################################
    #  Figures for Optimal Solutions (Altogether)
    # ###################################################

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    # #####################
    # ####  x vs. y    ####
    # #####################
    for i in range(number_agents):
        axes[0, 0].plot(p.get_val('traj.phase0.timeseries.states:agent%i_x'%i),
                    p.get_val('traj.phase0.timeseries.states:agent%i_y'%i),
                    ms=4,
                    linestyle=None,
                    marker='o',
                    label='solution')
    
        axes[0, 0].plot(exp_out.get_val('traj.phase0.timeseries.states:agent%i_x'%i),
                    exp_out.get_val('traj.phase0.timeseries.states:agent%i_y'%i),                
                    ms=4,
                    linestyle='-',
                    marker=None,                
                    label='simulation')
    
    # circle = np.array(number_static_obstacles)

    for i in range(number_static_obstacles):
        circle_i = plt.Circle((x0_so[i], y0_so[i]), 
                    so_radius[i], 
                    color='red', 
                    alpha = 0.5, 
                    edgecolor=None) 
        axes[0, 0].add_artist(circle_i)    
                              
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('y (m)')
    axes[0, 0].grid()
    # #####################
    # ### v versus time ###
    # #####################
    for i in range(number_agents):
        axes[0, 1].plot(p.get_val('traj.phase0.timeseries.time'),
                p.get_val('traj.phase0.timeseries.controls:agent%i_v'%i),
                ms=4,
                linestyle=None,
                marker='o')
        axes[0, 1].plot(exp_out.get_val('traj.phase0.timeseries.time'),
                exp_out.get_val('traj.phase0.timeseries.controls:agent%i_v'%i),
                ms=4,
                linestyle='-',
                marker=None)
    axes[0, 1].set_xlabel('time (s)')
    axes[0, 1].set_ylabel('v (m/s)') 
    axes[0, 1].grid()
    # ##########################
    # ### theta versus time #### 
    # ##########################   
    for i in range(number_agents): 
        axes[1, 0].plot(p.get_val('traj.phase0.timeseries.time'),
                p.get_val('traj.phase0.timeseries.controls:agent%i_theta'%i),
                ms=4,                
                marker='o',
                linestyle='None')
        axes[1, 0].plot(exp_out.get_val('traj.phase0.timeseries.time'),
                exp_out.get_val('traj.phase0.timeseries.controls:agent%i_theta'%i),
                linestyle='-',
                ms=4, 
                marker=None)
    axes[1, 0].set_xlabel('time (s)')
    axes[1, 0].set_ylabel('theta (deg)') 
    axes[1, 0].grid()
    # ####################################
    # ### static distance versus time #### 
    # ####################################
    time_val = p.get_val('traj.phase0.timeseries.time')
    time_val_shape = time_val.shape

    for i in range(number_agents):
        for j in range(number_static_obstacles):
        
            axes[1, 1].plot(p.get_val('traj.phase0.timeseries.time'),
                np.sqrt(p.get_val('traj.phase0.timeseries.static%i%i_distance' %(i,j) )),
                linestyle=None,
                ms=4,
                marker='o')
        
            axes[1, 1].plot(exp_out.get_val('traj.phase0.timeseries.time'),
                np.sqrt(exp_out.get_val('traj.phase0.timeseries.static%i%i_distance'%(i,j) )),
                linestyle = '-',
                ms = 4, 
                marker=None)    

            axes[1, 1].plot(p.get_val('traj.phase0.timeseries.time'),
                (so_radius[j]+agent_radius[i])*np.ones(time_val_shape),
                linestyle = '-',
                dashes=[6,2],
                ms = 4, 
                marker=None)

    axes[1, 1].set_xlabel('time (s)')
    axes[1, 1].set_ylabel('Distance from the static obstacles (m)') 
    axes[1, 1].grid()
    plt.suptitle('2-D Motion ODE Solution - High-Order Gauss-Lobatto Method')
    # fig.legend(loc='lower center', ncol=2)
    plt.savefig('ode_GL_nseg%d.pdf' % n_segments)
    # plt.show()

    # ###################################################################
    #  Figures for Optimal Solutions (Dynamic Obstacle Avoidance) #######
    # ###################################################################
    time_val = p.get_val('traj.phase0.timeseries.time')
    time_val_shape = time_val.shape  
    fig, axes = plt.subplots(nrows=number_agents, ncols=number_agents, figsize=(10, 8))


    for i in range(number_agents): 

        set_dynamic_avoidance = np.arange(i+1, number_agents)
 
        for k in set_dynamic_avoidance:
            
            print('agents', i, k)

            # fig, axes = plt.subplots()
    
            axes[i, k].plot(p.get_val('traj.phase0.timeseries.time'),
                      np.sqrt(p.get_val('traj.phase0.timeseries.dynamic%i%i_distance'%(i,k) )),
                      linestyle=None,
                      ms=4,
                      marker='o')

            axes[i, k].plot(exp_out.get_val('traj.phase0.timeseries.time'),
                      np.sqrt(exp_out.get_val('traj.phase0.timeseries.dynamic%i%i_distance'% (i,k) )),
                      linestyle = '-',
                      ms = 4, 
                      marker=None)  
            
            axes[i, k].plot(p.get_val('traj.phase0.timeseries.time'),
                     (agent_radius[i]+agent_radius[k])*np.ones(time_val_shape),
                      linestyle = '-', #dashes=[6,2],
                      ms = 4, 
                      marker=None)

            # axes.set_xlabel('time (s)')
            axes[i, k].set_ylabel('Dist.btw.A%i and A%i (m)' %(i,k) ) 
            axes[i, k].grid()
    
    plt.suptitle('2-D Motion ODE Solution - High-Order Gauss-Lobatto Method')
    plt.savefig('dyn_dist_GL_nseg%i_nagents%i.pdf' % (n_segments, number_agents)) 


    # ###################################################################
    #  Figures for Optimal Solutions (Static Obstacle Avoidance) #######
    # ###################################################################    
    fig, axes = plt.subplots(nrows=number_agents, ncols=number_static_obstacles, figsize=(10, 8))

    for i in range(number_agents):  
 
        for j in range(number_static_obstacles):
        
            axes[i,j].plot(p.get_val('traj.phase0.timeseries.time'),
                np.sqrt(p.get_val('traj.phase0.timeseries.static%i%i_distance' %(i,j) )),
                linestyle=None,
                ms=4,
                marker='o')
        
            axes[i,j].plot(exp_out.get_val('traj.phase0.timeseries.time'),
                np.sqrt(exp_out.get_val('traj.phase0.timeseries.static%i%i_distance'%(i,j) )),
                linestyle = '-',
                ms = 4, 
                marker=None)    

            axes[i,j].plot(p.get_val('traj.phase0.timeseries.time'),
                (so_radius[j]+agent_radius[i])*np.ones(time_val_shape),
                linestyle = '-', #dashes=[6,2],
                ms = 4, 
                marker=None)
            axes[i,j].grid()
            axes[i,j].set_ylabel('Dist.btw.A%i and SO%i (m)'%(i,j)) 
        # axes.set_xlabel('time (s)')
    
    plt.suptitle('2-D Motion ODE Solution - High-Order Gauss-Lobatto Method') 
    plt.savefig('static_dist_GL_nseg%i_nagents%i.pdf' % (n_segments, number_agents)) 

import numpy as np 
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

import matplotlib.pyplot as plt
import time


import pickle

def generate_airspace(nv=5, ns=25, limit=100.0, 
                      separation='grid', airspace_type=0, aggregate='mine', seed=0):
    np.random.seed(seed)
    print(30*"=")
    print("Starting run with nv = %i, ns = %i, limit = %2.2f" % (nv, ns, limit))
    print()
    class Airspace(om.Group):
        def initialize(self):
            self.options.declare('num_nodes', types=int)
            self.options.declare('num_vehicles', types=int, default=4)
            self.options.declare('separation', types=str, default='grid')

        def setup(self):
            nn = self.options['num_nodes']
            nv = self.options['num_vehicles']
            separation = self.options['separation']

            self.add_subsystem('vehicles', Vehicles(num_nodes=nn, 
                                                    num_v=nv), 
                                           promotes=['*'])

            if separation == 'grid':
                self.add_subsystem('distances', GridDistComp(num_nodes=nn, 
                                                             num_v=nv, 
                                                             limit=limit), 
                                                promotes=['*'])

            elif separation == 'pairwise':
                self.add_subsystem('demux', DeMux(num_nodes=nn, nv=nv), 
                                               promotes=['*'])
                nc = 0
                for i in range(nv):
                    for k in range(i + 1, nv):

                        self.add_subsystem('dist_%i_%i' % (i, k), 
                                           AllDistComp(num_nodes=nn, limit=limit))

                        self.connect('x_%i' % i, 'dist_%i_%i.x1' % (i, k))
                        self.connect('y_%i' % i, 'dist_%i_%i.y1' % (i, k))
                        self.connect('x_%i' % k, 'dist_%i_%i.x2' % (i, k))
                        self.connect('y_%i' % k, 'dist_%i_%i.y2' % (i, k))

                        nc += 1

                if aggregate == 'mine':
                    self.add_subsystem('aggmux', AggregateMux(num_nodes=nn, nc=nc))

                    nc = 0
                    for i in range(nv):
                        for k in range(i + 1, nv):
                            self.connect('dist_%i_%i.dist' % (i, k), 'aggmux.dist_%i' % nc)
                            nc += 1


                elif aggregate == 'ks':
                    self.add_subsystem('aggmux', AggregateMuxKS(num_nodes=nn, nc=nc))
                    nc = 0
                    for i in range(nv):
                        for k in range(i + 1, nv):
                            self.connect('dist_%i_%i.dist' % (i, k), 'aggmux.dist_%i' % nc)
                            nc += 1

                    self.add_subsystem('ks', om.KSComp(width=nc, vec_size=nn))
                    self.connect('aggmux.distks', 'ks.g')

                elif aggregate == 'none':
                    self.add_subsystem('aggmux', AggregateMuxKS(num_nodes=nn, nc=nc))
                    nc = 0
                    for i in range(nv):
                        for k in range(i + 1, nv):
                            self.connect('dist_%i_%i.dist' % (i, k), 'aggmux.dist_%i' % nc)
                            nc += 1                    

    p = om.Problem(model=om.Group())
    traj = dm.Trajectory()

    p.model.add_subsystem('traj', subsys=traj)
    p.model.linear_solver = om.DirectSolver()

    gl = dm.GaussLobatto(num_segments=ns)
    #gl = dm.Radau(num_segments=ns, order=3)
    nn = gl.grid_data.num_nodes

    phase = dm.Phase(ode_class=Airspace,
                     ode_init_kwargs={'num_vehicles' : nv, 'separation' : separation},
                     transcription=gl)

    traj.add_phase(name='phase0', phase=phase)


    phase.set_time_options(fix_initial=True, fix_duration=False, units='s')

    traj.add_phase(name='phase0', phase=phase)

    ds=1e-1
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
    #                 defect_scaler=ds,
    #                 fix_initial=False)


    p.driver = om.pyOptSparseDriver()
    # -------------------------
    # p.driver.options['optimizer'] = 'IPOPT'
    # p.driver.options['print_results'] = False
    # p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
    # # p.driver.opt_settings['mu_init'] = 1.0E-2
    # #p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    # p.driver.opt_settings['print_level'] = 5
    # p.driver.opt_settings['linear_solver'] = 'mumps'
    # p.driver.opt_settings['max_iter'] = 15000

    # --------------------------

    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.options['print_results'] = False
    p.driver.opt_settings['Major iterations limit'] = 1000000
    p.driver.opt_settings['Minor iterations limit'] = 1000000
    p.driver.opt_settings['Iterations limit'] = 1000000
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Verify level'] = -1  # if you set this to 3 it checks partials, if you set it ot zero, ot doesn't check partials

    #p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-7
    #p.driver.opt_settings['LU singularity tolerance'] = 1.0E-6

    # --------------------------


    phase.add_objective('time', loc='final')
    #phase.add_objective('E', loc='final')

    phase.add_control('Vx', targets=['Vx'], shape=(nv,), lower=-25, upper=25, units='m/s', ref=25.0, opt=True)
    phase.add_control('Vy', targets=['Vy'], shape=(nv,), lower=-25, upper=25, units='m/s', ref=25.0, opt=True)
    
    if separation == 'grid':
        phase.add_timeseries_output('dist', output_name='separation_constraint')

        p.model.add_constraint('traj.phase0.rhs_disc.dist', equals=0.0, ref=0.01)
        p.model.add_constraint('traj.phase0.rhs_disc.dist_good', upper=0.0, ref=0.01)

    elif separation == 'pairwise':

        if aggregate == 'mine':
            p.model.add_constraint('traj.phase0.rhs_disc.aggmux.dist' , equals=0.0, ref=0.01)
            #p.model.add_constraint('traj.phase0.rhs_disc.aggmux.dist_good' , upper=0.0, ref=0.01)
            phase.add_timeseries_output('aggmux.dist', output_name='separation_constraint')

        elif aggregate == 'ks':
            p.model.add_constraint('traj.phase0.rhs_disc.ks.KS' , upper=0.0, ref=0.01)
            phase.add_timeseries_output('ks.KS', output_name='separation_constraint')

        elif aggregate == 'none':

            p.model.add_constraint('traj.phase0.rhs_disc.aggmux.distks' , upper=0.0, ref=0.01)
            phase.add_timeseries_output('aggmux.distks', output_name='separation_constraint', shape=(nv*(nv-1)//2,))

    p.driver.declare_coloring() 
    #p.driver.use_fixed_coloring(coloring='coloring_files/total_coloring.pkl')

    p.setup(check=True)

    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 4e3)

    if airspace_type == 0:
        x_port = []
        y_port = []

        port_limit = 1.25*limit
        while len(x_port) < 1.5*nv:
            x = np.random.uniform(-1000, 1000)
            y = np.random.uniform(-1000, 1000)


            #r = np.sqrt((x**2) + (y**2))
            #theta = np.arctan2(y, x)

            #r = r + (1000.0 - r)*0.75
            #x = np.cos(theta)*r
            #y = np.sin(theta)*r


            too_close = False
            for k in range(len(x_port)):
                d1 = np.sqrt((x - x_port[k])**2 + (y - y_port[k])**2)
                
                if d1 < port_limit:
                    too_close = True
                    #print(len(x_port), d1)
                    break

            if not too_close:
                x_port.append(x)
                y_port.append(y)

                print("creating port", len(x_port), "of", nv)


        x_start = x_port[:nv]
        x_end = np.roll(x_port, nv//2)[:nv]

        y_start = y_port[:nv]
        y_end = np.roll(y_port, nv//2)[:nv]

    else:
        theta = np.linspace(0, 2*np.pi, int(1.5*nv))[:-2]
        #np.random.shuffle(theta)

        theta_start = theta[:nv]
        theta_end = np.roll(theta, nv//2)[:nv]
        #theta2 = x_end = np.roll(x_port, nv//2)[:nv]


        r = 1000
        x_start = r * np.cos(theta_start)
        y_start = r * np.sin(theta_start)  
        
        too_close = True 

        bad_idx = -1

        while too_close:
            #theta_end = theta_start - np.pi + np.random.uniform(-np.pi/12, np.pi/12, nv)
            if bad_idx == -1:
                theta_end = np.random.uniform(0, 2*np.pi, nv)

            else:
                theta_end[bad_idx] = np.random.uniform(0, 2*np.pi)

            x_end = r * np.cos(theta_end)
            y_end = r * np.sin(theta_end)

            too_close = False
            for k in range(nv):

                d0 = np.sqrt((x_start[k] - x_end[k])**2 + (y_start[k] - y_end[k])**2)
                if d0 < limit:
                    print("destination too close to origin...")
                    bad_idx = k
                    too_close=True
                    break

                for j in range(k + 1, nv):

                    d1 = np.sqrt((x_start[k] - x_start[j])**2 + (y_start[k] - y_start[j])**2)
                    d2 = np.sqrt((x_end[k] - x_end[j])**2 + (y_end[k] - y_end[j])**2)

                    if d1 < limit or d2 < limit:
                        bad_idx = k
                        print("ports too close! regenerating...")
                        too_close = True
                        break
                if too_close:
                    break


        #x_start[10], x_start[3] = x_start[3], x_start[10]
        #y_start[10], y_start[3] = y_start[3], y_start[10]
    # for i in range(nv):
    #     print(x_start[i], y_start[i], "--->", x_end[i], y_end[i])
    # quit()
    p.set_val('traj.phase0.states:X', phase.interpolate(ys=[x_start, x_end], nodes='state_input'))
    p.set_val('traj.phase0.states:Y', phase.interpolate(ys=[y_start, y_end], nodes='state_input'))

    
    p.run_model()

    # save coloring file by nn, nv, 
    t = time.time()
    print("starting")
    p.run_driver()

    #p.check_partials(compact_print=True)
    print(50*"=")
    print(nv, aggregate, separation, airspace_type)
    print(time.time() - t, "seconds")
    quit()
    # dist = p.get_val('traj.phase0.timeseries.dist')
    # for i in range(len(dist)):
    #     print(i, dist[i])


    sim_out = traj.simulate()

    t = sim_out.get_val('traj.phase0.timeseries.time')

    separation_constraint = sim_out.get_val('traj.phase0.timeseries.separation_constraint')
    print("max separation viol. :", np.max(separation_constraint))

    
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

    data = [t, X, Y, Vx, Vy, x_start, x_end, y_start, y_end, nv, ns, limit, separation, airspace_type, aggregate]
    with open('flight.dat', 'wb') as f:
        pickle.dump(data, f)



    # plt.figure()
    # plt.subplot(211)
    # plt.plot(t, Vx)
    # plt.ylabel('Vx')
    # plt.subplot(212)
    # plt.xlabel('time')
    # plt.ylabel('Vy')
    # plt.plot(t, Vy)

    # plt.figure()

    # if airspace_type == 0:
    #     plt.plot([-1000, 1000], [1000, 1000], 'k')
    #     plt.plot([-1000, -1000], [1000, -1000], 'k')
    #     plt.plot([-1000, 1000], [-1000, -1000], 'k')
    #     plt.plot([1000, 1000], [-1000, 1000], 'k')
    # else:
    #     circle = plt.Circle((0, 0), 1000, fill=False)
    #     plt.gca().add_artist(circle)
    # for i in range(nv):
    #     plt.scatter(x_start, y_start)
    #     plt.scatter(x_end, y_end)
    #     x = X[:, i]
    #     y = Y[:, i]
    #     plt.plot(x, y)
    # plt.show()


if __name__ == '__main__':
    # [2, 4, 8, 15, 30, 50, 100]
    generate_airspace(nv=49, # number of vehicles
                      ns=20, # number of sample points for dymos
                      limit=45.0, # separation limit (in km)
                      airspace_type = 0, # 0 = square region, low interaction. 1 = circular region, high interaction
                      separation='pairwise', # separation method. 'grid', 'pairwise', or 'none'
                      aggregate='ks', # separation constraint aggregation. 'mine', 'ks', or 'none'
                      seed=5)# random seed for numpy



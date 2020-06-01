import numpy as np
from numpy import sin, cos
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import time

a = 3.0

def vector_bool(x, a, b):
    pass



class Schedule(om.ExplicitComponent):


    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_v', types=int, default=4)

    def setup(self):
        nn = self.options['num_nodes']
        nr = self.options['num_v']

        self.add_input('t',
               val=np.zeros(nn), units='min')
        # params
        self.add_input('t_start',
               val=np.zeros(nr), units='min')

        self.add_input('t_end',
               val=np.zeros(nr), units='min')

        # schedule based scaling
        self.add_output('c_schedule',
               val=np.zeros((nn, nr)), units=None)

        self.add_output('schedule_ordering',
               val=np.zeros(nr), units=None)


        arange1 = np.arange(nn * nr, dtype=int)

        arange2 = []
        for i in range(nn):
            arange2 += nr * [i]

        arange3 = []
        for i in range(nn):
            arange3.extend(range(nr))

        self.declare_partials('c_schedule', 't', rows=arange1, cols=arange2)
        self.declare_partials('c_schedule', ['t_start', 't_end'], rows=arange1, cols=arange3)

        self.declare_partials('schedule_ordering', ['t_start', 't_end'], rows=np.arange(nr), cols=np.arange(nr))

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        t = inputs['t']
        t_start = inputs['t_start']
        t_end = inputs['t_end']

        # enable/disable control thrust based on start/stop times
        tt, tt_start = np.meshgrid(t, t_start)
        tt, tt_end = np.meshgrid(t, t_end)

        d_ton = np.exp(-a*(tt - tt_start)).T
        d_toff = np.exp(-a*(-tt + tt_end)).T

        d_ton[np.where(d_ton > 1e10)] = 1e10
        d_toff[np.where(d_toff > 1e10)] = 1e10

        y = 1 / (1 + d_ton) * 1 / (1 + d_toff) 

        outputs['c_schedule'] = y
        outputs['schedule_ordering'] = t_end - t_start


    def compute_partials(self, inputs, jacobian):
        nn = self.options['num_nodes']

        t = inputs['t']
        t_start = inputs['t_start']
        t_end = inputs['t_end']

        # enable/disable control thrust based on start/stop times
        tt, tt_start = np.meshgrid(t, t_start)
        tt, tt_end = np.meshgrid(t, t_end)

        d_ton = np.exp(-a*(tt - tt_start)).T
        d_toff = np.exp(-a*(-tt + tt_end)).T

        d_ton[np.where(d_ton > 1e10)] = 1e10
        d_toff[np.where(d_toff > 1e10)] = 1e10

        jacobian['c_schedule', 't'] = (a*d_ton/((1 +d_toff)*(1 + d_ton)**2) - a*d_toff/((1 +d_toff)**2*(1 + d_ton))).flatten()

        dc_dts = -a*d_ton/((1 +d_toff)*(1 + d_ton)**2)
        jacobian['c_schedule', 't_start'] = dc_dts.flatten()

        dc_dte = a*d_toff/((1 +d_toff)**2*(1 + d_ton))
        jacobian['c_schedule', 't_end'] = dc_dte.flatten()

        jacobian['schedule_ordering', 't_end'] = 1.0
        jacobian['schedule_ordering', 't_start'] = -1.0

        # jacobian['y', 't_start'] = -a*d_ton/((1 +d_toff)*(1 + np.exp(-a*(tt - tt_start)))**2)
        # jacobian['y', 't_end'] = a*np.exp(-a*(-tt + tt_end))/((1 +d_toff)**2*(1 + np.exp(-a*(tt - tt_start))))
        # jacobian['y', 'a'] = -(-tt + tt_start)*np.exp(-a*(tt - tt_start))/((1 +d_toff)*(1 + np.exp(-a*(tt - tt_start)))**2) - (tt - tt_end)*np.exp(-a*(-tt + tt_end))/((1 +d_toff)**2*(1 + np.exp(-a*(tt - tt_start))))


if __name__ == '__main__':
    np.random.seed(0)
    p = om.Problem()
    p.model = om.Group()
    nv = 30
    n = 30

    p.model.add_subsystem('test', Schedule(num_nodes=n, num_v=nv), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)
    start = np.random.uniform(0, 10, nv)
    p['t_start'] = start
    end = np.zeros(nv)
    for i in range(nv):
        end[i] = np.random.uniform(start[i], 10)
    p['t_end'] =  end
    p['t'] = np.linspace(0,10,n)
    p.run_model()
    
    print(p['schedule_ordering'])
    # p.run_model()
    # plt.plot(p['t'], p['c_schedule'])
    # plt.show()
    # quit()

    check_partials_data = p.check_partials(compact_print=True, method='cs')

    # plot in non-binary mode
    om.partial_deriv_plot('c_schedule', 't_end', check_partials_data, binary = False)



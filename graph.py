from graphviz import Digraph

dot = Digraph(comment='ATM', format='png')
dot.attr(size='5000,5000')


dot.node('t', 'time', shape='hexagon')

dot.node('T', 'thrust', shape='triangle')
dot.node('t_on', 't_on', shape='triangle')
dot.node('t_off', 't_off', shape='triangle')
dot.node('td', 'theta_dot', shape='triangle')

dot.node('schedule', 'schedule')
dot.node('vehicle_eom', 'vehicle_eom')
dot.node('distance', 'distance')

dot.node('xdot', 'xdot', shape='square')
dot.node('ydot', 'ydot', shape='square')
dot.node('xdotdot', 'xdotdot', shape='square')
dot.node('ydotdot', 'ydotdot', shape='square')
dot.node('x', 'x', shape='square')
dot.node('y', 'y', shape='square')
dot.node('E', 'E', shape='square')

dot.edge('t', 'schedule')
dot.edge('t_on', 'schedule')
dot.edge('t_off', 'schedule')


dot.edge('x', 'distance')
dot.edge('y', 'distance')
dot.edge('schedule', 'distance')
dot.edge('schedule', 'vehicle_eom')

dot.edge('T', 'vehicle_eom')
dot.edge('td', 'vehicle_eom')

dot.edge('vehicle_eom', 'xdot')
dot.edge('vehicle_eom', 'ydot')
dot.edge('vehicle_eom', 'xdotdot')
dot.edge('vehicle_eom', 'ydotdot')
dot.edge('vehicle_eom', 'E')


dot.render('ATM', view=True)
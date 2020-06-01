from sympy import *

# type in component param names here
inputs = 'a x1 x2 y1 y2 c1 c2 d_min d'

# ----------------
outputs = {}
inputs_unpacked = ', '.join(inputs.split())
exec('%s = symbols("%s")' % (inputs_unpacked, inputs))
exec('input_symbs = [%s]' % inputs_unpacked)
# -----------------

# paste compute() code here

#d = sqrt((x1 - x2)**2 + (y1 - y2)**2)


d_close = exp(-a*(-d + d_min * c1 * c2))

y = 1 / (1 + d_close)

outputs['y'] = y

# ------------------
# ------------------
print()
print("    def compute_partials(self, inputs, partials):\n")
inputs_ns = ', '.join(["inputs['%s']" % inp for inp in inputs.split()])
print("       ", inputs_unpacked, "=", inputs_ns )
declare = {}

for oname in outputs:
    print()
    declare[oname] = []
    for iname in input_symbs:
        deriv = diff(outputs[oname], iname)
        if deriv != 0:
            if deriv == 1:
                deriv = 1.0

            
            deriv = 'np.exp'.join(str(deriv).split('exp'))
            deriv = 'd_ton'.join(str(deriv).split('np.exp(-a*(t - t_on))'))
            deriv = 'd_toff'.join(str(deriv).split('np.exp(-a*(-t + t_off))'))
            
            st = "\t\tjacobian['%s', '%s'] = %s" % (oname, iname, deriv)
            print(st)
            declare[oname].append(iname)

# declare partials
# ------------------
print("")
print('.       ' + 20*'#')
print("        arange = np.arange(self.options['num_nodes'], dtype=int)")
for oname in declare:
    list_inputs = ["'" + str(i) + "'" for i in declare[oname]]
    list_inputs = ', '.join(list_inputs)
    declare_statements = "\t\tself.declare_partials('%s', [%s], rows=arange, cols=arange)" % (oname, list_inputs)
    print(declare_statements)


# -*- coding: utf-8 -*-
"""
A demonstration of interacting period-3 and period-5 
logistic mapping systems can synchronize to become period-4

@author: HTW
"""

import numpy as np
import matplotlib.pyplot as plt

step = 400    # total step running in program
rx = 3.8318741    # 3.8318741 for period-3
ry = 3.7389149    # 3.7389149, 3.9057065 for two period-5 near period-3
x0 = 0.12924    # initial value for x
y0 = 0.2304    # initial value for y
xdry_const = 0.1    # strength of x drive y
ydrx_const = xdry_const    # strength of y drive x
inte_sta_ste = 250    # interaction start at which step
plot_from = 378    # plot from which step
plot_to = 400    # plot to which step

def logi_map(r, x):

    return r * x * (1 - x)

x = np.full(step, None, dtype=np.float64)
y = np.full(step, None, dtype=np.float64)
x[0] = x0
y[0] = y0

for m in range(1, step):
    x_intr = logi_map(rx, x[m-1])
    y_intr = logi_map(ry, y[m-1])
    if m < inte_sta_ste:
        x[m] = x_intr
        y[m] = y_intr
    else:
        x[m] = x_intr + ydrx_const * (y_intr-x_intr)
        y[m] = y_intr + xdry_const * (x_intr-y_intr)


plt.figure()
plt.plot(range(inte_sta_ste-19,inte_sta_ste-2), x[inte_sta_ste-19:inte_sta_ste-2], 'bo-', label='Intrinsic P-3')
plt.plot(range(inte_sta_ste-19,inte_sta_ste-2), y[inte_sta_ste-19:inte_sta_ste-2], 'ro-', label='Intrinsic P-5')
plt.legend(loc='upper right')
plt.xlabel('step')
plt.ylabel('variable value')
plt.title('Period-3 and period-5 motions before interaction start')

plt.figure()
plt.plot(x[inte_sta_ste-19:inte_sta_ste-2],y[inte_sta_ste-19:inte_sta_ste-2], 'b.')
plt.xlabel('Variable value of the intrinsic period-3 system')
plt.ylabel('Variable value of the intrinsic period-5 system')
plt.title('Phase portrait of non-interacting Period-3 and 5 systems\' motion')

plt.figure()
plt.plot(range(inte_sta_ste-19,inte_sta_ste+40), x[inte_sta_ste-19:inte_sta_ste+40], 'bo-', label='Intrinsic P-3')
plt.plot(range(inte_sta_ste-19,inte_sta_ste+40), y[inte_sta_ste-19:inte_sta_ste+40], 'ro-', label='Intrinsic P-5')
plt.legend(loc='lower right')
plt.xlabel('step')
plt.ylabel('variable value')
plt.title('Two systems\' motions if the interaction starts at step %d'%inte_sta_ste)

plt.figure()
plt.plot(range(plot_to-22,plot_to), x[plot_to-22:plot_to], 'bo-', label='Intrinsic P-3')
plt.plot(range(plot_to-22,plot_to), y[plot_to-22:plot_to], 'ro-', label='Intrinsic P-5')
plt.legend(loc='upper right')
plt.xlabel('step')
plt.ylabel('variable value')
plt.title('Two systems\' motion become synchronized Period-4')

plt.figure()
plt.plot(x[plot_from:plot_to], y[plot_from:plot_to], 'b.')
plt.xlabel('Variable value of the intrinsic period-3 system')
plt.ylabel('Variable value of the intrinsic period-5 system')
plt.title('Phase portrait of the synchronized period-4 motion')


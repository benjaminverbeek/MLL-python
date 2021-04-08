from math import sin, cos

# defining F0-6 as in eq. (6.56) "Hardonic structure functions..."
# Fäldt, Kupsc
F0 = lambda th,th1,th2,ph1,ph2 : (sin(th))**2 * sin(th1) * sin(th2) * \
            cos(ph1) * cos(ph2) + (cos(th))**2 * cos(th1) * cos(th2)

F1 = lambda th,th1,th2,ph1,ph2 : 1

F2 = lambda th,th1,th2,ph1,ph2 : 1

F3 = lambda th,th1,th2,ph1,ph2 : 1

F4 = lambda th,th1,th2,ph1,ph2 : 1

F5 = lambda th,th1,th2,ph1,ph2 : 1

F6 = lambda th,th1,th2,ph1,ph2 : 1
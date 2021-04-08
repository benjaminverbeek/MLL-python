from math import sin, cos

# defining F0-6 as in eq. (6.56) "Hardonic structure functions..."
# Fäldt, Kupsc

# Note that these functions can be called with a list of args by using
# F(*args) (* unpacks list/tuple). 
# xi = (th,th1,th2,ph1,ph2)     (angle space) 
# th - theta; ph - phi
# beta = (eta, dPhi, aL, aLb)   (decay parameters)
# eta aka. alpaha; dPhi - phase; aL - alpha-lambda; aLb - alpha-lambda-bar
# W(*beta, xi) means W can work with (eta, dPhi, aL, aLb, xi)

F0 = lambda th,th1,th2,ph1,ph2 : 1

F1 = lambda th,th1,th2,ph1,ph2 : (sin(th))**2 * sin(th1) * sin(th2) * \
            cos(ph1) * cos(ph2) + (cos(th))**2 * cos(th1) * cos(th2)

F2 = lambda th,th1,th2,ph1,ph2 : sin(th) * cos(th) * (sin(th1) * cos(th2) * \
            cos(ph1) + cos(th1) * sin(th2) * cos(ph2))

F3 = lambda th,th1,th2,ph1,ph2 : sin(th) * cos(th) * sin(th1) * sin(ph1)

F4 = lambda th,th1,th2,ph1,ph2 : sin(th) * cos(th) * sin(th2) * sin(ph2)

F5 = lambda th,th1,th2,ph1,ph2 : (cos(th))**2

F6 = lambda th,th1,th2,ph1,ph2 : cos(th1) * cos(th2) - (sin(th))**2 * \
            sin(th1) * sin(th2) * sin(ph1) * sin(ph2)
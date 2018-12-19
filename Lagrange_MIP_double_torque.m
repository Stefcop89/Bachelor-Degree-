clear
syms thr thl thdr thdl ph phd mb r l L ib iw jt g taul taur ddphi ddthetal ddthetar thetal(t) thetar(t) phi(t) phidot(t) thetadotl(t) thetadotr(t)

L = (mb*(r^2*(thdl+thdr)^2/4 + (r^2+l^2+2*l*r*cos(ph))*phd^2+(r^2+r*l*cos(ph))*(thdl+thdr)*phd))/2 + (ib*phd^2)/2 + iw*(thdr^2+thdl^2)/2+jt*(thdl^2+thdr^2-2*thdl*thdr)*r^2/(2*L^2)-mb*g*l*cos(ph);
%Calcolo dei termini delle quazioni di eulero lagrange

dph = simplify(diff(L,ph));
dthr = simplify(diff(L,thr));
dthl = simplify(diff(L,thl));
ddph = simplify(diff(L,phd));
ddthr = simplify(diff(L,thdr));
ddthl = simplify(diff(L,thdl));

ddpht = subs(ddph,[thl,thr,ph,phd,thdl,thdr],[thetal,thetar,phi,phidot,thetadotl,thetadotr]);
ddthlt = subs(ddthl,[thl,thr,ph,phd,thdl,thdr],[thetal,thetar,phi,phidot,thetadotl,thetadotr]);
ddthrt = subs(ddthr,[thl,thr,ph,phd,thdl,thdr],[thetal,thetar,phi,phidot,thetadotl,thetadotr]);

dt_dph = diff(ddpht,t);
dt_dthl = diff(ddthlt,t);
dt_dthr = diff(ddthrt,t);


vars = [diff(phi,t), diff(thetal,t),diff(thetar,t), diff(phidot,t),diff(thetadotl,t), diff(thetadotr,t)];
subst =[phd, thdl,thdr, ddphi, ddthetal,ddthetar];
dt_dph_sim1 = subs(dt_dph,vars,subst);
dt_dthl_sim1 = subs(dt_dthl,vars,subst);
dt_dthr_sim1 = subs(dt_dthr,vars,subst);

vars = [thetal, thetar, phi, phidot, thetadotl,thetadotr];
subst = [thl, thr, ph, phd, thdl,thdr];

dt_dph_sim = subs(dt_dph_sim1, vars,subst);
dt_dthl_sim = subs(dt_dthl_sim1,vars,subst);
dt_dthr_sim = subs(dt_dthr_sim1,vars,subst);

euler_ph = dt_dph_sim -dph;
euler_thl = dt_dthl_sim -dthl-taul;
euler_thr = dt_dthr_sim -dthr-taur;

Y = solve([euler_ph,euler_thl,euler_thr], [ddphi, ddthetal,ddthetar]);


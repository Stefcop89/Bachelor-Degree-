import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class MipParams(): #Struttura che contiene tutti i parametri fisici del pendolo
    def __init__(self,g=9.81,mb=0.25,ib=0.0001,iw=0.00001,l=0.1,r=0.02, jt = 0.0002, L =0.2):
        self.g=g
        self.mb=mb
        self.ib=ib
        self.iw=iw
        self.jt = jt
        self.r=r
        self.l=l
        self.L = L  

class MipEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 20
    }

    def __init__(self):
        #Imposta i limiti del dominio
        self.max_speed=10          
        self.max_torque=0.8         
        self.dt=.005            
        self.params = MipParams()   
        self.viewer = None
        self.top_viewer = None
        ffall_speed =np.finfo(np.float32).max   
        self.theta_threshold_radians = np.pi/3

        self.x_threshold = 5
        self.y_threshold = 5


        high = np.array([np.finfo(np.float32).max,np.finfo(np.float32).max,np.pi,self.max_speed,self.max_speed,ffall_speed,self.x_threshold,self.y_threshold,np.pi,self.x_threshold,self.y_threshold])
        self.action_space = spaces.Box(low=np.array([-self.max_torque,-self.max_torque]), high=np.array([self.max_torque,self.max_torque]))
        self.observation_space = spaces.Box(low=-high, high=high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        
        thl,thr,ph,dthl,dthr,phd,x,y,alpha,x_g,y_g = self.state #11 dims!
        '''
        State description
        -thl, thr: encoder angle readings, always initialized to 0
        -ph, phd angle and angular rate readings from IMU
        -dthl, dthr estimated wheel angular rates
        -x,y actual position (updated from odometry or detected)
        -alpha: relative heading wtr to x axis
        -x_g, y_g: goal position
        '''
        mb = self.params.mb;    #Body mass
        l = self.params.l;      #COM lenght
        r = self.params.r;      #wheel radius
        g = self.params.g;      #Gravity
        ib = self.params.ib;    #Body vertical inertia
        iw = self.params.iw;    #wheel effective inertia (I + m_w*r^2)
        L = self.params.L       #distance between wheels
        jt = self.params.jt     #Body transverse inertia

        dt = self.dt
        taul = np.clip(u[0], -self.max_torque, self.max_torque)
        taur = np.clip(u[1], -self.max_torque, self.max_torque)

        xg_rel = x_g-x
        yg_rel = y_g-y

        beta_g = np.arctan2(yg_rel,xg_rel)

        #Funzione j(s)
        costs = (ph)**2 + .1*phd**2 + .001*(taul**2+taur**2)+20*((x-x_g)/self.params.r)**2+20*((y-y_g)/self.params.r)**2 + 0.01*dthl**2 + 0.01*dthr**2 + (beta_g-alpha)**2 + 0.005*(dthr-dthl)**2

        #Dinamica principale

        newdthl = dthl + dt*(4*L**2*ib*iw*taul + 4*ib*jt*r**2*taul + 4*ib*jt*r**2*taur + 4*jt*mb*r**4*taul + 4*jt*mb*r**4*taur + 4*L**2*iw*l**2*mb*taul + L**2*ib*mb*r**2*taul - L**2*ib*mb*r**2*taur +
                             4*L**2*iw*mb*r**2*taul + 4*jt*l**2*mb*r**2*taul + 4*jt*l**2*mb*r**2*taur + L**2*l**2*mb**2*r**2*taul - L**2*l**2*mb**2*r**2*taur - L**2*l**2*mb**2*r**2*taul*np.cos(ph)**2
                             + L**2*l**2*mb**2*r**2*taur*np.cos(ph)**2 - 4*g*jt*l*mb**2*r**4*np.sin(ph) + 8*jt*l*mb*r**3*taul*np.cos(ph) + 8*jt*l*mb*r**3*taur*np.cos(ph) +
                             4*jt*l**3*mb**2*phd**2*r**3*np.sin(ph) + 8*L**2*iw*l*mb*r*taul*np.cos(ph) + 2*L**2*iw*l**3*mb**2*phd**2*r*np.sin(ph) - 4*g*jt*l**2*mb**2*r**3*np.cos(ph)*np.sin(ph)
                             + 4*ib*jt*l*mb*phd**2*r**3*np.sin(ph) + 4*jt*l**2*mb**2*phd**2*r**4*np.cos(ph)*np.sin(ph) - 2*L**2*g*iw*l*mb**2*r**2*np.sin(ph) +
                             2*L**2*iw*l**2*mb**2*phd**2*r**2*np.cos(ph)*np.sin(ph) - 2*L**2*g*iw*l**2*mb**2*r*np.cos(ph)*np.sin(ph) +
                             2*L**2*ib*iw*l*mb*phd**2*r*np.sin(ph))/(2*(2*L**2*ib*iw**2 + 2*L**2*iw**2*l**2*mb + 2*L**2*iw**2*mb*r**2 + 2*jt*l**2*mb**2*r**4 + 4*ib*iw*jt*r**2
                                                                        + 2*ib*jt*mb*r**4 + 4*iw*jt*mb*r**4 + L**2*ib*iw*mb*r**2 + 4*iw*jt*l**2*mb*r**2 - 2*jt*l**2*mb**2*r**4*np.cos(ph)**2 +
                                                                        L**2*iw*l**2*mb**2*r**2 - L**2*iw*l**2*mb**2*r**2*np.cos(ph)**2 + 8*iw*jt*l*mb*r**3*np.cos(ph) + 4*L**2*iw**2*l*mb*r*np.cos(ph)))

        newdthr = dthr + dt*(4*L**2*ib*iw*taur + 4*ib*jt*r**2*taul + 4*ib*jt*r**2*taur + 4*jt*mb*r**4*taul + 4*jt*mb*r**4*taur + 4*L**2*iw*l**2*mb*taur - L**2*ib*mb*r**2*taul + L**2*ib*mb*r**2*taur +
                             4*L**2*iw*mb*r**2*taur + 4*jt*l**2*mb*r**2*taul + 4*jt*l**2*mb*r**2*taur - L**2*l**2*mb**2*r**2*taul + L**2*l**2*mb**2*r**2*taur + L**2*l**2*mb**2*r**2*taul*np.cos(ph)**2
                             - L**2*l**2*mb**2*r**2*taur*np.cos(ph)**2 - 4*g*jt*l*mb**2*r**4*np.sin(ph) + 8*jt*l*mb*r**3*taul*np.cos(ph) + 8*jt*l*mb*r**3*taur*np.cos(ph) +
                             4*jt*l**3*mb**2*phd**2*r**3*np.sin(ph) + 8*L**2*iw*l*mb*r*taur*np.cos(ph) + 2*L**2*iw*l**3*mb**2*phd**2*r*np.sin(ph) - 4*g*jt*l**2*mb**2*r**3*np.cos(ph)*np.sin(ph)
                             + 4*ib*jt*l*mb*phd**2*r**3*np.sin(ph) + 4*jt*l**2*mb**2*phd**2*r**4*np.cos(ph)*np.sin(ph) - 2*L**2*g*iw*l*mb**2*r**2*np.sin(ph) +
                             2*L**2*iw*l**2*mb**2*phd**2*r**2*np.cos(ph)*np.sin(ph) - 2*L**2*g*iw*l**2*mb**2*r*np.cos(ph)*np.sin(ph)
                             + 2*L**2*ib*iw*l*mb*phd**2*r*np.sin(ph))/(2*(2*L**2*ib*iw**2 + 2*L**2*iw**2*l**2*mb + 2*L**2*iw**2*mb*r**2 + 2*jt*l**2*mb**2*r**4 + 4*ib*iw*jt*r**2
                                                                          + 2*ib*jt*mb*r**4 + 4*iw*jt*mb*r**4 + L**2*ib*iw*mb*r**2 + 4*iw*jt*l**2*mb*r**2 - 2*jt*l**2*mb**2*r**4*np.cos(ph)**2
                                                                          + L**2*iw*l**2*mb**2*r**2 - L**2*iw*l**2*mb**2*r**2*np.cos(ph)**2 + 8*iw*jt*l*mb*r**3*np.cos(ph) + 4*L**2*iw**2*l*mb*r*np.cos(ph)))
        
        newdphi = phd -dt*(mb*r**2*taul + mb*r**2*taur - g*l*mb**2*r**2*np.sin(ph) - 2*g*iw*l*mb*np.sin(ph) + l*mb*r*taul*np.cos(ph) + l*mb*r*taur*np.cos(ph) +
                           l**2*mb**2*phd**2*r**2*np.cos(ph)*np.sin(ph) - 2*iw*l*mb*phd**2*r*np.sin(ph))/(2*ib*iw + 2*iw*l**2*mb + ib*mb*r**2 + 2*iw*mb*r**2 + l**2*mb**2*r**2 - l**2*mb**2*r**2*np.cos(ph)**2
                                                                                                          + 4*iw*l*mb*r*np.cos(ph))
        newthl = thl +dt*newdthl
        newthr = thr +dt*newdthr
        
        newphi = ph +dt*newdphi
        newphi = np.arctan2(np.sin(newphi),np.cos(newphi))

        newx = x + dt*r*(dthl+dthr)*np.cos(alpha)/2
        newy = y + dt*r*(dthl+dthr)*np.sin(alpha)/2
        newalpha = alpha +dt*(dthr-dthl)*r/L
        newalpha = np.arctan2(np.sin(newalpha),np.cos(newalpha))

        #Check bordi e prossimità del goal

        done = False
        if(np.abs(newphi)>self.theta_threshold_radians):
            done = True
            costs = 100*costs
        elif(np.abs(x)>self.x_threshold or np.abs(y)>self.y_threshold):
             done = True 
        elif (np.sqrt((x-x_g)**2+(y-y_g)**2)<0.42):
            done = True
            costs = 0.01*costs
        
        self.state = np.array([newthl, newthr, newphi,newdthl, newdthr, newdphi,newx,newy,newalpha,x_g,y_g])
        return self._get_obs(), -costs, done, {}

    def reset(self):

        init_xg = self.x_threshold 
        init_yg = self.y_threshold
        init_x = 0
        init_y = 0

        init_alpha  = 0

        HIGH = [0,0,np.pi/9,self.max_speed,self.max_speed,0,init_x,init_y,init_alpha, init_xg,init_yg]
        LOW = [0,0,-np.pi/9,0,0,0,-init_x,-init_y,-init_alpha, -init_xg,-init_yg]
        self.state = self.np_random.uniform(low=LOW, high=HIGH)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state


    #Animazione dall'alto
    def render1(self, mode='human'):
        screen_width = 600
        polewidth = 10.0
        world_width = 2*self.x_threshold
        scale = screen_width/world_width

        half_base =10
        height = 40

        if self.top_viewer is None:
            from gym.envs.classic_control import rendering
            self.top_viewer = rendering.Viewer(screen_width,screen_width)
            self.curr_pos_r1 = rendering.Transform()
            
            self.goal_pos_r1 = rendering.Transform()

            self.goal_r1 = rendering.make_circle(polewidth/2)
            self.position_r1 = rendering.make_circle(polewidth/2)
            
            self.goal_r1.set_color(10,0,0)
            self.position_r1.set_color(0,10,0)

            self.goal_r1.add_attr(self.goal_pos_r1)
            self.position_r1.add_attr(self.curr_pos_r1)

            self.top_viewer.add_geom(self.goal_r1)
            self.top_viewer.add_geom(self.position_r1)

            x1,x2,x3,y1,y2,y3 = -half_base,half_base, 0, 0, 0, height

            triangle_r1  = rendering.FilledPolygon([(x1,y1), (x2,y2), (x3,y3)])
            self.triangletrans_r1 = rendering.Transform()
            triangle_r1.add_attr(self.triangletrans_r1)
            self.top_viewer.add_geom(triangle_r1)
            
        if self.state is None: return None


        x = self.state
        
        tx = x[6]*scale+screen_width/2.0
        ty = x[7]*scale+screen_width/2.0
        gx = x[9]*scale+screen_width/2.0
        gy = x[10]*scale+screen_width/2.0
        
        self.goal_pos_r1.set_translation(gx,gy)
        self.curr_pos_r1.set_translation(tx,ty)
        self.triangletrans_r1.set_translation(tx,ty)
        self.triangletrans_r1.set_rotation(x[8]+np.pi/2)
        

        return self.top_viewer.render(return_rgb_array = mode =='rgb_array')

    #Animazione laterale
    def render2(self, mode='human'):
        screen_width = 1200
        screen_height = 400
        world_width = 2*self.x_threshold
        scale = screen_width/world_width
        carty = 100 
        polewidth = 10.0
        polelen = 2*scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart  = rendering.make_circle(4*polewidth)
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None
        
        x = self.state
        cartx = screen_width/2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

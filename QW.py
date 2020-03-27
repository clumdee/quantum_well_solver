import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from ipywidgets import interactive, widgets, VBox, HTML



# Get constants
# Electron Charge
Qe    = 1.60218e-19      # C
# Mass of Electron
Me    = 9.10938e-31      # kg
# Plank's constant
H_BAR = 1.05457e-34      # J.s


    
def qw_init_box(qw):
    
    title_text = 'Please fill quantum well and calculation parameters.'
    title_html = HTML(value='<{size}>{text}</{size}>'.format(text=title_text, size='h3'))
    
    desc_w     = 'w = Width of quantum well (nm)'
    desc_E_w   = 'E_w = Lowest energy level of a quantum well (eV)'
    desc_E_b   = 'E_b = Energy level of barrier material (eV)'
    desc_Me_w  = 'Me_w = Effective electron mass in quantum well'
    desc_Me_b  = 'Me_b = Effective electron mass in barrier'
    desc_n     = 'n = Number of finite elements to break calculation into, per 1 QW width (larger n -> longer run time, default is 100)'
    desc_bound = 'bound = Thickness of barrier material to pad when perform calculation, in a QW width unit (larger bound -> longer run time, default is 6)'
    
    desc_text = f'{desc_w}<br>{desc_E_w}<br>{desc_E_b}<br>{desc_Me_w}<br>{desc_Me_b}<br>{desc_n}<br>{desc_bound}'
    desc_html = HTML(value=desc_text)
    
    qw_init   = interactive(qw.e_solver, 
                            {'manual': True, 'manual_name': 'Run State Solver'}, 
                            w='E.g. 5.60', 
                            E_w='e.g. 0', 
                            E_b='e.g. 0.23', 
                            Me_w='e.g. 0.067 (GaAs)', 
                            Me_b='e.g. 0.092 (AlGaAs)', 
                            n=widgets.IntSlider(min=30, max=500, step=10, value=100),
                            bound=widgets.IntSlider(min=3, max=10, step=1, value=6)
                           )

    return VBox((title_html, desc_html, qw_init))



class QW():
    '''
    A quantum well object.
    '''
    
    def __init__(self):
        '''
        Initialize a blank quantum well object.
        '''
        None
    
    
    
    def e_solver(self, w, E_w, E_b, Me_w, Me_b, n=100, bound=6):
        
        '''
        Set parameters for a quantum well and solve for energy states and wavefunctions.
        
        Input parameters:
        w (float): Width of a quantum well (nm)
        E_w (float): Lowest energy level of a quantum well (eV)
        E_b (float): Energy level of barrier material (eV)
        Me_w (float): (Electron mass inside quantum well) / (Free electron mass) 
        Me_b (float): (Electron mass inside barrier layer) / (Free electron mass)
        '''
        
        print('Solving for energy states and wavefunctions of the quantum well.')
        
        # convert string to float
        w    = float(w)
        E_w  = float(E_w)
        E_b  = float(E_b)
        Me_w = float(Me_w)
        Me_b = float(Me_b)
        
        # define width of quantum well and break them to finite elements
        w = w * 1e-9

        # add left and right padding of the quantum well to stabilize finite element calculation
        n_total = (2*bound + 1) * n
        x, dx   = np.linspace(-bound*w, (bound+1)*w, n_total, retstep=True)

        # get energy barrier
        delta_Ec = (E_b - E_w)        # eV
        delta_Ec = delta_Ec*Qe        # J

        # define electron mass at different locations in quantum well
        Me_w    = Me_w * Me
        Me_b    = Me_b * Me
        Me_dist = np.concatenate([np.full((1, bound*n + 1), Me_b), # at 1 to get boundary parameter
                                  np.full((1, n), Me_w), 
                                  np.full((1, bound*n + 1), Me_b)  # at 1 to get boundary parameter 
                                 ], 
                                 axis=1)

        # define bound energy level at different locations in quantum well
        V = np.concatenate([np.full((1, bound*n), delta_Ec), 
                            np.full((1, n), 0), 
                            np.full((1, bound*n), delta_Ec) 
                           ], 
                           axis=1)

        # initialize parameters of quantum well
        self.w        = w
        self.n        = n
        self.bound    = bound
        self.n_total  = n_total
        self.x        = x
        self.dx       = dx 
        self.delta_Ec = delta_Ec
        self.Me_dist  = Me_dist 
        self.V        = V
        
        # get eneygy states and wavefunctions
        self.M        = self.get_fem_matrix()
        self.e_lvl, self.w_func = self.get_energy_states()
        
        qw_desc = f'''
This quantum well is {(self.w/1e-9):.2f} nm wide.
It lies in a {self.delta_Ec / Qe} eV energy barrier.
Finite element calculation shows {len(self.e_lvl)} bounded electron states.
'''
        print('*'*20)
        print(qw_desc)

        # plot_box = interactive(self.plot, state=(0, len(self.e_lvl)-1), plot_scale=(1,10))
        
        return self #plot_box
    
    
    
    def get_fem_matrix(self):    
        '''
        Get finite element calculation matrix to solve for energy levels and wavefunctions.
        '''
        
        # get elements of FEM matrix
        main_diag = ( H_BAR**2 / (self.dx**2 * self.Me_dist[0, 1:-1]) ) + self.V
        upp_diag  = -H_BAR**2 / (2 * self.dx**2) * ( 1/self.Me_dist[0, 1:-2] 
                                                    + 1/4 * ( 1/self.Me_dist[0, 2:-1] - 1/self.Me_dist[0, 0:-3] )
                                                   )
        low_diag  = -H_BAR**2 / (2 * self.dx**2) * ( 1/self.Me_dist[0, 2:-1] 
                                                    - 1/4 * ( 1/self.Me_dist[0, 3:]   - 1/self.Me_dist[0, 1:-2] )
                                                   )

        # create FEM matrix of the quantum well
        M    = diags([main_diag, upp_diag, low_diag], [0, 1, -1], shape=(self.n_total, self.n_total))
        
        return M


    
    def get_energy_states(self):
        '''
        Solve and get energy levels and corresponding wavefunctions of an electron in a quantum well.
        '''
        
        self.get_fem_matrix()
        
        
        
        # solve for eigen values and eigen vectors
        e, w_f      = np.linalg.eig(self.M.toarray())
        
        # find valid bounded states -- energy level < bounded energy
        states      = [(e[state], w_f[:, state]) for state in range(len(e)) if e[state] < self.delta_Ec]
        
        # get energy states and corresponding wav functione
        e_lvl       = [state[0]/Qe for state in states]
        w_func      = [state[1] for state in states]

        # get rank of valid energy levels
        e_rank      = [sorted(e_lvl).index(x) for x in e_lvl]

        # re-order energy levels and corresponding wavefunctions
        e_lvl  = [e_lvl[e_rank.index(i)] for i in range(len(e_lvl))]
        w_func = [w_func[e_rank.index(i)] for i in range(len(e_lvl))]
        
        return e_lvl, w_func
    
    
    
    def plot(self, state, plot_scale):
        '''
        Plot wavefunction and corresponding electron probability inside a quantum well at a target eneygy state.
        
        Input parameters:
        level (int): Target energy state
        plot_scale (int): Scaling factor to improve visualization 
        
        
        
        ***Amplitude of wavefunction/probability is not normalized***
        '''
        
#         print(f'This quantum has {len(self.e_lvl)} bounded electron states.')
        
        
        
        # define scaling for visualization
        plot_scaling = self.delta_Ec / Qe / plot_scale

        
        
        # initialize figure object
        fig, ax      = plt.subplots(1, 1, figsize=(10,6))

        

        # plot bound condition of quantum well
        lw    = 3
        c     = 'k'
        y_low = -(self.delta_Ec / Qe)*0.2

        ax.hlines(y=self.delta_Ec / Qe, xmin=-1, xmax=0, linewidth=lw, color=c)
        ax.hlines(y=0,             xmin=0,  xmax=1, linewidth=lw, color=c)
        ax.hlines(y=self.delta_Ec / Qe, xmin=1,  xmax=2, linewidth=lw, color=c)
        ax.fill_between(range(-1, 1), self.delta_Ec / Qe, y_low, color='k', alpha=0.05)
        ax.fill_between(range(0, 2),  0,                  y_low, color='k', alpha=0.05)
        ax.fill_between(range(1, 3),  self.delta_Ec / Qe, y_low, color='k', alpha=0.05)

        ax.vlines(x=0, ymin=0, ymax=self.delta_Ec / Qe, linewidth=lw, color=c)
        ax.vlines(x=1, ymin=0, ymax=self.delta_Ec / Qe, linewidth=lw, color=c)



        # plot energy level of the selected state
        ax.hlines(y=self.e_lvl[state], xmin=-1,  xmax=2, 
                  linewidth=2, linestyle='--', color='b', 
                  label=f'Energy level = {self.e_lvl[state]:.3f} eV'
                 )



        # plot wavefunction and/or electron probability (wavefunction ** 2)
        x_plot       = self.x[int((self.bound - 1)*self.n):int((self.bound + 2)*self.n)] / self.w

        w_func_plot  = self.w_func[state][int((self.bound - 1)*self.n):int((self.bound + 2)*self.n)]
        w_func_plot  = w_func_plot / max(abs(w_func_plot)) * plot_scaling

        w_prob_plot  = w_func_plot**2
        w_prob_plot  = w_prob_plot / max(w_prob_plot) * plot_scaling

        ax.plot(x_plot, self.e_lvl[state] + w_func_plot, color='g', label=' Wavefunction')
        ax.plot(x_plot, self.e_lvl[state] + w_prob_plot, color='r', label='|Wavefunction|$\mathregular{^2}$')
        ax.fill_between(x_plot, self.e_lvl[state] + w_prob_plot, self.e_lvl[state], color='r', alpha=0.1)

        

        # set plot parameters
        ax.grid(linestyle='--', color='lightgray')
        ax.set_title(f'Wavefunction at energy state [{state}] \n (wavefunction\'s amplitude not normalized)',
                     size=24
                    )
        ax.set_xlim(-1,2)
        ax.set_ylim(y_low, (self.delta_Ec / Qe)*1.2 )
        ax.set_xlabel('Position in the quantum well scaled to width of the quantum well', size=14)
        ax.set_ylabel('Energy level relative to the bootom of the quantum well (eV)', size=14)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(loc=3)
        # ax.plot()

        
        
        return fig

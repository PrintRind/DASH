These are basic numerical simulations for the methods DASH and c-DASH as published in the paper Nam et. al.

DASH_published is the most basic simulation and available as Jupyter notebook (.ipynb) as well. 

The simulation assumes that a 2D phase scatterer is in the same optical plane as the spatial light modulator. 
The scatterer is defined by its peak-to-valley phase P2V and a parameter defining its spatial frequency: 

For example: 
P2V = 2*pi  #peak to valley phase of scatterer
sigma_scat = .1    #gaussian blur kernel sigma of scatterer
scat = create_scat(N, P2V, sigma_scat) #calculate scatterer

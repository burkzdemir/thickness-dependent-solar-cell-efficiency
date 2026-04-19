This code (soce.py) allows to calculate thickness depdendent solar cell efficiencies using real and imaginary parts of the dielectric function as the input. The file AM is the AM1.5 solar irradiance data from NREL which is used by the soce.py.
Please refer to the articles (https://doi.org/10.1016/j.solmat.2020.110557, https://doi.org/10.48550/arXiv.2407.03733) for the equations used and citing for this code.
The output solar.txt is structured as j, V, ff, ne where j is thickness, V is open-circuit voltage, ff is the fill factor, and ne is the solar cell efficiency.

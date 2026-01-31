import numpy as np

# ==============================
# Load dielectric function
# ==============================
a = np.loadtxt("a") #file a should contain the energies in eV as the first column and the imaginary part of the dielectric function as the second column and the third column should be real part of the dielectric function

# material thickness (initial, overwritten in loop)
d = (1000000) * 10**(-10)

# ==============================
# Short-circuit current parameters
# ==============================
eg = 1.46 #bang gap of the cell
ni = 0.85773 #non-ideality factor
Tcell = 300 #cell temperature
kb = 8.617333262e-5 #boltzman constant
c = 299792458 #speed of light
hp = 4.135667662e-15 #planck constant
hbar = hp / (2 * np.pi) #dirac constant

u = eg / (kb * Tcell)
io = 1/(kb) * 15 * 5.670374419e-8 / np.pi**4 * Tcell**3 #dark saturation current factor

f = np.arange(u, 100000 + 0.1, 0.1)
h = np.exp(f)
l = f**2 / (h - 1)
l = np.column_stack((f, l))

integral = 0.0
for y in range(len(l) - 1):
    integral += (l[y+1,1] + l[y,1]) / 2 * (l[y+1,0] - l[y,0])

jo = integral
io = io * jo

# ==============================
# Load solar spectrum
# ==============================
AM = np.loadtxt("AM") #solar flux measurement AM1.5 from NREL
ji = 1000.4 #total incoming solar power
nc0 = 1.0 #refractive index of the environment

# ==============================
# Optical constants
# ==============================
n = np.sqrt((np.sqrt(a[:,1]**2 + a[:,2]**2) + a[:,2]) / 2)
k = np.sqrt((np.sqrt(a[:,1]**2 + a[:,2]**2) - a[:,2]) / 2)
nc = n + 1j * k

solar = []

# ==============================
# Thickness loop
# ==============================
for j in range(10, 1000001, 10):

    d = j * 10**(-10)

    # -------- Absorbance ----------
    A = (1 - np.exp(-a[:,0]/hbar/c * a[:,1]/n * d))

    # -------- Reflectance ----------
    r1 = (-nc + nc0) / (nc + nc0)
    R1 = r1 * np.conj(r1)

    R = (r1 - r1 * np.exp(-a[:,0]/hbar/c * a[:,1]/n * d
         + 1j * a[:,0]/hbar/c * n * d * 2)) / \
        (1 - (-r1) * (-r1) * np.exp(-a[:,0]/hbar/c * a[:,1]/n * d
         + 1j * a[:,0]/hbar/c * n * d * 2))

    R = R * np.conj(R)

    # -------- Transmittance ----------
    T = (1 + r1) * (1 - r1) * np.exp(
        -a[:,0]/hbar/c * a[:,1]/n * d/2
        + 1j * a[:,0]/hbar/c * n * d) / \
        (1 - (-r1) * (-r1) * np.exp(
        -a[:,0]/hbar/c * a[:,1]/n * d
        + 1j * a[:,0]/hbar/c * n * d * 2))

    T = T * np.conj(T)

    A = 1 - T - R

    print(j)

    # ==============================
    # Solar cell efficiency
    # ==============================
    x = AM[:,0]
    coeffs = np.polyfit(a[:,0], A.real, 7)

    yfit = (coeffs[0]*x**7 + coeffs[1]*x**6 + coeffs[2]*x**5 +
            coeffs[3]*x**4 + coeffs[4]*x**3 + coeffs[5]*x**2 +
            coeffs[6]*x + coeffs[7])

    m = np.column_stack((x, yfit * AM[:,1] / AM[:,0]))

    integral = 0.0
    for y in range(len(m) - 1):
        integral += (m[y+1,1] + m[y,1]) / 2 * (m[y+1,0] - m[y,0])

    h = integral
    V = ni * kb * 300 * np.log(h/io + 1)
    v = V / (kb * 300)
    ff = (1 - np.log(v)/v) * (1 - 1/v) * (1 - np.exp(-v))**(-1)

    ne = ff * V * h / ji * 100
    print(j, ne)

    solar.append([j, V, ff, ne])

solar = np.array(solar)
np.savetxt("solar.txt", solar)

# ==============================
# Save final spectra
# ==============================
A_out = np.column_stack((a[:,0], A.real))
R_out = np.column_stack((a[:,0], R.real))
R1_out = np.column_stack((a[:,0], R1.real))
T_out = np.column_stack((a[:,0], T.real))

np.savetxt("A.txt", A_out)
np.savetxt("R.txt", R_out)
np.savetxt("R1.txt", R1_out)
np.savetxt("T.txt", T_out)

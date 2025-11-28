#------------------------------------------------------------------------------#
# This script computes the estimation error for homodyne detection             #
# in the boundary time crystal system by numerically diagonalizing             #
# the deformed master equation. It varies the system size N, the               #
# driving strength beta, and the phase phi-beta, and saves the results         #
# to CSV files for further analysis and plotting.                              #  
#------------------------------------------------------------------------------#
from qutip import *
import numpy as np
import pandas as pd
#------------------------------------------------------------------------------#
for N in [40]: # Vary N here, e.g., [10, 20, 40, 80]
    print('Currently running N = ', N)
    phi0 = 0        # initial phase difference phi-beta
    s0 = 0          # initial bias for deformed master equation
    dds = 10**(-5)  # small increment for numerical derivatives
    ddphi = dds     # small increment for numerical derivatives
    omega_c = N/2   # the critical frequency for the respective system size, 
                    # for numerics, all parameters are measured in units of 
                    # $\kappa$
    ratio0 = 0.1    # initial ration $\omega/\omega_c$ for the first point
    omega0 = ratio0 * omega_c # initial Rabi frequency
#------------------------------------------------------------------------------#
    # Collective spin operators for the sensor BTC of size N
    S_x = jmat(N/2, 'x')
    S_y = jmat(N/2, 'y')
    S_z = jmat(N/2, 'z')
    S_p = jmat(N/2, '+')
    S_m = jmat(N/2, '-')
    idenJ = qeye(S_z.shape[0])
#------------------------------------------------------------------------------#
    # Some useful functions
    def deformed_L(omega, s, phi):
        """ Returns the deformed Master operator L_phi(omega, s, phi)
        (Eq. (S27) in the Supplementary Material) for homodyne detection for the 
        respective system size N, Rabi frequency omega, bias s, and phase 
        difference phi-beta."""
        H = omega*S_x
        HT = (H.dag()).conj()
        SpSmT = ((S_p*S_m).dag()).conj()
        SpT = (S_p.dag()).conj()
        Lphi = (-1j*(tensor(idenJ, H)-tensor(HT, idenJ))+tensor(S_m.conj(), S_m) 
        -0.5*tensor(idenJ, S_p*S_m)-0.5*tensor(SpSmT, idenJ)
        - (s)*(np.exp(-1j*(phi))*tensor(idenJ, S_m) + 
                    np.exp(1j*(phi))*tensor(SpT, idenJ))
        + (s)**2/2 * tensor(idenJ, idenJ))
        return Lphi

    def dominant_eigval(L):
        """ Returns the dominant eigenvalue (with the largest real part) of
        the Liouvillian L, using sparse diagonalization with qutip."""
        evals = L.eigenenergies(sparse=True, sort='high', eigvals=1)
        return np.real(evals[0])
#------------------------------------------------------------------------------#
    # Numerical derivatives to compute the homodyne estimation error via the 
    # cumulant generating function approach (see Eq. (S28)).
    # θ(δs, φ)
    Lphi = deformed_L(omega0, s0+dds, phi0)
    lambda_P0 = dominant_eigval(Lphi)

    # θ(0, φ)
    Lphi = deformed_L(omega0, s0, phi0)
    lambda_00 = dominant_eigval(Lphi)

    # θ(-δs, φ)
    Lphi = deformed_L(omega0,-dds+s0, phi0)
    lambda_M0 = dominant_eigval(Lphi)

    # θ(δs, φ+δφ)
    Lphi = deformed_L(omega0, dds+s0, ddphi+phi0)
    lambda_PP = dominant_eigval(Lphi)

    # θ(0, φ+δφ)
    Lphi = deformed_L(omega0, s0, ddphi+phi0)
    lambda_0P = dominant_eigval(Lphi)

    # θ(0, φ-δφ)
    Lphi = deformed_L(omega0, s0, -ddphi+phi0)
    lambda_0M = dominant_eigval(Lphi)

    # θ(-δs, φ-δφ)
    Lphi = deformed_L(omega0, -dds+s0, -ddphi+phi0)
    lambda_MM = dominant_eigval(Lphi)

    deriv_ss   = (lambda_P0 - 2*lambda_00 + lambda_M0)/(dds*dds)
    deriv_sphi = (lambda_PP - lambda_P0 - lambda_0P + 2*lambda_00 
                  - lambda_M0 - lambda_0M + lambda_MM)/(2*dds*ddphi)

    EST_ERROR = np.array([[ratio0, phi0, np.sqrt((deriv_ss)/(deriv_sphi)**2), 
                           np.sqrt(deriv_ss), np.sqrt(deriv_sphi**2)]])
 
    for ratio in np.linspace(0.1, 4, 200): # Vary omega/omega_c here
        for phi in np.arange(0, np.pi, 0.125 * np.pi): # Vary phi-beta here
            omega = ratio * omega_c # Rabi frequency for the current point
            # θ(δs, φ)
            Lphi = deformed_L(omega,dds+s0, phi)
            lambda_P0 = dominant_eigval(Lphi)

            # θ(0, φ)
            Lphi = deformed_L(omega, s0, phi)
            lambda_00 = dominant_eigval(Lphi)

            # θ(-δs, φ)
            Lphi = deformed_L(omega,-dds+s0, phi)
            lambda_M0 = dominant_eigval(Lphi)

            # θ(δs, φ+δφ)
            Lphi = deformed_L(omega, dds+s0, ddphi+phi)
            lambda_PP = dominant_eigval(Lphi)

            # θ(0, φ+δφ)
            Lphi = deformed_L(omega, s0, ddphi+phi)
            lambda_0P = dominant_eigval(Lphi)

            # θ(0, φ-δφ)
            Lphi = deformed_L(omega, s0, -ddphi+phi)
            lambda_0M = dominant_eigval(Lphi)

            # θ(-δs, φ-δφ)
            Lphi = deformed_L(omega, -dds+s0, -ddphi+phi)
            lambda_MM = dominant_eigval(Lphi)

            deriv_ss   = (lambda_P0 - 2*lambda_00 + lambda_M0)/(dds*dds)
            deriv_sphi = (lambda_PP - lambda_P0 - lambda_0P + 2*lambda_00 
                          - lambda_M0 - lambda_0M + lambda_MM)/(2*dds*ddphi)

            EST_ERROR = np.append(EST_ERROR,
                                  [[ratio, phi, 
                                    np.sqrt((deriv_ss)/(deriv_sphi)**2), 
                                    np.sqrt(deriv_ss), np.sqrt(deriv_sphi**2)]],
                                    axis=0)
    df = pd.DataFrame(EST_ERROR, 
                      columns=['ratio', 'phi', 'est_err_homod', 'std', 
                               'abs_deriv'])
    filename = 'est_error_homodyne_N'+str(N)+'.csv'
    df.to_csv(path_or_buf=filename)
    print('Data saved to '+filename)
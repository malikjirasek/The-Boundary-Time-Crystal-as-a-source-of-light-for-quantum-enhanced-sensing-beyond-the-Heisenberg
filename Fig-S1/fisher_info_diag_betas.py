#------------------------------------------------------------------------------#
# This script computes the QFI for an unknown optical phase shift              #
# using the cascaded protocol described in the main text. The QFI              #
# is obtained from numerical diagonalization of the deformed Master            #
# equation. The script can be used to compute the data shwon in Fig. S1(a) and #
# (b) by adjusting the parameters accordingly. The output is a csv  #file with #
# three columns: N, omega/omega_c, qfi.                                        #
# For Fig. S1(a): Ns=[20,40,80], ratios=np.linspace(0,5,100)                   #
# For Fig. S1(b): Ns=np.arange(10,110,5), ratios=[0.5,1,4]                     #                  
#------------------------------------------------------------------------------#
import qutip as qt
import numpy as np
from scipy.sparse.linalg import eigs
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path
#------------------------------------------------------------------------------#
ratios=[0.5,1,4] # omega/omega_c values to iterate over
Ns=np.arange(10,110,5) # system sizes to iterate over
# save in the same directory as this script (fallback to cwd if running interactively)
script_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
filename = script_dir / f'QFI_figS1b.csv'
#------------------------------------------------------------------------------#
QFI = np.array([np.zeros(3)]) #initialize array to store data, each row
                                #corresponds to [N, omega/omega_c, QFI]
#------------------------------------------------------------------------------#
for N in Ns: # iterate over system sizes
    print('Currently running N = ', N)
    # system parameterss
    ddphi = 5*10**(-4) # small increment for phase shift for numerical derivatives
    omega_c = N/2 # critical frequency for given system size N
    # Collective spin operators for the sensor
    S_x = qt.jmat(N/2, 'x')
    S_y = qt.jmat(N/2, 'y')
    S_z = qt.jmat(N/2, 'z')
    S_p = qt.jmat(N/2, '+')
    S_m = qt.jmat(N/2, '-')
    idenJ = qt.qeye(S_z.shape[0])
#------------------------------------------------------------------------------#
    def deformed_L(omega, ddphi):
        """Function that builds the vectorized Lindblad Superoperator for the
        deformed Master Equation."""
        H = omega*S_x
        HT = (H.dag()).conj()
        SpSmT = ((S_p*S_m).dag()).conj()
        Lphi = (-1j*(qt.tensor(idenJ, H)-qt.tensor(HT, idenJ))
        -0.5*qt.tensor(idenJ, S_p*S_m)-0.5*qt.tensor(SpSmT, idenJ)
        +np.exp(-1j*(ddphi))*qt.tensor(S_m.conj(), S_m))
        return Lphi
#------------------------------------------------------------------------------#
    def dominant_eigval(L):
        """Function that diagonalizes a Master operator and returns its
        dominant eigenvalue."""
        evals = L.eigenenergies(sparse=True, sort='high', eigvals=1)
        return np.real(evals[0])
#------------------------------------------------------------------------------#
    for ratio in ratios: # iterate over omega/omega_c values
        omega = ratio*omega_c
        # Δφ + δφ, ω=beta*omega_c
        lambda_EP = dominant_eigval(deformed_L(omega,ddphi))
        # Δφ - δφ, ω=beta*omega_c
        lambda_EM = dominant_eigval(deformed_L(omega,-ddphi))
        QFI = np.append(QFI,[[N, ratio,-4*(lambda_EP+lambda_EM)/(ddphi*ddphi)]], 
                        axis=0)
#------------------------------------------------------------------------------#
# Save data to csv file
QFI = QFI[1:,:] #remove initial zero row
df = pd.DataFrame(QFI, columns=['N', 'ratio', 'QFI'])
df.to_csv(path_or_buf=str(filename))
print('Data saved to ', str(filename))
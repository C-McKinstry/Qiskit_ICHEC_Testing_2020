from openfermion.ops import InteractionRDM
from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
import numpy as np
from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ, BasicAer, Aer
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
backend = BasicAer.get_backend("statevector_simulator")
optimizer = SLSQP(maxiter=5)

simulated_molecule='H2O'

def round_matrix(matrix, points): #Rounds all the entries of a 2d numpy matrix to a certain number of digits
    for x in np.arange(0,len(matrix)):
        for y in np.arange(0,len(matrix[x])):
            matrix[x][y]=np.round(matrix[x][y], decimals=points)
    return matrix

def print_decimal(matrix): #Simply prints a matrix in decimals rather than scientific notation, more readable
    dec_mat=[]
    for x in np.arange(0,len(matrix)):
        dec_mat.append([])
        for y in np.arange(0,len(matrix[x])):
            if matrix[x][y] ==0:
                dec_mat[x].append('0.000000')
            else: 
                dec_mat[x].append('%f' % (matrix[x][y]))
        print(dec_mat[x])
#    print(dec_mat)

def merge_spins(matrix): #Merges the entries of spin up & down electrons in corresponding orbitals
    merged_matrix=[]
    new_size= int( len(matrix)/2 )
    for x in range(new_size):
        merged_matrix.append([])
    for x in range(new_size):
        for y in range(new_size):
            entry_xy = matrix[2*x][2*y]+matrix[2*x +1][2*y+1]
            merged_matrix[x].append(entry_xy)
    return merged_matrix
 
def generate_molecule_dict(): #Saves the molecule information  in arrays
    geometry={}
    multiplicity={}
    charge={}
    #LiH
    geometry['LiH'] = [ ['Li',          [0,             0,              0]],
                        ['H',           [0,             0,              1.45]]]
    multiplicity['LiH'] = 1
    charge['LiH'] = 0
    #H2
    geometry['H2'] = [  ['H',           [0,             0,              0]],
                        ['H',           [0,             0,              0.74]]]
    multiplicity['H2'] = 1
    charge['H2'] = 0
    #H2O
    geometry['H2O'] = [ ['O',           [5.63379480,    1.55870443,     -0.07375042]],
                        ['H',           [6.59379480,    1.55870443,     -0.07375042]],
                        ['H',           [5.31334021,    2.46364026,     -0.07375042]]]
    multiplicity['H2O'] = 1
    charge['H2O'] = 0
    return geometry, multiplicity, charge

def generate_molecule_hamiltonian(target_molecule, basis):
    geometry, multiplicity, charge=generate_molecule_dict()
    molecule = MolecularData(geometry[target_molecule], 
                             basis, 
                             multiplicity[target_molecule], 
                             charge[target_molecule])
    return molecule

def make_one_rdm(target_molecule):        
    geometry, multiplicity, charge = generate_molecule_dict() #Returns the basic info of the molecule
    molecule = generate_molecule_hamiltonian(target_molecule, 'sto-3g') #Returns MolecularData object (OpenFermion Docs p41)
    molecule = run_psi4(molecule, run_scf=False,run_mp2=False, run_cisd=True, run_ccsd=False, run_fci=False) #Calculates energies, integrals
    one_RDM= molecule.cisd_one_rdm #Returns the 1-RDM as calculated in the CISD basis
    one_RDM=merge_spins(one_RDM)
    return one_RDM

def calculate_noons(one_RDM):
    w,v=np.linalg.eig(one_RDM)#Returns the array of eigenvalues w and the corrensponding eigenvectors 
    return w

'''
print("The eigenvalues of the 1-RDM, corresponding NOON")
print_decimal([w])

#Diagonalising the matrix, diag d= p_inv*one_RDM*p
p =v
p_inv=np.linalg.inv(p)
rdm_p=np.dot(one_RDM, p)
d=np.dot(p_inv, rdm_p)

d=round_matrix(d,6)

print("The diagonalised version of the 1-RDM, which should have the eigenvalues/NOONs as entries")
print_decimal(d)'''

def generate_freeze_remove_list(eigens):
    freeze_list=[]
    remove_list=[]
    for x in range(len(eigens)):
        precission=0.0005
        freeze_bound=2.0-precission
        if eigens[x]<precission:
            remove_list.append(x)
        elif eigens[x]>freeze_bound:
            freeze_list.append(x)
    '''print("The freeze list is:")
    print(freeze_list)
    print("The remove list is:")
    print(remove_list)'''
    return freeze_list,remove_list

def geometry_convert(target_molecule): #Takes in the geometry for a molecule in OpenFermion format (arrays w/ floats), converts to form Qiskit takes (string)
    geometry, multiplicity, charge=generate_molecule_dict()
    qiskit_molecule=''
    molecule=geometry[target_molecule]
    for x in range(len(molecule)):
        qiskit_molecule += str(molecule[x][0])
        for coord in range(3):
            qiskit_molecule += " "
            qiskit_molecule += str(molecule[x][1][coord])
        if x+1 != len(molecule):
            qiskit_molecule += "; "
    return qiskit_molecule

def get_qubit_op(target_molecule):
    geometry, multiplicity, charge=generate_molecule_dict()
    driver = PySCFDriver(atom= geometry_convert(target_molecule), unit=UnitsType.ANGSTROM, charge=charge[target_molecule], spin=0, basis='sto3g')
    molecule = driver.run()
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    one_RDM=make_one_rdm(target_molecule)
    w=calculate_noons(one_RDM)
    freeze_list,remove_list = generate_freeze_remove_list(w)
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type='bravyi_kitaev', threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    shift = energy_shift + repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift

def calculate_ground_energy(target_molecule):
    qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op(target_molecule)
    initial_state = HartreeFock(num_spin_orbitals, num_particles, qubit_mapping='parity')        
    var_form = UCCSD(num_orbitals=num_spin_orbitals, num_particles=num_particles, initial_state=initial_state, qubit_mapping='parity')
    vqe = VQE(qubitOp, var_form, optimizer)
    vqe_result = np.real(vqe.run(backend)['eigenvalue'] + shift)
    return vqe_result
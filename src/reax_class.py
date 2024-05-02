# -------------------------------------------------
#
#   DipoleAnalyzer: Reax_Class
#
# -------------------------------------------------
import numpy as np


class Atom(object):

    symbol = ''
    coord = np.zeros(3)
    index = 0
    dipole_moment = np.zeros(3)
    charge = 0.0
    neighbors = []
    atomtype = 0
    bondtypes = []
    oplstype = ''
    opls_index = 0
    mass = 0
    n_neighbors = 0
    tag =''
    molid=1

    def __init__(self, index, symbol, coord, charge=0.0,
                 atype=1, mass= 0, molid=1):
        self.symbol = symbol
        self.index = index
        self.coord = np.array(coord)
        self.charge = charge
        self.atomtype = atype
        self.neighbors = []
        self.mass = mass
        self.tag = symbol # can change later
        self.molid=molid

    def print_atom_info(self):
        print(self.index, self.symbol, self.mass, self.coord)

    def neighbor_symbols(self):
        nsymbols = []
        for atom in self.neighbors:
            nsymbols.append(atom.symbol)
        return nsymbols


class Frame:
    timestep = 0

    def __init__(self, timestep, natoms=0):
        self.timestep = timestep
        self.natoms = natoms
        self.atoms = []
        self.energy = 0.0
        self.dftenergy = 0.0
        self.elements = []
        self.boxlo = np.array([0.0, 0.0, 0.0])
        self.boxhi = np.array([80.0, 80.0, 80.0])
        self.xy = 0.0
        self.xz = 0.0
        self.yz = 0.0
        self.xlo = 0.0
        self.ylo = 0.0
        self.zlo = 0.0
        self.xhi = 200.0
        self.yhi = 200.0
        self.zhi = 200.0
        self.lx = 80.0
        self.ly = 80.0
        self.lz = 80.0
        self.alpha = 90.0
        self.betha = 90.0
        self.gamma = 90.0
        self.a = 80
        self.b = 80
        self.c = 80
        self.bonds = []
        self.bondmins = []
        self.bondmaxs = []
        self.angles = []
        self.dihedrals = []
        self.anglecounts = []
        self.anglemins = []
        self.anglemaxs = []
        self.atoms = []
        self.bonds = []
        self.bondcounts = []
        self.polarization = np.zeros(3)
        self.dipole_coordinates = []
        self.dipole_moments = []

    def set_box(self, boxlo, boxhi):
        self.boxlo = np.array(boxlo)
        self.boxhi = np.array(boxhi)

    def calculate_dipole_moments(self):
        for atom in self.atoms:
            atom.dipole_moment = atom.coord * atom.charge

    def set_charges(self, charges):
        for i, charge in enumerate(charges):
            self.atoms[i].charge = float(charge)

    def calculate_total_polarization(self):
        p = np.zeros(3)
        for atom in self.atoms:
            p += atom.dipole_moment
        self.polarization = p
        del p

    def print_atoms(self):
        for atom in self.atoms:
            print (atom.symbol, atom.coord, atom.charge, atom.dipole_moment)

    def type_to_symbol(self, atype):
        symbols = ['C', 'H', 'O']
        return symbols[atype-1]

    def symbol_to_type(self, symbol):
        symbols = ['C', 'H', 'O']
        try:
            return symbols.index(symbol)
        except ValueError:
            try:
                return symbols.index(symbol.strip())
            except ValueError:
                return 0

    def add_atom(self, line, format_string):
        """
        inserts one atom with a
        string
        """
        words = line.split()
        keywords = format_string.split('_')
        molid=1
        for word, keyword in zip(words, keywords):
            if keyword == 'x':
                x = float(word)
            if keyword == 'y':
                y = float(word)
            if keyword == 'z':
                z = float(word)
            if keyword == 'q':
                q = float(word)
            if keyword == 'index':
                atomindex = int(word)
            if keyword == 'symbol':
                symbol = word
            if keyword == 'type':
                atype = int(word)
            if keyword == 'mass':
                mass = float(word)
                symbol = mass_to_symbol(mass)
            if keyword == 'molid':
                molid = int(word)
        try:
            symbol
        except NameError:
            symbol = self.type_to_symbol(atype)
        try:
            atype
        except NameError:
            atype = self.type_to_symbol(symbol)
        coord = np.array([x, y, z], dtype=float)
        atom = Atom(atomindex, symbol, coord, q, atype, mass = mass,molid=molid)
        #atom.print_atom_info()
        self.atoms.append(atom)

    def read_bond(self, line, format_string):
        """
        inserts bonding information
        """
        words = line.split()
        keywords = format_string.split('_')
        for word, keyword in zip(words, keywords):
            # if keyword == 'index':
            #    index = int(word)
            if keyword == 'btype':
                btype = int(word)
            if keyword == 'atomA':
                indexatomA = int(word)
            if keyword == 'atomB':
                indexatomB = int(word)
        #  setup bonds in the frames

        atomA = self.atoms[indexatomA-1]
        atomB = self.atoms[indexatomB-1]
        self.atoms[indexatomA-1].neighbors.append(atomB)
        self.atoms[indexatomB-1].neighbors.append(atomA)
        self.atoms[indexatomA-1].bondtypes.append(btype)
        self.atoms[indexatomB-1].bondtypes.append(btype)
        del atomA
        del atomB

    def print_dihedrals(self):
        from converter import dihedralsymbol_to_dihedraltype
        """
        Test
        """
        cnt = 0
        dihedrallines = []
        for CLatom in self.atoms:
            for Latom in CLatom.bonds:
                for CRatom in CLatom.bonds:
                    if CRatom == Latom:
                        continue
                    for Ratom in self.atoms[CRatom].bonds:
                        if Ratom == CLatom.index:
                            continue
                        symbolL = self.atoms[Latom].symbol
                        symbolCL = CLatom.symbol
                        symbolR = self.atoms[Ratom].symbol
                        symbolCR = self.atoms[CRatom].symbol
                        cnt += 1
                        line = arraytostring([cnt,
                                              dihedralsymbol_to_dihedraltype
                                              (symbolL, symbolCL,
                                               symbolCR, symbolR),
                                              Latom + 1, CLatom.index + 1,
                                              CRatom + 1, Ratom + 1], '3d')
                        dihedrallines.append(line)
        return dihedrallines


class Bond(object):
    def __init__(self, index, atomi, atomj, bond_cs=[]):
        self.index = index
        self.atomi = atomi
        self.atomj = atomj
        self.bond_cs = bond_cs
    def get_bond_pars(self):
        data =[self.bond_cs.be, self.bond_cs.bd]
        return np.asarray(data, dtype =float)
class Angle(object):
    def __init__(self,index, atomL, atomC, atomR, angle_cs=[] ):
        self.index = index
        self.angle_cs = angle_cs
        self.atomL = atomL
        self.atomC = atomC
        self.atomR = atomR
    def get_angle_pars(self):
        data = [self.angle_cs.ae, self.angle_cs.av]
        return np.asarray(data, dtype=float)

class Dihedral(object):
    def __init__(self, index, atomL,atomLC, atomRC, atomR, dihedral_cs=[]):
        self.index = index
        self.dihedral_cs = dihedral_cs
        self.atomL = atomL
        self.atomLC = atomLC
        self.atomRC = atomRC
        self.atomR = atomR
    def get_dihedral_pars(self):
        data = [self.dihedral_cs.c1, self.dihedral_cs.c2,
                self.dihedral_cs.c3, self.dihedral_cs.c4]
        return np.asarray(data, dtype=float)
class Frame_opls(Frame):
    def __init__(self, timestep=0, natoms=0, pairwisepars=[], bondpars=[], anglepars=[], dihedralpars=[]):
        Frame.__init__(self,timestep, natoms)
        self.pairwisepars = pairwisepars
        self.bondpars = bondpars
        #print self.bondpars
        self.anglepars = anglepars
        self.dihedralpars = dihedralpars
    def add_bond(self, bondindex, bondtype, atomi_index, atomj_index,):
        atomi = self.atoms[atomi_index-1]
        atomj = self.atoms[atomj_index-1]
        atomi.neighbors.append(atomj)
        atomj.neighbors.append(atomi)
        atomi.n_neighbors += 1
        atomj.n_neighbors += 1
        if len(self.bondpars):
            bond_cs = self.bondpars[bondtype-1]
            bond= Bond(bondindex, atomi, atomj, bond_cs)
        else:
            bond= Bond(bondindex, atomi, atomj)
        self.bonds.append(bond)
    def add_bond_pdb(self,atomi_index,atomj_index):
        atomi = self.atoms[atomi_index-1]
        atomj = self.atoms[atomj_index-1]
        if atomj not in atomi.neighbors:
            atomi.neighbors.append(atomj)
            atomi.n_neighbors += 1
        if atomi not in atomj.neighbors:
            atomj.neighbors.append(atomi)
            atomj.n_neighbors += 1
    def add_angle(self, angleindex, angletype, atomL_index, atomC_index,
                    atomR_index,):
        atomL = self.atoms[atomL_index-1]
        atomC = self.atoms[atomC_index-1]
        atomR = self.atoms[atomR_index-1]
        angle_cs = self.anglepars[angletype-1]
        angle = Angle(angleindex, atomL, atomC, atomR, angle_cs)
        self.angles.append(angle)
    def add_dihedral(self, dihedralindex, dihedraltype,
                     atomL_index, atomLC_index, atomRC_index, atomR_index):
        atomL = self.atoms[atomL_index-1]
        atomLC = self.atoms[atomLC_index-1]
        atomRC = self.atoms[atomRC_index-1]
        atomR = self.atoms[atomR_index-1]
        dihedral_cs = self.dihedralpars[dihedraltype-1]
        dihedral = Dihedral(dihedralindex, atomL, atomLC,
                                atomRC, atomR, dihedral_cs)
        self.dihedrals.append(dihedral)
    def print_bgf_file(self, filename):
        from arraytostring import arraytostring
        with open(filename,'w') as f:
            f.write('XTLGRF\n')
            f.write('DESCRP TEST\n')
            line = 'CRYSTX'+'  '+arraytostring(self.boxhi,'10.5f')\
                   + ' ' +arraytostring([90,90,90],'10.5f')
            f.write(line+'\n')
            line = 'FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)'
            f.write(line+'\n')
            for i,atom in enumerate(self.atoms):
                #print atom.mass
                if atom.tag == 'None':
                    line='HETATM' + ' ' +arraytostring([i+1],'5d')\
                    + ' ' + '%-5s' % mass_to_symbol(atom.mass) \
                    + ' ' + '%3s' % '' + ' ' + '%1s' % '' + '%5s' % ''\
                    + ' ' + '%10.5f%10.5f%10.5f' % (atom.coord[0],atom.coord[1],atom.coord[2]) \
                    + ' ' + '%-5s' % mass_to_symbol(atom.mass) \
                    +  '%3d%2d %8.5f' % ( 0 , 0, 0)
                else:
                    line='HETATM' + ' ' +arraytostring([i+1],'5d')\
                    + ' ' + '%-5s' % mass_to_symbol(atom.mass) \
                    + ' ' + '%3s' % '' + ' ' + '%1s' % '' + '%5s' % ''\
                    + ' ' + '%10.5f%10.5f%10.5f' % (atom.coord[0],atom.coord[1],atom.coord[2]) \
                    + ' ' + '%-5s' % atom.tag \
                    +  '%3d%2d %8.5f' % ( 0 , 0, 0)

                f.write(line+'\n')


#    def print_angles(self):
#        for ang in self.angles:
#            #print ang
def mass_to_symbol(mass):
    if abs(mass - 12.0) < 0.2:
        return "C"
    if abs(mass - 1.0) < 0.2:
        return "H"
    if abs(mass - 16.0) < 0.2:
        return "O"
    if abs(mass - 14.0 ) < 0.2:
        return "N"
    if abs(mass - 19.0) < 0.2:
        return  "F"
    if abs(mass - 28.0) < 0.2:
        return "Si"
    if abs(mass - 30.9) < 0.2:
        return "P"
    if abs(mass - 32.06) < 0.2:
        return "S"
    if abs(mass - 35.45) < 0.2:
        return "Cl"
    if abs(mass- 39.09 ) < 0.2:
        return "K"
    if abs(mass - 79.904) < 0.2:
        return "Br"
    if abs(mass - 126.904) < 0.2:
        return "I"
    return None
def symbol_to_mass(symbol):
    if symbol == 'C':
        return np.float(12.011)
    if symbol == 'H':
        return np.float(1.008)
    if symbol == 'O':
        return np.float(15.999)
    if symbol == 'N':
        return np.float(14.007)
    if symbol == 'Si':
        return np.float(28.085)
    if symbol == 'S':
        return np.float(32.060)
    if symbol == 'P':
        return np.float(30.974)
    if symbol == 'Cl':
        return np.float(35.450)
    if symbol == 'K':
        return np.float(39.09)
    if symbol == 'Br':
        return np.float(126.904)
    if symbol == 'F':
        return np.float(18.998)
def cryst_to_triclinic(lattice_constants, angles):
    """ Converts crystal format to lammps triclinic format"""
    a = lattice_constants[0]
    b = lattice_constants[1]
    c = lattice_constants[2]
    lx = a
    alpha = np.deg2rad(angles[0])
    beta = np.deg2rad(angles[1])
    gamma = np.deg2rad(angles[2])
    xy = b*np.cos(gamma)
    xz = c*np.cos(beta)
    ly = np.sqrt(b*b-xy*xy)
    yz = (b*c*np.cos(alpha)-xy*xz)/ly
    lz = np.sqrt(c*c-xz*xz-yz*yz)
    return [lx, ly, lz, xy, xz, yz]
def triclinic_to_crys(dims, tilts):
    a = dims[0]
    b = np.sqrt(dims[1]**2 + tilts[0]**2)
    c = np.sqrt(dims[2]**2 + tilts[1]**2 + tilts[2]**2)
    cosalpha = (tilts[0]*tilts[1] + dims[1] * tilts[2]) / (b*c)
    cosbeta = tilts[1] / c
    cosgamma = tilts[0] / b
    alpha = np.rad2deg(np.arccos(cosalpha))
    beta = np.rad2deg(np.arccos(cosbeta))
    gamma = np.rad2deg(np.arccos(cosgamma))
    return [a, b, c, alpha, beta, gamma]

    return None
def symboltotype(typeorder,symbol):
    symbols = typeorder.split()
    return symbols.index(symbol)+1
from collections import defaultdict

class Graph:

    # init function to declare class variables
    def __init__(self,V):
        self.V = V
        self.adj = [[] for i in range(V)]

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:

                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc


def identify_molecules(frame):
    """ Create labels specific to dcp
    """
    # identify molecules
    g = Graph(frame.natoms)
    for atom in frame.atoms:
        iatom = atom.index
        for neighbor in atom.neighbors:
            jatom = neighbor.index
            g.addEdge(iatom-1, jatom-1)
    molecules = g.connectedComponents()
    for i,molecule in enumerate(molecules):
        for iatom in molecule:
            frame.atoms[iatom].molid = i


    return frame
def dcp_oxygen_tag(frame):
    dcpmols = []
    for atom in frame.atoms:
        if atom.symbol == 'O':
            #print atom.symbol
            for n in atom.neighbors:
                if n.index < atom.index:
                    n.tag ='OL'
                    atom.tag ='OR'
            if atom.molid not in dcpmols:
                dcpmols.append(atom.molid)
    for atom in frame.atoms:
        if atom.symbol =='H':
            if atom.molid in dcpmols:
                atom.tag='H'+str(atom.molid)
    return frame

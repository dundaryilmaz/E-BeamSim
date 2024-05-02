#!/storage/work/d/duy42/Software/lammps_april2020/myenv/bin/ python -i
"""Electron Beam Simulator Driver

This is the driver for electron beam implementation.
Imports lammps as a python library via pylammps interface.
Creates a lammps instance based on the information supplied with 
settings.in file.
Here is the example settings file:
# Settings File for Ebeam Simulator
 Ebeam_Energy       80e3
 DumpFile           FBMC_beam.lammpstrj
 Coordinates        ws2slab_linedefect_zz_relaxed.lmp
 ForceField         ffield_wsoal
 Temperature        300
 NBeamIteration     720
 BeamFrequency      100
 NBeamMDSteps       2000
 RelaxMDSteps       1000
 RelaxNPTSteps      50000
 NBinsX             12
 NBinsY             12
 #FBMC               0.05 100.0 100
"""
from lammps import lammps
from random import randint
from electron_beam import *
from reax_class import symboltotype, symbol_to_mass
import os
from mpi4py import MPI
import numpy as np
#------------------------------------------------------------------------------
# Module variables
#typeorder =' C O Ti H'
global EBeamEnergy, type_orders, target_atoms
type_orders = []
target_atoms = []
EBeamEnergy=0.0
nbinx = 10
nbiny = 10
borderx = 5
bordery = 5
global beam_borderx
global beam_bordery
global cursor
global hitcount
global hitflag
hitflag = False
cursor = np.zeros(2,dtype=int)
beam_borderx = 4
beam_bordery = 4
cursor[0]= beam_borderx
cursor[1]= beam_bordery
write_out=[]
hitcount=0
boxlo = np.zeros(3)
boxhi = np.zeros(3)
#------------------------------------------------------------------------------
# Functions
def init_sim(datafile, ffield,typeorder,
             replicate='replicate 1 1 1',
             cnt_log_reax=0,dumpfile='dump.lammpstrj'):
    """Initializing lammps simulation.
       Creates a lammps object and loads coordinates of atoms.
       Loads ReaxFF forcefield to the lammps engine.
       arguments:
         datafile : file contains coordinates of atoms
         ffield: file contains force field parameters
         typeorder: element symbols for atom types in order
         If you set 1 : C 2: O 3: Ti in your data file
         then type order should be:
         "C O Ti"
         replicate: if you replicate the system in x and/or y directions
                 replicating in z irrelevant since beam runs along z axis
         cnt_log_relax: counter for log file naming - obsolute
         dumpfile: name for the trajectory file
    """
    lmp = lammps()  #  'cmdargs=["-log", logfile,"-screen" ,"none"])
    lmp.command("units           real")
    lmp.command("boundary p p f")
    lmp.command("processors * * 1")
    lmp.command("atom_style      charge")
    lmp.command("atom_modify map yes")
    lmp.command("read_data %s " % datafile)
    lmp.command(replicate)
    lmp.command("pair_style      reax/c  NULL safezone 1.8 mincap 400")
    lmp.command("pair_coeff      * * %s %s" % (ffield, typeorder))
    for i,symb in enumerate(typeorder.split()):
        dict =atom_info(i+1,typeorder)     
        line = "mass %3d %8.2f # %s " %(i+1,
              dict['Mass'], dict['Symbol'])
        lmp.command(line)
    lmp.command("neighbor        2 bin")
    lmp.command("neigh_modify    every 10 delay 0 check no")
    lmp.command("fix             fqeq all qeq/reax 1 0.0 10.0 1e-6 reax/c")
    lmp.command("compute masses all property/atom mass")
    lmp.command("compute totmass all reduce sum c_masses")
    lmp.command("variable dens equal c_totmass/vol/0.6022140857")
    lmp.command("thermo_style custom step temp epair etotal v_dens lx ly lz")
    lmp.command("thermo_modify lost ignore")
    lmp.command("timestep        0.25")
    lmp.command("variable zl equal lz*0.05")
    lmp.command("variable zh equal lz*0.65")
    lmp.command("region upbarrier block  EDGE EDGE EDGE EDGE ${zh} EDGE ")
    lmp.command("region downbarrier block  EDGE EDGE EDGE EDGE 0 ${zl} ")
    lmp.command("region lostregion union 2 upbarrier downbarrier")
    lmp.command("thermo 1000")
    lmp.command("fix mom all momentum 1 linear 0 0 1")
    boxlo , boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    boxlo = np.asarray(boxlo)
    boxhi = np.asarray(boxhi)
    return lmp, cnt_log_reax
def dump_trajectory(lmp,dumpfile,typeorder):
    """ Set up the dump file to record trajectory of atoms
        append=yes option is important to temporarily pause dumping
    """
    lmp.command("dump opls all custom 500 %s id element q x y z"%
                    dumpfile)
    lmp.command("dump_modify opls sort id element %s"% typeorder)
    lmp.command("dump_modify opls append yes")
    lmp.command('dump_modify opls format line "%5d %s  %12.8f %12.4f %12.4f %12.4f"')
    return lmp
def undump_trajectory(lmp):
    """ Stops dumping trajectory
    """
    lmp.command("undump opls")
    return lmp

def set_groups(lmp,nbinx,nbiny,beam_borderx,beam_bordery):
    """
     Set up regions and groups of atoms in the simulation where different 
     thermostats applied.
     The system is divided into grids of nbinx x nbiny. 
     There are three regions in the system:
     1- active region: Ebeam applies to this region. 
        at the center of the sample
        no thermostat applied NVE integrator was used.
     2- sink region: this region integrated with NVT 
        thermostat at set temperature in settings file
     3- freeze : atoms do not move in this region.
        at the border with a thickness of 5x5 Angstrom
        idea is to keep the sample fixed 
      Here is the diagram for regions 
      ------------------------------------------------------
     |               freeze region    5Ang                   |
     |     ---------------------------------------------     |
     |    |           Sink region: beam_borderx     b   |    |
     |    |     ----------------------------------- o   |    |
     |  5 |    |    ^                              |r   |    |
     |  A |    |    |   Active region              |d   |    |
     |  n |    |<- Lx- (nbinsx-beam_borderx)*Lx -> |e   |    |
     |  g |    |   Ly- (nbinsy-beam_bordery)*Ly    |r   |    |
     |    |    |    |                              |y   |    |
     |    |    |    v                              |    |    |
     |    |     ----------------------------------      |    |
     |    |                                             |    |
     |      --------------------------------------------     |
     |                                                       |
      -------------------------------------------------------
     arguments:
      lmp: lammps instance 
      nbinx: global variable, number of bins in x direction
      nbiny: global variable, number of bins in y direction
      beam_borderx: thickness of the sink region in x direction
      beam_bordery: thickness of the sink region in y direction
      refer to diagram above.
    returns:
      lmp: lammps instance
    """
    boxlo , boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    command = '%d %d ' %(boxlo[0]+borderx , boxhi[0]-borderx)
    command = command + '%d %d ' %(boxlo[1]+bordery , boxhi[1]-bordery) 
    command = command + 'EDGE EDGE side out'
    lmp.command("region freeze block " + command )
    lmp.command("group freeze region freeze ")
    lmp.command("fix fr freeze setforce 0 0 0")
    dx  =float(beam_borderx)/float(nbinx)*(boxhi[0]-boxlo[0])
    dy  =float(beam_bordery)/float(nbiny)*(boxhi[1]-boxlo[1])
    xlo = boxlo[0] + dx
    xhi = boxhi[0] - dx
    ylo = boxlo[1] + dy
    yhi = boxhi[1] - dy
    command = 'region active block %f8.2 %f8.2  %f8.2 %f8.2 EDGE EDGE side in' %(xlo, xhi, ylo, yhi)
    lmp.command(command)
    lmp.command("group active region active")
    command = 'region sink block %f8.2 %f8.2 %f8.2 %f8.2 EDGE EDGE side out' %(xlo, xhi, ylo, yhi)
    lmp.command(command)
    lmp.command("group sink region sink")
    lmp.command("group sink subtract sink freeze")
    return lmp

def reset_border(lmp):
    """
      resets border
      for a full relaxation of the system
      returns: lammps instance
    """
    lmp.command('unfix fr')
    lmp.command('group active delete')
    lmp.command('group freeze delete')
    return lmp

def run_sim_gun(lmp, gunfrequency=100,
                tempi=300, tempf=300,damp=50.0,
                 steps=100, current_step=0):
    """runs simulations with ebeam implementation 
       ebeam implemented via post-force callback feature of the lammps library
       after calculating forces based on the force field description, before
       doing the verlet integration, MD engine calls a python function. 
       In that python function velocities added to atoms which hit by e-beam.
       lmp : lammps instance
       gunfrequency: the frequency to call ebeam function 
       tempi: initial temperature for sink region thermostat
       tempf: final temperature for sink region thermostat
         note: tempi and tempf are usually equal since we do not need change in
         sink temperature during ebeam.
       damp: damping constant for thermostat of the sink region
       steps: number of MD steps.
       current_step: keep track of MD steps for records.
       returns: current_step which is input current step + steps
    """
    lmp.command("timestep        0.1")
    lmp.command("reset_timestep %d" % current_step)
    lmp.command("fix blnc all balance 1000 1.1 shift xy 20 1.1 out tmp.balance")
    lmp.command(
        'fix run_sim sink nvt temp %f %f %f ' % ( tempi, tempf, damp))
    lmp.command(
        'fix run_sim_nve active nve' )
    lmp.command('fix pf  all python/invoke %d post_force post_force_callback'
               % gunfrequency)
    lmp.command('run %d  post no ' % steps)
    lmp.command('unfix run_sim')
    lmp.command('unfix run_sim_nve')
    lmp.command('unfix pf')
    lmp.command('unfix blnc')
    lmp.command("timestep        0.25")
    current_step = lmp.get_thermo("step")
    return lmp, current_step

def run_sim_relax(lmp, simtype, tempi=100,group='all',
                  tempf=100, steps=1, current_step=0,dumpfile='dump.lammpstrj'):
    """runs simulation to relax the whole system  
       lmp : lammps instance, simtype : nvt or nve , temp
    """
    lmp.command("reset_timestep %d" % current_step)
    lmp.command("fix blnc all balance 5000 1.1 shift xy 20 1.1 out tmp.balance")
    if simtype == 'nvt':
        lmp.command(
        'fix run_sim %s %s temp %f %f 100.0 ' % (group,simtype, tempi, tempf))
        lmp.command('run %d ' % steps)
    if simtype == 'npt':
        lmp.command(
        'fix run_sim all %s temp %f %f '% (simtype, tempi, tempf) +
            ' 100.0 x 0 0 1000 y 0 0 1000 fixedpoint 0 0 0' )
        lmp.command('run %d post no' % steps)

    lmp.command('unfix run_sim')
    lmp.command('unfix blnc')
    current_step = lmp.get_thermo("step")
    return lmp, current_step

def run_sim_nve(lmp,  tempi=100, tempf=100,
            steps=1, current_step=0,
            damp=50.0,group='all'):
    """lmp : lammps instance, simtype : nvt or nve , temp
    """
    lmp.command("reset_timestep %d" % current_step)
    lmp.command("fix blnc all balance 500 0.9 shift xy 20 1.1 out tmp.balance")
    lmp.command(
        'fix run_sim_nve active nve' )
    lmp.command(
        'fix run_sim sink nve')
    lmp.command('run %d post no' % steps)
    lmp.command('unfix run_sim')
    lmp.command('unfix run_sim_nve')
    lmp.command('unfix blnc')
    current_step = lmp.get_thermo("step")
    return lmp, current_step
def run_sim(lmp,  tempi=100, tempf=100,
            steps=1, current_step=0,
            damp=50.0,group='all'):
    """lmp : lammps instance, simtype : nvt or nve , temp
    """
    lmp.command("reset_timestep %d" % current_step)
    lmp.command("fix blnc all balance 500 0.9 shift xy 20 1.1 out tmp.balance")
    lmp.command(
        'fix run_sim_nve active nve' )
    lmp.command(
        'fix run_sim sink nvt temp %f %f %f ' % ( tempi, tempf, damp))
    lmp.command('run %d post no' % steps)
    lmp.command('unfix run_sim')
    lmp.command('unfix run_sim_nve')
    lmp.command('unfix blnc')
    current_step = lmp.get_thermo("step")
    return lmp, current_step
def minimize_system(
      lmp, maxstepi=1000, maxstepg=10000, convi=10e-8, convg=10e-8):
    lmp.command('fix minimize_system all nve')
    lmp.command('minimize %f %f %d %d' % (convi, convg, maxstepi, maxstepg))
    lmp.command('unfix minimize_system')
    return lmp
def run_sim_fbmc(lmp, simtype='tfmc', delta=0.1, temp=300,
                 steps=1, current_step=0,dumpfile='dump.lammpstrj'):
    """ lmp : lammps instance, simtype : tfbmc , temp
    """
    lmp.command("thermo 100")
    lmp.command("reset_timestep %d" % current_step)
    lmp.command(
        'fix run_sim_nvt sink nvt temp %f %f %f ' % ( temp, temp, 100.0))
    lmp.command(
        'fix run_sim active %s %f  %f 95302 ' % (simtype, delta, temp ))
    lmp.command('run %d post no' % steps)
    lmp.command('unfix run_sim')
    lmp.command('unfix run_sim_nvt')
    lmp.command("thermo 1000")
    current_step = lmp.get_thermo("step")
    return lmp, current_step
def post_force_callback(lammps_ptr, vflag):
    """post-force call back function
       lammps instance calls this function after calculating forces.
       this function is the driver for ebeam implementation.
       First access the atom info via lammps/python interface
       Selects an atom to shoot with an electron.
       Simulation box divided into a grid in xy plane.
       Cursor travels on this grid, at each step an atom which falls into
       the cursor location randomly selected to shoot with ebeam. 
       this function receives a pointer as an argument 
       so it has access to all data that lammps instance generates.
    """
    global hitflag
    global hitcount
    global threshold_energies
    pid = os.getpid()
    L = lammps(ptr=lammps_ptr)
    me = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if me == 0:
        ranks = np.zeros(size)
        print(
        '---------CALL BACK FUNCTION------------------------------------------')
    #--------------------------------------------------------------------------
    # Accessing MD data, atom coordinates, types etc
    t = L.extract_global("ntimestep", 0)
    nlocal = L.extract_global("nlocal", 0)
    nghost = L.extract_global("nghost", 0)
    ntypes = L.extract_global("ntypes", 0)
    mass = L.numpy.extract_atom_darray("mass", ntypes+1)
    atype = L.numpy.extract_atom_iarray("type", nlocal)
    x = L.numpy.extract_atom_darray("x", nlocal, dim=3)
    v = L.numpy.extract_atom_darray("v", nlocal+nghost, dim=3)
    f = L.numpy.extract_atom_darray("f", nlocal+nghost, dim=3)
    mylist = L.find_pair_neighlist("reax/c", request=0)
    xlo = L.extract_global('boxxlo')
    ylo = L.extract_global('boxylo')
    zlo = L.extract_global('boxzlo')
    xhi = L.extract_global('boxxhi')
    yhi = L.extract_global('boxyhi')
    zhi = L.extract_global('boxzhi')

    picked_atom, xright, xleft, yright, yleft = 0,0.0,0.0,0.0,0.0
    #--------------------------------------------------------------------------
    # note that this code runs in parallel via MPI.
    # this part executed by all ranks. each rank try to pick an atom within the 
    # cursor. Since Lammps already spatially distributed atoms along the ranks
    # only one or two - depending on the number of cores you are using- ranks 
    # will be able to find an atom to pick
    # if a rank picks an atom, it will assign it to the picked_atom variable
    # we scatter an empty array of 1xN_ranks an each rank, will put the index 
    # of the atom if it finds one. Then we gather back these arrays via 
    # comm.gather on rank=0
    #--------------------------------------------------------------------------
    picked_atom, xright, xleft, yright, yleft = pick_atom(x,atype,
                                xlo,xhi,ylo,yhi,nbinx,nbiny,cursor)
    #create empty arrays
    new_velocity = np.zeros(3)
    final_velocity = np.zeros(3)
    # broadcast empty array to nodes
    if me == 0:
        data = np.zeros(size)
    else:
        data = None
    comm.scatter(data,root=0)
    # gather picked atoms from nodes into array data
    data = picked_atom
    newdata = comm.gather(data,root=0)
    rn = None
    if me == 0:
        FLAG = all( v is None for v in newdata)
        if FLAG is False :
            # at rank=0 we check the picked atoms array "newdata" if it was not
            # an empty array. then we select one randomly.
            rn = np.random.choice([i for i in range(len(newdata)) if newdata[i] != None])
        else:
            rn = -1
    #--------------------------------------------------------------------------
    # after selection process the rank=0 will broadcast the selected atom to 
    # selected rank. 
    selected_rank = comm.bcast(rn,root=0)
    # here selected rank will process the electron beam effect
    if selected_rank == me:
        idx = picked_atom
        coord = x[idx]
        atominfo = atom_info(atype[idx,0],typeorder)
        print(idx,coord,atominfo)
        # selected rank will call the "shoot_atom" function from the 
        # electron_beam module and gets the velocity added to 
        # selected atom as a result of collision with electron
        new_velocity = shoot_atom(atominfo,v[idx,:],EBeamEnergy)
        # the atom shot by electron beam gets its updated velocity
        v[idx] = v[idx] + new_velocity
        final_velocity = v[idx]+new_velocity
        # if selected rank is not 0 then send picked atom info to node 0
    # this part is just book keeping
    # we wanted to keep a record of coordinates of atoms shot by electron
    # here we send this info to rank=0 to write a file
    if selected_rank > -1:
        if selected_rank != 0 and me == selected_rank:
            comm.send((idx,x[idx],atominfo,
                      xright,xleft,yright,yleft,new_velocity,final_velocity),dest=0)
        if selected_rank != 0 and me == 0:
            idx,coord,atominfo,xright,xleft,\
                yright,yleft,new_velocity,final_velocity = comm.recv()

        if me == 0:
            hitcount +=1
            string = ('%4d %4s %4d %4d %8.4f %8.4f %8.4f ' %
                      (hitcount, atominfo['Symbol'], cursor[0], cursor[1],coord[0],coord[1],coord[2])) #+\
            string = string + ('%8.4f %8.4f %8.4f %8.4f '%
                       (xright,xleft,yright,yleft))
            string = string + ('%8.4f %8.4f %8.4f %8.5f %8.5f %8.5f\n'%
                               (new_velocity[0],new_velocity[1],new_velocity[2],
                               final_velocity[0],final_velocity[1],final_velocity[2]))
            if np.dot(new_velocity,new_velocity) > 0:
                hitflag = True
                
                #print('Hitflag ',True)
            write_out.append(string)
    hitflag = comm.bcast(hitflag,root=0)
    cursor[0] +=1
    # here we update the cursor position to scan the sample
    if cursor[0] == nbinx-beam_borderx:
        cursor[1] += 1
        cursor[0] = beam_borderx
        if cursor[1] == nbiny-beam_bordery:
            cursor[1] = beam_bordery
    if me == 0:
        if np.dot(new_velocity,new_velocity) > 0:
            print(
        '---------CALL BACK FUNCTION : SUCCESSFULL HIT------------------------')
        else:
            print(
        '---------CALL BACK FUNCTION : UNSUCCESSFULL HIT----------------------')
    return L
def pick_atom(x,atom_type,
                 xlo,xhi,ylo,yhi,
                 nxbins,nybins,cursor):
    global target_atoms,type_orders
    #print(target_atoms)
    lx = (xhi-borderx)-(xlo+borderx)
    ly = (yhi-bordery)-(ylo+bordery)
    deltax = lx/nxbins
    deltay = ly/nybins
    xright = cursor[0]*deltax+xlo+borderx
    xleft = xright +deltax
    yright = cursor[1]*deltay+ylo+bordery
    yleft = yright +deltay
    xc = x[:,0]
    yc = x[:,1]
    zc = x[:,2]
    types= atom_type[:,0]
    symbols = [ type_orders[t-1] for t in types]
    # here we select atoms inside the cursor location
    # here we are only selecting C and O atoms
    idx = np.argwhere( (xleft >xc) & (xc >= xright)&
                      (yleft >yc) & (yc >= yright) )
    idx = idx.reshape(idx.size)
    idx = [ i for i in idx if symbols[i] in target_atoms ]
    idx = np.asarray(idx)

    # if there is no atom inside the cursor return none 
    if idx.size == 0:
        return None,None,None,None,None
    # since electron beam effects the atoms at the top side of the sample
    # - we are shooting electrons at z=0 to z =inf -------
    # we only pick the atoms with higher z coordinate
    m  =np.mean(zc[idx])
    idxp = np.argwhere(zc[idx] >= m)
    idx = idx[idxp].reshape(idxp.shape[0])
    idx = np.random.choice(idx)
    return idx ,xright,xleft,yright,yleft

def atom_info(atom_type,typeorder):
    types = typeorder.split(' ')
    symbol = types[atom_type-1]
  
    atom_info ={
        "W":{"Symbol":"W", "Name":"Tungsten",
             "Z":74, 'Mass':183.84,"Threshold_Energy":45},
        "Mo":{"Symbol":"W", "Name":"Molybdenium",
             "Z":42, 'Mass':95.94,"Threshold_Energy":45},
        "S":{"Symbol":"S","Name":"Sulfur",
             "Z":16,"Mass":32.06,"Threshold_Energy":7},
        "Al":{"Symbol":"Al","Name":"Alumunium",
             "Z":13,"Mass":26.982,"Threshold_Energy":15},
        "H":{"Symbol":"H","Name":"Hydrogen",
             "Z":1,"Mass":1,"Threshold_Energy":1},
        "C":{"Symbol":"C","Name":"Carbon",
             "Z":6,"Mass":12,"Threshold_Energy":12},
        "O":{"Symbol":"O","Name":"Oxygen",
             "Z":8,"Mass":16,"Threshold_Energy":6},
        "Ti":{"Symbol":"Ti","Name":"Titanium",
             "Z":22,"Mass":47,"Threshold_Energy":12}
        }
    try:
        dic = atom_info[symbol]
        dic['Threshold_Energy'] = threshold_energies[atom_type-1]
        return dic
    except:
        return atom_info
    # return dicatom_info[symbol]
def main():
    global beam_borderx
    global beam_bordery
    global EBeamEnergy
    global write_out, hitcount,hitflag
    global nbinx,nbiny,cursor,target_atoms,type_orders,typeorder
    global threshold_energies
    comm = MPI.COMM_WORLD
    f = open('electrondata.txt','w')
    #         1234 1234 1234 1234 12345678 12345678 12345678 12345678 12345678
    header = '# hc symb   cx   cy        x        y        z      mxr      mxl'
    #                   12345678 12345678 12345678 12345678 12345678 12345678 12345678 12345678
    header = header + '      myr      myl       vx       vy       vz      fvx      fvy      fvz\n'
    f.write(header)
    f.close()
    ffield = 'ffield_wsoal'
    #typeorder =' C O Ti'
    datafile = 'ws2slab.lmp'
    EBeamEnergy=80e3
    FBMC_FLAG = False
    RelaxNPTSteps = None
    settings=''
    with open('settings.in') as settings:
        lines = settings.readlines()
        settings=lines
        for line in lines:
            words=line.split()
            #print(words[0],'--->',words[1])
            if words[0] == 'Ebeam_Energy':
                EBeamEnergy = float(words[1])
                #print('Ebeam Energy is set to : %8.2g' %EBeamEnergy)
            if words[0] == 'Coordinates':
                datafile = words[1]
                #print('Coordinates will be read from %s '% datafile)
            if words[0] == 'ForceField':
                ffield = words[1]
            if words[0] == 'TypeOrder':
                type_orders= words[1:]
                typeorder = ' '.join(type_orders)
            if words[0] == 'TargetAtoms':
                target_atoms= words[1:]
            if words[0] == 'ThresholdEnergies':
                threshold_energies=np.asarray( words[1:],dtype=float)

            if words[0] == 'DumpFile':
                dumpfile = words[1]
                f = open (dumpfile,'w')
                f.close()
            if words[0] == 'FBMC':
                FBMC_FLAG = True
                fbmc_delta = float(words[1])
                fbmc_temperature = float(words[2])
                fbmc_steps = int(words[3])
            if words[0] == 'NBeamIteration':
                NBeamIteration=int(words[1])
            if words[0] == 'Temperature':
                Temperature = float(words[1])
            if words[0] =='BeamFrequency':
                BeamFrequency = int(words[1])
            if words[0] =='NBeamMDSteps':
                NBeamMDSteps = int(words[1])
            if words[0] =='RelaxMDSteps':
                RelaxMDSteps = int(words[1])
            if words[0] =='RelaxNPTSteps':
                RelaxNPTSteps = int(words[1])
            if words[0] =='BeamBorderx': 
                beam_borderx = int(words[1])
            if words[0] =='BeamBordery': 
                beam_bordery = int(words[1])
            if words[0] =='NBinsX':
                nbinx = int(words[1])
            if words[0] =='NBinsY':
                nbiny = int(words[1])
    me = MPI.COMM_WORLD.Get_rank()
    cursor[0] = beam_borderx
    cursor[1] = beam_bordery
    if me == 0:
        for line in settings:
            print(line.strip('\n'))
    lmp , cnt_log_reax= init_sim(datafile, ffield, typeorder)
    lmp =set_groups(lmp,nbinx,nbiny,beam_borderx,beam_bordery) 
    #lmp.command('fix bl all box/relax x 0.0 y 0.0 vmax 0.001 fixedpoint 0.0 0.0 0.0')
    #lmp = minimize_system(lmp,maxstepi=100)
    #lmp.command('unfix bl')
    #lmp = minimize_system(lmp,maxstepi=100)

    #lmp =set_border(lmp) 
    current_step = 0
    if RelaxNPTSteps:
        lmp = dump_trajectory(lmp,'intialize.lammpstrj',typeorder)
        lmp.command('velocity active create %8.2f %d  ' % (Temperature,randint(1,100000)))
        lmp, current_step = run_sim_relax( lmp, 'npt',group='active',steps=RelaxNPTSteps/4,
                                    tempi=Temperature
                                ,tempf=Temperature,
                                   current_step=current_step,
                                   dumpfile=dumpfile)
        current_temperature = lmp.get_thermo("temp")
        if me == 0: print('Current temperature after NPT HALF :%f ' %current_temperature)
        lmp = minimize_system(lmp,maxstepi=1000)
        lmp.command('velocity active create %8.2f %d  ' % (Temperature/2,randint(1,100000)))
        lmp.command("reset_timestep %d" % current_step)
        lmp, current_step = run_sim_relax( lmp, 'npt',group='all',steps=RelaxNPTSteps/4,
                                    tempi=Temperature
                                ,tempf=Temperature,
                                current_step=current_step,
                                   dumpfile=dumpfile)
        current_temperature = lmp.get_thermo("temp")
        if me == 0: print('Current temperature after NPT 2 HALF :%f ' %current_temperature)
        lmp = minimize_system(lmp,maxstepi=100)
        lmp.command('velocity active create %8.2f %d  ' % (Temperature,randint(1,100000)))
        lmp.command("reset_timestep %d" % current_step)
        lmp, current_step = run_sim_relax( lmp, 'nvt',group='all',steps=RelaxNPTSteps/2,
                                tempi=Temperature
                               ,tempf=Temperature,
                                current_step=current_step,
                               dumpfile=dumpfile)
        current_temperature = lmp.get_thermo("temp")
        if me == 0: print('Current temperature after NVT :%f ' %current_temperature)
        lmp.command('write_restart relaxed.restart')
        lmp.command('write_dump all xyz relaxed.xyz modify sort id element '+typeorder)
        lmp.command('velocity active create %8.2f %d  ' % (Temperature,randint(1,100000)))
        lmp =undump_trajectory(lmp)
    lmp = dump_trajectory(lmp,dumpfile,typeorder)
    current_step = 0
    cnthit=0
    for i in range(NBeamIteration):
        if me == 0 :
            print(" =================== ITERATION %d of %d ================" % 
                  (i,NBeamIteration))
        hitsperscan = (nbinx - 2*beam_borderx) * (nbiny - 2*beam_bordery)
        if me ==0:
            print('Hits per scan ', hitsperscan)
        write_out = []
        for j in range(hitsperscan):
            cnthit +=1
            if me == 0:
                print(
                " ===================Hit:%d===of:%d hits per scan====%d=hits=="%
                        (j+1,hitsperscan,cnthit))
            lmp, current_step = run_sim_gun( lmp,
                                steps=BeamFrequency,
                                gunfrequency=BeamFrequency,
                                tempi=Temperature,
                                tempf=Temperature,
                                current_step=current_step,
                                )
            lmp.command("timestep        0.1")
            lmp, current_step = run_sim( lmp,
                                steps=(NBeamMDSteps-BeamFrequency),
                                tempi=Temperature,
                                tempf=Temperature,
                                current_step=current_step)
            lmp.command("timestep        0.25")
            if cnthit % 5 == 0 and cnthit > 0  :
                if me == 0:
                    print(
                " ===================Relaxing The System : hitflag %d========="%hitflag )

                #lmp, current_step = run_sim( lmp, steps=RelaxMDSteps/2,
                #                  tempi=Temperature,
                #                 tempf=Temperature,
                #             current_step=current_step)
                lmp, current_step = run_sim_nve ( lmp, steps=RelaxMDSteps/2,
                                  tempi=Temperature,
                                 tempf=Temperature,
                             current_step=current_step)
                if FBMC_FLAG:
                    if me == 0:
                        print(
                    " MC Simulation for %6d steps" % (fbmc_steps))
                    lmp, current_step = run_sim_fbmc( lmp, 'tfmc',delta=fbmc_delta,
                                temp=fbmc_temperature,
                                steps=fbmc_steps,
                                current_step=current_step)
                if me == 0:
                    print(
                " MD Simulation for %6d steps and low damping" % (RelaxMDSteps/4))
                lmp, current_step = run_sim( lmp,steps=RelaxMDSteps/4,
                               tempi=Temperature,
                               tempf=300,damp=100.0,
                               current_step=current_step)
                if me == 0:
                    print(
                " MD Simulation for %6d steps and regular damping" % (RelaxMDSteps/4))
                lmp, current_step = run_sim( lmp,steps=RelaxMDSteps/4,
                                      tempi=300,
                                      tempf=Temperature,
                                      current_step=current_step)
                if hitflag:
                    MDSTEPS= RelaxMDSteps
                    if me == 0:
                        print(
                    " Extra MD Simulation for %6d steps and low damping" % MDSTEPS)
                    lmp, current_step = run_sim( lmp,steps=MDSTEPS,
                               tempi=Temperature,
                               tempf=Temperature,damp=100.0,
                               current_step=current_step)
                    if me == 0:
                        print(
                    " ExtraE MD Simulation for %6d steps and regular damping" % MDSTEPS)
                    lmp, current_step = run_sim( lmp,steps=MDSTEPS,
                                      tempi=Temperature,
                                      tempf=Temperature,
                                      current_step=current_step)

                hitflag = False
        if me == 0:
           f = open('electrondata.txt','a')
           for line in write_out:
               f.write(line)
           f.close()
        lmp.command('write_restart RESTART_ITERATIO%d.restart'%i)

if __name__ == '__main__':
    main()

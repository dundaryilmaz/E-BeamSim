#!/storage/work/d/duy42/Software/lammps_april2020/myenv/bin/ python -i
from lammps import lammps
from random import randint
from electron_beam import *
from reax_class import symboltotype, symbol_to_mass
import os
from mpi4py import MPI
import numpy as np
typeorder =' W S O Al H'
cursor = np.zeros(2,dtype=int)
cursor[0]= 1
cursor[1]= 1
global EBeamEnergy
EBeamEnergy=0.0
nbinx = 10
nbiny = 10
write_out=[]

def init_sim(datafile, ffield, typeorder, 
             replicate='replicate 1 1 1',
	     cnt_log_reax=0, cmdoption="cpu"):

    if cmdoption == "kokkos":
        cmdargs=["-k","on", "t", 
                 "7","g","6", "-pk", "kokkos", 
                 "gpu/direct","off","-sf", "kk",
                 ]
    if cmdoption =="cpu":
            cmdargs=[]
    lmp = lammps(cmdargs= cmdargs)

#def init_sim(datafile, ffield,typeorder,
#             replicate='replicate 1 1 1',
#             cnt_log_reax=0):
    lmp = lammps()  #  'cmdargs=["-log", logfile,"-screen" ,"none"])
    lmp.command("units           real")
    lmp.command("boundary p p f")
    lmp.command("processors * * 1")
    lmp.command("atom_style      charge")
    lmp.command("atom_modify map yes")
    lmp.command("read_data %s " % datafile)
    lmp.command(replicate)
    lmp.command("pair_style      reax/c  NULL")
    lmp.command("pair_coeff      * * %s %s" % (ffield, typeorder))
    lmp.command("mass 1 95.94 # W ")
    lmp.command("mass 2 32.06 # S ")
    lmp.command("mass 3 15.99 # O ")
    lmp.command("mass 4 26.98 # Al")
    lmp.command("mass 5 1.0 #H ") 
    lmp.command("neighbor        2 bin")
    lmp.command("neigh_modify    every 10 delay 0 check no")
    lmp.command("fix             fqeq all qeq/reax 1 0.0 10.0 1e-6 reax/c")
    lmp.command("compute masses all property/atom mass")
    lmp.command("compute totmass all reduce sum c_masses")
    lmp.command("variable dens equal c_totmass/vol/0.6022140857")
    lmp.command("thermo_style custom step temp epair etotal v_dens lx ly lz")
    lmp.command("thermo_modify lost ignore")
    lmp.command("timestep        0.25")
    bxlo = float(1.0)/float(nbinx)
    bxhi = float(nbinx -1)/float(nbinx)
    bylo = float(1.0)/float(nbiny)
    byhi = float(nbiny -1)/float(nbiny)
    lmp.command("variable xl equal lx*%4.3f"%bxlo)
    lmp.command("variable xh equal lx*%4.3f"%bxhi)
    lmp.command("variable yl equal ly*%4.3f"%bylo)
    lmp.command("variable yh equal ly*%4.3f"%byhi)
    lmp.command("variable zl equal lz*0.1")
    lmp.command("variable zh equal lz*0.9")
    lmp.command("region active block  ${xl} ${xh} ${yl} ${yh} ${zl} ${zh} ")
    lmp.command("group active region active")
    lmp.command("group freeze subtract all active")
    lmp.command("fix fr freeze setforce NULL NULL 0")
    lmp.command("thermo 100")
    lmp.command("dump opls all custom 100 dump.lammpstrj id element q x y z")
    lmp.command("dump_modify opls sort id element %s"% typeorder)
    lmp.command("dump_modify opls append yes")
    lmp.command('dump_modify opls format line "%5d %s  %12.8f %12.4f %12.4f %12.4f"')
    lmp.command("undump opls")
    lmp.command("fix mom all momentum 1 linear 0 0 1")
    return lmp, cnt_log_reax
def run_sim_gun(lmp, simtype, tempi=100, tempf=100, gunfrequency=100,
                steps=100, current_step=0):
    """ lmp : lammps instance, simtype : nvt or nve , temp
    """
    lmp.command("reset_timestep %d" % current_step)
    lmp.command("dump opls all custom 100 dump.lammpstrj id element q x y z")
    lmp.command("dump_modify opls sort id element %s"% typeorder)
    lmp.command("dump_modify opls append yes")
    lmp.command('dump_modify opls format line "%5d %s  %12.8f %12.4f %12.4f %12.4f"')
    lmp.command("fix blnc all balance 500 0.9 shift xy 20 1.1 out tmp.balance")
    lmp.command(
        'fix run_sim all %s temp %f %f 100.0 ' % (simtype, tempi, tempf))
    lmp.command('fix pf  all python/invoke %d post_force post_force_callback'
               % gunfrequency)
    lmp.command('velocity active create %8.2f %d  ' % (tempi,randint(1,100000)))
    lmp.command('run %d ' % steps)
    lmp.command('unfix run_sim')
    lmp.command('unfix pf')
    lmp.command('unfix blnc')
    lmp.command("undump opls")
    current_step = lmp.get_thermo("step")
    return lmp, current_step
def run_sim_relax(lmp, simtype, tempi=100, tempf=100, steps=1, current_step=0):
    """ lmp : lammps instance, simtype : nvt or nve , temp
    """
    lmp.command("reset_timestep %d" % current_step)
    lmp.command("dump opls all custom 100 dump.lammpstrj id element q x y z")
    lmp.command("dump_modify opls sort id element %s"% typeorder)
    lmp.command('dump_modify opls format line "%5d %s  %12.8f %12.4f %12.4f %12.4f"')
    lmp.command("dump_modify opls append yes")
    lmp.command("fix blnc all balance 500 0.9 shift xy 20 1.1 out tmp.balance")
    if simtype == 'nvt':
        lmp.command(
        'fix run_sim all %s temp %f %f 100.0 ' % (simtype, tempi, tempf))
    if simtype == 'npt':
        lmp.command(
        'fix run_sim all %s temp %f %f 100.0 x 0 0 1000 y 0 0 1000 ' % (simtype, tempi, tempf))

    lmp.command('run %d ' % steps)
    lmp.command('unfix run_sim')
    lmp.command('unfix blnc')
    lmp.command("undump opls")
    current_step = lmp.get_thermo("step")
    return lmp, current_step
def run_sim(lmp, simtype, tempi=100, tempf=100, steps=1, current_step=0):
    """ lmp : lammps instance, simtype : nvt or nve , temp
    """
    lmp.command("reset_timestep %d" % current_step)
    lmp.command("dump opls all custom 100 dump.lammpstrj id element q x y z")
    lmp.command("dump_modify opls sort id element %s"% typeorder)
    lmp.command('dump_modify opls format line "%5d %s  %12.8f %12.4f %12.4f %12.4f"')
    lmp.command("dump_modify opls append yes")
    lmp.command("fix blnc all balance 500 0.9 shift xy 20 1.1 out tmp.balance")
    if simtype == 'nvt':
        lmp.command(
        'fix run_sim all %s temp %f %f 100.0 ' % (simtype, tempi, tempf))
    if simtype == 'npt':
        lmp.command(
        'fix run_sim all %s temp %f %f 100.0 x 0 0 1000 y 0 0 1000 ' % (simtype, tempi, tempf))

    lmp.command('velocity active create %8.2f %d  ' % (tempi,randint(1,100000)))
    lmp.command('run %d ' % steps)
    lmp.command('velocity active create %8.2f %d  ' % (tempf,randint(1,100000)))
    lmp.command('unfix run_sim')
    lmp.command('unfix blnc')
    lmp.command("undump opls")
    current_step = lmp.get_thermo("step")
    return lmp, current_step
def minimize_system(
      lmp, maxstepi=1000, maxstepg=10000, convi=10e-8, convg=10e-8):
    lmp.command('fix minimize_system all nve')
    lmp.command('minimize %f %f %d %d' % (convi, convg, maxstepi, maxstepg))
    lmp.command('unfix minimize_system')
    return lmp
def run_sim_fbmc(lmp, simtype='tfmc', delta=0.1, temp=300, steps=1, current_step=0):
    """ lmp : lammps instance, simtype : tfbmc , temp
    """
    lmp.command("reset_timestep %d" % current_step)
    lmp.command("dump opls all custom 100 dump.lammpstrj id element q x y z")
    lmp.command("dump_modify opls sort id element %s"% typeorder)
    lmp.command('dump_modify opls format line "%5d %s  %12.8f %12.4f %12.4f %12.4f"')
    lmp.command("dump_modify opls append yes")
    lmp.command(
        'fix run_sim active %s %f  %f 95302 ' % (simtype, delta, temp ))
    lmp.command('run %d ' % steps)
    lmp.command('velocity active create %8.2f 3234234  ' % temp)
    lmp.command('unfix run_sim')
    lmp.command("undump opls")
    current_step = lmp.get_thermo("step")
    return lmp, current_step
def post_force_callback(lammps_ptr, vflag):
    pid = os.getpid()
    L = lammps(ptr=lammps_ptr)
    me = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if me == 0:
        ranks = np.zeros(size)
        print('-----------------CALLBACK-----------------------')

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
    xlo = L.extract_global('boxxlo',type=1)
    ylo = L.extract_global('boxxlo',type=1)
    zlo = L.extract_global('boxxlo',type=1)
    xhi = L.extract_global('boxxhi',type=1)
    yhi = L.extract_global('boxyhi',type=1)
    zhi = L.extract_global('boxzhi',type=1)
    Xmesh,Ymesh,Zmesh = get_mesh(xlo,xhi,ylo,yhi,zlo,zhi,
                                 x[:,0],x[:,1],x[:,2])
    picked_atom, xright, xleft, yright, yleft = 0,0.0,0.0,0.0,0.0
    picked_atom, xright, xleft, yright, yleft = pick_atom_v2(x,atype,
                                xlo,xhi,ylo,yhi,nbinx,nbiny,cursor)
    new_velocity = np.zeros(3)
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
        #print('DATA: ',newdata)
        FLAG = all( v is None for v in newdata)
        if FLAG is False :
            rn = np.random.choice([i for i in range(len(newdata)) if newdata[i] != None])
        else:
            rn = -1
        print('Selected Node is %d '%rn)
    selected_rank = comm.bcast(rn,root=0)
    if selected_rank == me:
        idx = picked_atom
        coord = x[idx]
        #margins=np.array([xright,xleft,yright,yleft])
        atominfo = atom_info(atype[idx,0],typeorder)
        #print('I am selected rank ', me)
        #print('Coordinates of picked atom: %8.4f %8.4f %8.4f \n'%
        #       (coord[0],coord[1],coord[2]))
        # update velocity of selected atom
        new_velocity = shoot_atom(atominfo,v[idx,:],EBeamEnergy)
        v[idx] = v[idx] + new_velocity
        # if selected rank is not 0 then send picked atom info to node 0
    if selected_rank > -1:
        if selected_rank != 0 and me == selected_rank:
            comm.send((idx,x[idx],atominfo,
                      xright,xleft,yright,yleft,new_velocity),dest=0)
        if selected_rank != 0 and me == 0:
            idx,coord,atominfo,xright,xleft,\
                yright,yleft,new_velocity = comm.recv()

        if me == 0:
            print( 'Index of selected atom : %d and coordinates %8.2f %8.2f %8.2f'
              % (idx, coord[0],coord[1],coord[2]))
            string = ('%4d %4d %8.4f %8.4f %8.4f ' %
                      (cursor[0], cursor[1],coord[0],coord[1],coord[2])) #+\
            string = string + ('%8.4f %8.4f %8.4f %8.4f '%
                       (xright,xleft,yright,yleft))
            string = string + ('%8.4f %8.4f %8.4f \n'%
                               (new_velocity[0],new_velocity[1],new_velocity[2]))
            write_out.append(string)
    cursor[0] +=1
    if cursor[0] == nbinx-1:
        cursor[1] += 1
        cursor[0] = 1
        if cursor[1] == nbiny-1:
            cursor[1] = 1
    return L
def pick_atom_v2(x,atom_type,
                 xlo,xhi,ylo,yhi,
                 nxbins,nybins,cursor):
    lx = xhi-xlo
    ly = yhi-ylo
    deltax = lx/nxbins
    deltay = ly/nybins
    xright = cursor[0]*deltax+xlo
    xleft = xright +deltax
    yright = cursor[1]*deltay
    yleft = yright +deltay
    xc = x[:,0]
    yc = x[:,1]
    zc = x[:,2]
    #print(xright,xleft)
    #print(yright,yleft)
    idx = np.argwhere((atom_type[:,0] == 2) & 
                      (xleft >xc) & (xc >= xright)&
                      (yleft >yc) & (yc >= yright))
    if idx.size == 0:
        return None,None,None,None,None
    idx = idx.reshape(idx.size)
    #print(idx.shape,idx.size)
    #print('Selected indexes', idx)
    m  =np.mean(zc[idx])
    #print('Mean : ', m)
    idxp = np.argwhere(zc[idx] >= m)
    #print('idxP',idxp,idx[idxp])
    idx = idx[idxp].reshape(idxp.shape[0])
    #print('idx' ,idx)
    idx = np.random.choice(idx)
    #print('Select',idx)
    return idx ,xright,xleft,yright,yleft


def pick_atom(Xmesh,Ymesh,Zmesh,Types, xb,yb,zb,atom_type):
    idx = np.argwhere((Xmesh == xb) & (Ymesh == yb) & (Zmesh == zb) & (Types[:,0] == 2))
    print('idx', idx.shape)
    try:
        picked_atom =np.random.choice(idx[:,0])
        return picked_atom
    except:
        return None
def get_mesh(xlo,xhi,ylo,yhi,zlo,zhi,X,Y,Z):
    xbins=np.arange(1,nbinx) 
    xbins = np.linspace(xlo, xhi, num=nbinx)
    ybins = np.linspace(ylo, yhi, num=nbiny)
    #for xb,yb in zip(xbins,ybins):
    #    print('Bins : ',xb,yb)
    zbins = np.linspace(zlo, zhi, num=3)
    xmesh = np.digitize(X,xbins, right=True)
    ymesh = np.digitize(Y,ybins, right=True)
    zmesh = np.digitize(Z,ybins)
    return xmesh,ymesh,zmesh

def atom_info(atom_type,typeorder):
    types = typeorder.split(' ')
    symbol = types[atom_type]
    atom_info ={
        "W":{"Symbol":"W", "Name":"Tungsten",
             "Z":74, 'Mass':183.84,"Threshold_Energy":45},
        "S":{"Symbol":"S","Name":"Sulfur",
             "Z":16,"Mass":32.06,"Threshold_Energy":4},
        "1Al":{"Symbol":"Al","Name":"Alumunium",
             "Z":13,"Mass":26.982,"Threshold_Energy":15},
        "H":{"Symbol":"H","Name":"Hydrogen",
             "Z":1,"Mass":1,"Threshold_Energy":100}
        }
    return atom_info[symbol]
def main():
    global EBeamEnergy
    global write_out
    global nbinx,nbiny
    comm = MPI.COMM_WORLD
    f = open('test.txt','w')
    #         1234 1234 12345678 12345678 12345678 12345678 12345678
    header = '# cx   cy        x        y        z      mxr      mxl'
    #                   12345678 12345678 12345678 12345678 12345678
    header = header + '      myr      myl       vx       vy       vz\n'
    f.write(header)
    f.close()
    f = open ('dump.lammpstrj','w')
    f.close()
    ffield = 'ffield_wsoal'
    typeorder =' W S O Al H'
    datafile = 'ws2slab.lmp'
    EBeamEnergy=80e3
    FBMC_FLAG = False
    RelaxNPTSteps = None
    RunType = 'CPU'
    with open('settings.in') as settings:
        lines = settings.readlines()
        for line in lines:
            words=line.split()
            print(words[0],'--->',words[1])
            if words[0] == 'RunType':
                RunType = words[1]
            if words[0] == 'Ebeam_Energy':
                EBeamEnergy = float(words[1])
                print('Ebeam Energy is set to : %8.2g' %EBeamEnergy)
            if words[0] == 'Coordinates':
                datafile = words[1]
                print('Coordinates will be read from %s '% datafile)
            if words[0] == 'ForceField':
                ffield = words[1]
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
            if words[0] =='NBinsX':
                nbinx = int(words[1])+2
            if words[0] =='NBinsX':
                nbinx = int(words[1])+2
    if RunType == 'KOKKOS':
    	lmp , cnt_log_reax= init_sim(datafile, ffield, typeorder,
				cmdoption='kokkos')
    else:
    	lmp , cnt_log_reax= init_sim(datafile, ffield, typeorder,
				cmdoption='cpu')
	

    current_step = 0
    me = MPI.COMM_WORLD.Get_rank()
    if RelaxNPTSteps:
    #file = open ('test.txt','w')
        lmp, current_step = run_sim( lmp, 'npt',steps=RelaxNPTSteps,
                                    tempi=Temperature
                                ,tempf=Temperature,
                                current_step=current_step)
    for i in range(NBeamIteration):
        if me == 0 :
            print(" =================== ITERATION %d of %d ================" % 
                  (i,NBeamIteration))
            write_out = []
        lmp, current_step = run_sim_gun( lmp, 'nvt',steps=NBeamMDSteps,
                                        gunfrequency=BeamFrequency,
                                        tempi=Temperature,
                                        tempf=Temperature,
                                        current_step=current_step)
        if me == 0:
            f = open('test.txt','a')
            for line in write_out:
                f.write(line)
            f.close()
        lmp, current_step = run_sim( lmp, 'npt',steps=RelaxMDSteps/2,
                                    tempi=Temperature,
                                    tempf=Temperature,
                                current_step=current_step)
        lmp, current_step = run_sim( lmp, 'nvt',steps=RelaxMDSteps/2,
                                    tempi=Temperature,
                                    tempf=Temperature,
                                current_step=current_step)
        if FBMC_FLAG:
            lmp, current_step = run_sim_fbmc( lmp, 'tfmc',delta=fbmc_delta,
                                    temp=fbmc_temperature,
                                    steps=fbmc_steps,
                                current_step=current_step)
            lmp = minimize_system(lmp)

if __name__ == '__main__':
    main()
    

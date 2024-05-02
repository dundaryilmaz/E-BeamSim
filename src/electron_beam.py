# Electron:
import numpy as np
emass_to_pmass = 1836.152673
erest_mass = 0.510998e6 # eV
hbarc = 0.1973269804e-6 # ev m
fineconst =  0.0072973525693 # 1/137
evtoJ = 1.602e-19
amasstoKg = 1.660e-27
def beta_ev(electron_energy):
    """electron_energy : electron energy in units of eV
    """
    return np.sqrt(1.0-1.0/(1+electron_energy/erest_mass)**2)

def t_max(electron_energy,Mass_of_Atom):
    """ Calculates  maximum transferred energy to a target atom
    from incident electron.
    Equation 6 of  " Nanoscale, 2013, 5, 6677
    Arguments:
        electron_energy : Energy of the incident electron in eV
        Mass_of_atom : Mass of the target atom in atomic units
    Returns:
        Maximum transferred enegy in eV, scalar.
    """
    part1 = 2*electron_energy*(electron_energy+2*erest_mass)
    part2 = erest_mass*(Mass_of_Atom*emass_to_pmass+2+1/(Mass_of_Atom*emass_to_pmass))\
            +2*electron_energy
    return part1/part2

def transferred_energy(electron_energy,Mass_of_Atom,theta):
    """ Calculates the energy transferred to target atom from electron scattered with an
    angle theta
    Equation 7 of  " Nanoscale, 2013, 5, 6677
    Arguments:
        electron_energy: Energy of the incident electron in eV
        Mass_of_Atom: Mass of the target atom in atomic units
        theta: Scattering angle of the electron in radians
    Returns:
        transferred energy in eV, scalar.
    """
    return t_max(electron_energy, Mass_of_Atom)*np.sin(theta/2.0)**2

def rutherford_xsection(Z,electron_energy,theta):
    """ Rutherford scattering cross section for relativistic electrons
    Arguments:
        Z: Nuclear charge of the target atom, in atomic units
        electron_energy: Energy of the incident electron in eV
        theta: Scattering angle of the electron
    Returns:
        Scattering cross section of the electron. Scalar.
    """
    beta = beta_ev(electron_energy)
    part1 = (Z*fineconst*hbarc/(2*erest_mass))**2
    part2 = (1-beta**2)/(beta*np.sin(theta/2))**4
    return part1*part2

def coupling(Z,electron_energy,theta):
    """ The coupling function between Rutherford cross section to McKinley Feschbach cross section.
    Arguments:
        Z : Nuclear charge of the target atom
        electron_energy: Energy of the incident electron in eV
        theta: Scattering angle of the electron
    Returns:
        Scalar Value of the coupling function
    """
    beta = beta_ev(electron_energy)
    return 1- beta**2*np.sin(theta/2)**2+np.pi*Z*fineconst*beta*np.sin(theta/2)\
            *(1-np.sin(theta/2))

def scatter_xsection(Z,electron_energy,theta):
    """
    Scattering cross section of an electron scttared with an angle theta
    Basically multiplies rutherford_xsection and coupling functions defined above
    Arguments:
        Z : Nuclear charge of the target atom
        electron_energy: Energy of the incident electron in eV
        theta: Scattering angle of the electron
    Returns:
        Scalar Value of the scattering cross sectionn
    """
    return rutherford_xsection(Z,electron_energy,theta)*\
            coupling(Z,electron_energy,theta)

def theta_min(Tmax, Tmin):
    """ Minimum scattering angle
    If the maximum transferred energy to the target atom from incident electron is greater than
    minimum threshold energy to displace the target atom, then we can calculate scattering cross 
    section. Otherwise we ignore and we wont calculate scattering cross section.
    Thus this function returns 0 for Tmax < Tmin
    Arguments:
        Tmax : Maximum transferred energy to the target atom from incident electron in eV
        Tmin: Minimum displacement energy of the target atom in eV
    Returns:
        theta_min : Minimum scattering angle if Tmax > Tmin , else 0
    """
    c=np.where(Tmax>Tmin ,np.sqrt(Tmin/Tmax), 0)
    return 2*np.arcsin(c)


def xsection_theta_min_to_theta(Mass,Z,Threshold_Energy,electron_energy,theta):
    Tmax = t_max(electron_energy,Mass)
    Threshold_min = Threshold_Energy
    Theta_min = theta_min(Tmax,Threshold_min)
    thetas = np.linspace(Theta_min,theta,num=20)
    c = np.sin(thetas)*scatter_xsection(Z,electron_energy,thetas)
    d = np.where(Theta_min > 0,2*np.pi*np.trapz(c,thetas),0)
    return d

def total_emission_xsection(Mass,Z,Threshold_Energy,electron_energy):
    beta = beta_ev(electron_energy)
    gamma = np.pi*4*(Z*fineconst*hbarc/(2*erest_mass))**2*(1-beta**2)/beta**4
    Tmax= t_max(electron_energy,Mass)
    Tmin = Threshold_Energy
    q = Tmax/Tmin
    q = np.where(Tmax/Tmin > 1.0 , q,1)
    coupling =  q - 1 - beta**2*np.log(q) + np.pi*Z*fineconst*beta*(2*np.sqrt(q)-np.log(q)-2)
    return gamma*coupling

def integrate(Z,Mass,Threshold_Energy,electron_energy):
    """ Calculates the numerical integral of scattering cross section from theta_min to Pi to find
    total cross section
    Arguments :
        Z : Nuclear charge of the target atom
        Mass: Mass of the target atom -atomic units-
        Electron_energy: energy of the incident electron - in eV-
    Returns:
        total_xsection: Scalar value for total cross section
    Conditions:
        Normally there should be a condition for minimum threshold energy but this function only
        called from scattering_probability function and these conditions were implemented there.
    """
    Threshold = Threshold_Energy
    tmax = t_max(electron_energy=electron_energy,Mass_of_Atom=Mass)
    thetamin = np.where(tmax > Threshold, theta_min(tmax,Threshold), np.pi)
    thetas = np.linspace(thetamin,np.pi,num=100)
    total_xsection=np.trapz(scatter_xsection(Z,electron_energy=electron_energy,theta=thetas)*\
             np.sin(thetas)*2*np.pi,thetas)
    return total_xsection

def evofe_to_beta(electron_energy):
    """ electron_energy : electron's kinetic energy 
    in units of kilo electron volts
    returns v/c ratio
    """
    return( np.sqrt(1.0-1.0/(1.0+electron_energy/0.510998950e3)**2))
	
def scattering_probability(Z, Mass,Threshold_Energy, electron_energy):
    """
    Calculates probability distribution function for scattering angle of the incident electron.
    Arguments :
        Z : Nuclear charge of the target atom
        Mass: Mass of the target atom -atomic units-
        Electron_energy: energy of the incident electron - in eV-
    Returns:
        thetas: linear array of size 100 for scattering angle values between theta_min to Pi
        dist_func:  Probability distribution for corresponding angles , array of size 100
        prbability_function : Cumulative probability distribution function for scattering angle
    Conditions:
        If Maximum Transferred energy is lower than Threshold energy then returns array of zeros
    """
    trf_max = t_max(Mass_of_Atom=Mass,electron_energy=electron_energy)
    thrs = Threshold_Energy
    nsamples = 100000
    if  trf_max < thrs :
        return np.zeros(nsamples),np.zeros(nsamples),np.zeros(nsamples), trf_max
    else:
        thetamin = theta_min(Tmax=trf_max,Tmin=thrs)
        total_xsection = integrate(Z=Z,Mass=Mass,Threshold_Energy=Threshold_Energy,
                                   electron_energy=electron_energy)
        thetas ,dtheta = np.linspace(thetamin,np.pi,num=nsamples,retstep=True)
        dist_func = scatter_xsection(Z,electron_energy,thetas)*2.0*np.pi*np.sin(thetas)/total_xsection
        probability_function = np.cumsum(dist_func)*dtheta
        return thetas,dist_func,probability_function,trf_max

def shoot_atom(atom_info,velocity,electron_energy):
    import time
    """
      This is the entrance point of the electron_beam Module
      arguments:
        atom_info : information of selected atom to be shot with electron
        velocity : current implementation does not require initial velocity
                   of the atom however this passed just to display purposes
        electron_energy: energy of the incident electron
      returns:
        velocity to be added to the atom hit by electron
    """
    from random import choices
    print('---------------------------------------------------------------------')
    print('---------------------------------------------------------------------')
    print('---------------------------------------------------------------------')
    print('----------  Electron Beam Module-------------------------------------')
    print('Target Atom                = %s' % atom_info['Name'])
    print('Initial Velocity           = %8.4e %8.4e %8.4e A/fs'%(velocity[0],velocity[1],velocity[2])  )
    print('Incident Electron Energy   = %8.3f KeV' %(electron_energy*1e-3))
    thetas, dist_func, prob_func, trf_max = scattering_probability(
        atom_info["Z"],atom_info["Mass"],atom_info["Threshold_Energy"],
        electron_energy)
    print('Maximum Transferred Energy = %8.4f eV'%trf_max)
    theta_selected = choices(population=thetas,weights=dist_func)[0]
    rn = np.random.random()
    idx =(np.abs(prob_func-rn).argmin())
    print('Theta min is               = %8.4f '% thetas[0])
    #print(prob_func[idx],thetas[idx])
    Phi = np.random.uniform(0,2*np.pi)
    #Omega = (np.pi-thetas[idx])/2
    Omega = (np.pi-theta_selected)/2
    Transferred_Energy = trf_max * np.sin(theta_selected/2.0)**2
    print('Electron Scattering Angle  = %4.2f degrees '%(theta_selected*180.0/np.pi))
    print('Atom Emission Angle        = %4.2f degrees '%(Omega*180.0/np.pi))
    print('Azimuthal Angle of Atom    = %4.2f degrees '%(Phi*180.0/np.pi))
    print('Transferred Energy         = %8.4e eV' % Transferred_Energy)
    XEJ = Transferred_Energy*evtoJ
    v =np.sqrt(2*XEJ/atom_info["Mass"]/amasstoKg)/1e5 # A/fs
    #Omega = 2*np.pi
    vz =v*np.cos(Omega)
    vx = v*np.sin(Omega)*np.cos(Phi)
    vy = v*np.sin(Omega)*np.sin(Phi)
    print('Added Velocity             = %8.4e %8.4e %8.4e A/fs'%(vx,vy,vz))
    print('---------------------------------------------------------------------')
    print('---------------------------------------------------------------------')
    print('---------------------------------------------------------------------')
    #time.sleep(10)
    new_velocity = np.asarray([vx,vy,vz])

    return new_velocity

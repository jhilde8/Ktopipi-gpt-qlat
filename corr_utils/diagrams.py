import numpy as np
import qlat as q

#----
# qlattice correlator load -> numpy array
#----

#function that loads in qlattice correlator data
def load_corr(dst,ext,traj_list,v=False):
    ld_corr = []
    
    
    for traj in traj_list:
        try:
            ld = q.load_lat_data(dst + f"/traj-{traj}/" + ext)
            if len(ld_corr) == 0 and v == True:
                print(ld.info()[0][2]) #expression and index information
            ld_corr.append(ld.to_numpy())
        except:
            print(f"path: {dst} \nextension: {ext} \ntrajectory: {traj} \ndoes not exist!\n")
            

    corr_arr = np.array(ld_corr)
    print(f"array shape: {corr_arr.shape}")
        
    return corr_arr


#function that averages two correlators of the same type.
def corr_point_avg(corr1, corr2):
    #assuming corr1 and corr2 have the exact same shape, (ncf, nexpr, tsep)
    avg_corr = np.zeros_like(corr1)

    #we can simply use element wise operations here since these are numpy arrays
    avg_corr = (corr1 + corr2)/2
    
    return avg_corr

# ----
# pion scattering diagram utils
# ----

#sum over spatial separations, project to momentum space
def wave_function_mode_000(xrel,yrel,zrel,NS):
    w = np.ones_like(xrel)
    return w

#constructs the Fourier component for the (0,0,0) momentum config. 
#inputs are relative coordinates. 
def wave_function_mode_001_f(xrel,yrel,zrel,NS):
    w1 = np.exp(-1j*(2.0 * np.pi * xrel / NS))
    #w2 = np.exp(-1j*(2.0 * np.pi * yrel / NS))
    #w3 = np.exp(-1j*(2.0 * np.pi * zrel / NS))
    #w = (w1 + w2 + w3)/3.0
    #w1 = np.exp(-1j * (2.0 * np.pi * xrel / NS))
    return w1

def wave_function_mode_001_b(xrel,yrel,zrel,NS):
    w1 = np.exp(1j*(2.0 * np.pi * xrel / NS))
    #w2 = np.exp(1j*(2.0 * np.pi * yrel / NS))
    #w3 = np.exp(1j*(2.0 * np.pi * zrel / NS))
    #w = (w1 + w2 + w3)/3.0
    #w1 = np.exp(1j * (2.0 * np.pi * xrel / NS))
    return w1

def wave_function_mode_011(xrel,yrel,zrel,NS):
    w1 = np.cos((2.0 * np.pi * xrel / NS) + (2.0 * np.pi * yrel / NS))
    w2 = np.cos((2.0 * np.pi * yrel / NS) + (2.0 * np.pi * zrel / NS))
    w3 = np.cos((2.0 * np.pi * xrel / NS) + (2.0 * np.pi * zrel / NS))
    w4 = np.cos((2.0 * np.pi * xrel / NS) - (2.0 * np.pi * yrel / NS))
    w5 = np.cos((2.0 * np.pi * yrel / NS) - (2.0 * np.pi * zrel / NS))
    w6 = np.cos((2.0 * np.pi * xrel / NS) - (2.0 * np.pi * zrel / NS))
    
    
    w = (w1 + w2 + w3 + w4 + w5 + w6)/6.0
    return w

def wave_function_mode_011_f(xrel,yrel,zrel,NS):
    w1 = np.exp(-1j* ((2.0 * np.pi * xrel / NS) + (2.0 * np.pi * yrel / NS)))
    
    #w = (w1 + w2 + w3 + w4 + w5 + w6)/6.0
    return w1

def wave_function_mode_011_b(xrel,yrel,zrel,NS):
    w1 = np.exp(1j* ((2.0 * np.pi * xrel / NS) + (2.0 * np.pi * yrel / NS)))
    
    #w = (w1 + w2 + w3 + w4 + w5 + w6)/6.0
    return w1

def wave_function_mode_111(xrel,yrel,zrel,NS):
    w1 = np.cos((2.0 * np.pi * xrel / NS) + (2.0 * np.pi * yrel / NS) + (2.0 * np.pi * zrel / NS))
    w2 = np.cos(-(2.0 * np.pi * xrel / NS) + (2.0 * np.pi * yrel / NS) + (2.0 * np.pi * zrel / NS))
    w3 = np.cos((2.0 * np.pi * xrel / NS) - (2.0 * np.pi * yrel / NS) + (2.0 * np.pi * zrel / NS))
    w4 = np.cos((2.0 * np.pi * xrel / NS) + (2.0 * np.pi * yrel / NS) - (2.0 * np.pi * zrel / NS))
    
    w = (w1 + w2 + w3 + w4)/4.0
    return w

def wave_function_mode_111_f(xrel,yrel,zrel,NS):
    w1 = np.exp(-1j*((2.0 * np.pi * xrel / NS) + (2.0 * np.pi * yrel / NS) + (2.0 * np.pi * zrel / NS)))
    
    #w = (w1 + w2 + w3 + w4)/4.0
    return w1

def wave_function_mode_111_b(xrel,yrel,zrel,NS):
    w1 = np.exp(1j*((2.0 * np.pi * xrel / NS) + (2.0 * np.pi * yrel / NS) + (2.0 * np.pi * zrel / NS)))
    
    #w = (w1 + w2 + w3 + w4)/4.0
    return w1

#pos_corr will have shape (ncf, NS, NS, NS, NT)
def mom_project(pos_corr):
    Ncf,nexpr,Nx,Ny,Nz,Nt = pos_corr.shape #extent in each direction
    x,y,z = np.meshgrid(np.arange(Nx),np.arange(Ny),np.arange(Nz))

    ph_000 = wave_function_mode_000(x,y,z,Nx)
    ph_001_f = wave_function_mode_001_f(x,y,z,Nx)
    ph_001_b = wave_function_mode_001_b(x,y,z,Nx)
    ph_011_f = wave_function_mode_011_f(x,y,z,Nx)
    ph_011_b = wave_function_mode_011_b(x,y,z,Nx)
    ph_111_f = wave_function_mode_111_f(x,y,z,Nx)
    ph_111_b = wave_function_mode_111_b(x,y,z,Nx)
    
    nexpr = 8
    mom_corr = np.zeros((Ncf,nexpr,Nt), np.complex128)

    for cf in range(Ncf):
    #sum over spatial points for each timeslice
        for t in range(Nt):
            c_arr = pos_corr[cf,0,:,:,:,t] #<1>
            pos_arr = pos_corr[cf,1,:,:,:,t]

            mom_corr[cf,0,t] = np.sum((ph_000 * c_arr), axis=(0,1,2)) #counter
            mom_corr[cf,1,t] = np.sum((ph_000 * pos_arr),axis=(0,1,2))
            mom_corr[cf,2,t] = np.sum((ph_001_f * pos_arr),axis=(0,1,2))
            mom_corr[cf,3,t] = np.sum((ph_001_b * pos_arr),axis=(0,1,2))
            mom_corr[cf,4,t] = np.sum((ph_011_f * pos_arr),axis=(0,1,2))
            mom_corr[cf,5,t] = np.sum((ph_011_b * pos_arr),axis=(0,1,2))
            mom_corr[cf,6,t] = np.sum((ph_111_f * pos_arr),axis=(0,1,2))
            mom_corr[cf,7,t] = np.sum((ph_111_b * pos_arr),axis=(0,1,2))
            

    return mom_corr


#from corr data saved in terms of tsrc and tsnk, this creates a corr in terms of tsep. 
def construct_tsep(corr,tsep_max):
    ncf = corr.shape[0]
    nexpr = corr.shape[1]
    corr_red = np.zeros((ncf, nexpr, tsep_max),dtype=np.complex128)
    for tsep in range(tsep_max):
        for tsrc in range(tsep_max):
            t1 = tsrc
            t2 = (tsrc + tsep) % tsep_max

            corr_red[:,:,tsep] += corr[:,:,t1,t2]

    return corr_red

#constructs the vacuum diagram from two two pion vacuum expectation values
def construct_V(bubble_raw_1,bubble_raw_2,tsep_max,Delta):
    ncf = bubble_raw_1.shape[0]
    nt = bubble_raw_1.shape[-1]
    vac = np.zeros((ncf, tsep_max),dtype=np.float64)

    for cf in range(ncf):
        for tsep in range(1,tsep_max):
            for t_src in range(nt):
                t1 = t_src
                t2 = (tsep + t_src + Delta) % nt

                vac[cf,tsep] += (bubble_raw_1[cf,t1] * bubble_raw_2[cf,t2])

    #print(vac.shape)
    return vac


#function that constructs the direct diagram contribution from an array of two bubbles. 
#input arrays must be shape (ncf, nt), as this correlation needs to be done on an expression by expression basis
def construct_D(bubble1, bubble2, tsep_max, Delta,t_size):
    ncf = bubble1.shape[0]
    nt = bubble1.shape[1]
    D1 = np.zeros((ncf, tsep_max), dtype=np.complex128)
    D2 = np.zeros((ncf, tsep_max), dtype=np.complex128)
    
    #for tsep in range(tsep_max):
    #    for tsrc in range(nt):
    #        tsrc1 = tsrc
    #        tsrc2 = (tsrc - Delta) % t_size
    #        tsnk = (tsrc + tsep) % t_size
    #        tsnk2 = (tsrc + tsep + Delta) % t_size
            
            #src1 to snk1, src2 to snk2
    #        D1[:,tsep] += bubble1[:,tsnk,tsrc] * bubble1[:,tsnk2,tsrc2]

            #src1 to snk2, src2 to snk1
    #        D2[:,tsep] += bubble2[:,tsnk2, tsrc] * bubble2[:, tsnk, tsrc2]

    for tsep in range(tsep_max):
        tsep_2 = (tsep + 2*Delta) % t_size
        tsep_3 = (tsep + Delta) % t_size
        
        D1[:,tsep] += bubble1[:,tsep] * bubble2[:,tsep_2]
        D2[:,tsep] += bubble1[:,tsep_3] * bubble2[:,tsep_3]

    direct = (D1 + D2)
    return direct

#function that takes the raw correlator array with shape (ncf, nexpr, nt, nt), and iterates through the expression
#index to form the direct contribution for each term based on the isospin algebra for each two pion two point function
def construct_full_direct_diagram(pion_data, Delta, t_size, tsep_max, isospin=2):
    #pion_data expression order is set by the user in the auto contractor, so we should be able to just follow 
    #that to assign each correlator to a tag in a dictionary. 
    pion_dict = {
        "c": pion_data[:,0,:],
        "pp_000_f": pion_data[:,2,:],
        "pp_000_b": pion_data[:,2,:],
        "pp_001_f": pion_data[:,4,:],
        "pp_001_b": pion_data[:,4,:],
        "pp_011_f": pion_data[:,6,:],
        "pp_011_b": pion_data[:,6,:],
        "pp_111_f": pion_data[:,8,:],
        "pp_111_b": pion_data[:,8,:],
        
    }

    ncf = pion_data.shape[0]

    #build each necessary term
    C = construct_D(pion_dict['c'],pion_dict['c'],tsep_max,Delta,t_size)
    D1_000 = construct_D(pion_dict['pp_000_f'],pion_dict['pp_000_b'],tsep_max,Delta,t_size)
    D1_001 = construct_D(pion_dict['pp_001_f'],pion_dict['pp_001_b'],tsep_max,Delta,t_size)
    D1_011 = construct_D(pion_dict['pp_011_f'],pion_dict['pp_011_b'],tsep_max,Delta,t_size)
    D1_111 = construct_D(pion_dict['pp_111_f'],pion_dict['pp_111_b'],tsep_max,Delta,t_size)

    
    #manually writing the terms for now
    D_000 = 2*D1_000
    D_001 = 2*D1_001
    D_011 = 2*D1_011
    D_111 = 2*D1_111

    direct_full = np.stack((C,D_000,D_001,D_011,D_111),axis=1)
    return direct_full

#type sum function for pipi scattering. input shape is assumed to be (ncf,nt)
def pipi_type_sum(D_data, C_data, R_data=None, V_data=None, isospin=2):
    ncf = D_data.shape[0]
    tsep_max = D_data.shape[1]
    assert D_data.shape == C_data.shape
    corr_sum = np.zeros_like(D_data)
    if isospin == 2:
        for cf in range(ncf):
            corr_sum[cf,:] = D_data[cf,:] + C_data[cf,:]
    elif isospin == 0:
        assert R_data.shape == D_data.shape
        assert V_data.shape == D_data.shape
        for cf in range(ncf):
            corr_sum[cf,:] = D_data[cf,:] + C_data[cf,:] + R_data[cf,:] + V_data[cf,:]

    return corr_sum




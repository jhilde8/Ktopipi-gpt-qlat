import math
import os
import time
import importlib
import sys

from qlat.geometry import Geometry

import qlat_gpt as qg

from qlat_scripts.v1 import *
from auto_contractor.operators import *
from auto_contractor.wick import *

load_path_list[:] = [
    "/data1/qcddata2",
]

#standard pseudoscalar operator function.
def mk_scalar5(f1:str, f2:str, p:str, is_dagger=False):
    """
    q1bar g5 q2
    """
    s1 = new_spin_index()
    s2 = new_spin_index()
    c = new_color_index()
    if not is_dagger:
        return Qb(f1, p, s1, c) * G(5, s1, s2) * Qv(f2, p, s2, c) + f"({f1}bar g5 {f2})({p})"
    else:
        return -Qb(f2, p, s1, c) * G(5, s1, s2) * Qv(f1, p, s2, c) + f"(-{f2}bar g5 {f1})({p})"

#smeared pseudoscalar operator with explicit gauge fixing.
def mk_scalar5_sm(f1:str, f2:str, p1:str, p2:str, is_dagger=False):
    """
    q1bar g_inv g5 g q2
    """
    #color matrices are implemented as U, with an explicit mu index, so it is instead an implementation of a Lorentz 4 vector of color matrices. 
    # we can potentially work with that though.
    #At this point I want to say we simply include the color matrix in the operator, then in the expression function we connect it to the
    # gauge transformation matrices. We just set mu=0 for all of them. 
    
    s1 = new_spin_index()
    s2 = new_spin_index()
    c_1 = new_color_index()
    c_2 = new_color_index()
    c_3 - new_color_index()
    mu = 0 #color matrix implementation requires a mu argument, since the gauge fields are single SU(3) matrices we just set this to zero.
    gt_tag = 'gt' #not sure about these tags yet. 
    gt_inv_tag = 'gt_inv'
    if not is_dagger:
        return Qb(f1, p1, s1, c_1) * U(gt_inv_tag, p1, mu, c_1, c_2) * G(5, s1, s2) * U(gt_tag, p2, mu, c_2, c_3) * Qv(f2, p2, s2, c_3) + f"({f1}bar({p1}) gt_inv g5 gt {f2}({p2}))"
    else:
        return -Qb(f2, p2, s1, c_1) * U(gt_inv_tag, p1, mu, c_1, c_2) * G(5, s1, s2) * U(gt_tag, p2, mu, c_2, c_3) * Qv(f1, p1, s2, c_3) + f"(-{f2}bar({p2}) gt_inv g5 gt {f1}({p1}))"

def mk_meson_sm(f1:str, f2:str, p1:str, p2:str, is_dagger=False):
    """
    i q1bar g5 q2  #dag: i q2bar g5 q1
    """
    if not is_dagger:
        return sympy.I * mk_scalar5_sm(f1, f2, p1, p2, is_dagger) + f"(i {f1}bar({p1}) g5 {f2}({p2}))"
    else:
        return -sympy.I * mk_scalar5_sm(f1, f2, p1, p2, is_dagger) + f"(i {f2}bar({p2}) g5 {f1}({p1}))"

def mk_pi_p_sm(p1:str, p2:str, is_dagger=False):
    """
    i ubar g5 d  #dag: i dbar g5 u
    """
    return mk_meson_sm("u", "d", p1, p2, is_dagger) + f"pi+({p1},{p2}){show_dagger(is_dagger)}"

#now we must connect with the real world. 

@q.timer(is_timer_fork=True)
def extract_gauge_transform(job_tag,traj,get_psel_prob,get_fsel_prob,get_gt):
    total_site = q.Coordinate(get_param(job_tag, "total_site"))
    gt = get_gt()
    gt_inv = gt.inv()
    psel_prob = get_psel_prob()
    fsel_prob = get_fsel_prob()
    psel = psel_prob.psel
    fsel = fsel_prob.fsel
    if not fsel.is_containing(psel):
        q.displayln_info(-1, f"WARNING: fsel is not contatining psel. The probability weighting may be wrong.")
    fsel_n_elems = fsel.n_elems
    fsel_prob_arr = fsel_prob[:].ravel()
    psel_prob_arr = psel_prob[:].ravel()
    xg_psel_arr = psel[:] #all source and sink points array
    xg_fsel_arr = fsel.to_psel_local()[:]
    geo = Geometry(total_site)
    total_volume = geo.total_volume

    for pidx_src in range(len(xg_psel_arr)):
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])

        gt_ind = geo.index_from_coordinate(xg_src)
        
        gt_src = gt.get_elem(gt_ind)
        gt_src_inv = gt.get_elem(gt_ind)
        #gt_src_2 = gt.get_elem_xg(q.Coordinate(xg_src.to_tuple()),0)

        #direct indexing 
        #gt_src_3 = gt[xg_src]

        #gt_src is a numpy array/ Check unitarity
        gt_inv_np = np.linalg.inv(gt_src)
        gt_mat = np.matrix(gt_src)

        q.displayln_info(-1, f"gt inv 1: {gt_inv_np}")
        q.displayln_info(-1, f"gt inv 2: {gt_src_inv}")
        q.displayln_info(-1, f"np gt dagger: {gt_mat.getH()}")
        #q.displayln_info(-1, f"difference: {gt_inv_}")
        

        #q.displayln_info(-1,f"Gauge transformation matrix for element {gt_ind}, point {xg_src}: {gt_src}, type: {type(gt_src)}")
        #q.displayln_info(-1,f"Gauge transformation matrix for point {xg_src}: {gt_src_2}, type: {type(gt_src_2)}")
        #q.displayln_info(-1,f"Gauge transform direct indexing for point {xg_src}: {gt_src_3}, type: {type(gt_src_3)}")

        qu.ColorMatrix

@q.timer(is_timer_fork=True)
def run_man_contraction(
        job_tag, traj,
        *,
        get_get_prop,
        get_psel_prob,
        get_fsel_prob,
    ):

    fname = q.get_fname()
    #fn = f"{job_tag}/auto-contract-fsel-test/traj-{traj}/pipi_corr_psnk_psrc.lat"
    #if get_load_path(fn) is not None:
    #    return
    #cexpr = get_cexpr_pipi_corr_psnk_psrc()
    #expr_names = get_expr_names(cexpr)
    total_site = q.Coordinate(get_param(job_tag, "total_site"))
    t_size = total_site[3]
    get_prop = get_get_prop()
    psel_prob = get_psel_prob()
    fsel_prob = get_fsel_prob()
    psel = psel_prob.psel
    fsel = fsel_prob.fsel
    if not fsel.is_containing(psel):
        q.displayln_info(-1, f"WARNING: fsel is not containing psel. The probability weighting may be wrong.")
    fsel_n_elems = fsel.n_elems
    fsel_prob_arr = fsel_prob[:].ravel()
    psel_prob_arr = psel_prob[:].ravel()
    xg_psel_arr = psel[:]
    xg_fsel_arr = fsel.to_psel_local()[:]
    pidx_list_list = [ [] for i in range(t_size) ]
    fidx_list_list = [ [] for i in range(t_size) ]
    for pidx in range(len(xg_psel_arr)):
        xg = xg_psel_arr[pidx]
        pidx_list_list[xg[3]].append(pidx)

    for fidx in range(len(xg_fsel_arr)):
        xg = xg_fsel_arr[fidx]
        fidx_list_list[xg[3]].append(fidx)
    
    #
    geo = q.Geometry(total_site)
    total_volume = geo.total_volume

    # --- fsel test ---

    print(f"DEBUG: psel arr size: {xg_psel_arr.shape}")
    print(f"DEBUG: fsel arr size: {xg_fsel_arr.shape}")

    assert 1 == 0
    
    # -----------------
    
    #these are some strange type, but these are what is used by the auto contractor so I could just copy how the 
    #ac code deals with these. 
    #Sl_test_get1 = get_prop("l",pd["x_1"],pd["x_2"])
    #Sl_test_get2 = get_prop("l", pd["x_2"],pd["x_1"])

    #Sl_test_1 = load_prop(Sl_test_get1)
    #Sl_test_2 = load_prop(Sl_test_get2)
    
    #this is a color matrix, shape (3,3) as a numpy array. 
    #gt_inv = gt.inv()
    #it_inv_1 = gt_inv.get_elem(2127)
    #gt_inv_2 = gt_inv.get_elem(2128)
    
    #q.displayln_info(f"DEBUG: propagator trace attempt {type(Sl_test_1)}")
    #q.displayln_info(f"DEBUG: source gauge fix attempt: {gt_inv_local}")

    v = [f"{fname} {job_tag} {traj} done"]
    return v

def fill_prop_cache(job_tag, traj):
    traj_gf = traj
    
    fns_need = [
        (f"{job_tag}/psel-prop-psrc-light/traj-{traj}.qar", f"{job_tag}/psel-prop-psrc-light/traj-{traj}/checkpoint.txt",),
        (f"{job_tag}/psel-prop-psrc-strange/traj-{traj}.qar", f"{job_tag}/psel-prop-psrc-strange/traj-{traj}/checkpoint.txt",),
        (f"{job_tag}/prop-psrc-light/traj-{traj}.qar", f"{job_tag}/prop-psrc-light/traj-{traj}/geon-info.txt",),
        (f"{job_tag}/prop-psrc-strange/traj-{traj}.qar", f"{job_tag}/prop-psrc-strange/traj-{traj}/geon-info.txt",),
        f"{job_tag}/gauge-transform/traj-{traj_gf}.field",
        f"{job_tag}/points-selection/traj-{traj}.lati",
        f"{job_tag}/field-selection/traj-{traj}.field",
    ]

    get_gf = None
    get_gt = run_gt(job_tag, traj_gf, get_gf)
    get_f_weight = run_f_weight_uniform(job_tag, traj)
    get_f_rand_01 = run_f_rand_01(job_tag, traj)
    get_fsel_prob = run_fsel_prob(job_tag, traj, get_f_rand_01=get_f_rand_01, get_f_weight=get_f_weight)
    get_psel_prob = run_psel_prob(job_tag, traj, get_f_rand_01=get_f_rand_01, get_f_weight=get_f_weight)
    get_fsel = run_fsel_from_fsel_prob(get_fsel_prob)
    get_psel = run_psel_from_psel_prob(get_psel_prob)

    prop_types = [
        "psrc psel s",
        "psrc psel l",
        "psrc fsel s",
        "psrc fsel l",
    ]

    #get_get_prop = run_get_prop(
    #    job_tag, traj,
    #    get_gf = get_gf,
    #    get_gt = get_gt,
    #    get_psel = get_psel,
    #    get_fsel = get_fsel,
    #    prop_types = prop_types,
    #)

    run_r_list(job_tag)
    #extract_gauge_transform(job_tag, traj, get_psel_prob, get_fsel_prob, get_gt)
    run_man_contraction(job_tag, traj, get_get_prop=get_get_prop, get_psel_prob=get_psel_prob, get_fsel_prob=get_fsel_prob)
    
    #do something here
    q.clean_cache()
    
set_param("16IH2", "traj_list")([1000])
set_param("16IH2", "measurement", "auto_contractor_chunk_size")(128)
set_param("16IH2", "measurement", "meson_tensor_t_sep")(12)
set_param("16IH2", "measurement", "pipi_op_t_sep")(5) #Delta
set_param("16IH2", "measurement", "pipi_op_dis_4d_sqr_limit")(25.0) #minimum squared distance for single pions
set_param("16IH2", "measurement", "pipi_corr_t_sep_list")(list(range(1, 23)))
set_param("16IH2", "measurement", "tsep_snk_src_3pt")(24) #constant source-sink separation in 3pt function
set_param("16IH2", "measurement", "use_fsel_prop")(True)

set_param("48I", "traj_list")([1102]) #list(range(1102, 1493,10))+list(range(1505,1636,10))+list(range(1705, 2176,10)) + list(range(1005, 1096, 10)))
set_param("48I", "measurement", "auto_contractor_chunk_size")(128)
set_param("48I", "measurement", "meson_tensor_t_sep")(12)
set_param("48I", "measurement", "pipi_op_t_sep")(5) #time separation between the two pions in a two pion operator. this is Delta
set_param("48I", "measurement", "pipi_op_dis_4d_sqr_limit")(25.0) #Minimum squared 4d distance between the two pion operators. We need to try with 9.0 and 16.0
set_param("48I", "measurement", "pipi_corr_t_sep_list")(list(range(1, 21))) #list of time separations between the two pion operators that we want to measure
set_param("48I", "measurement", "pipi_tensor_t_sep_list")([ 1, 2, ]) #not used
set_param("48I", "measurement", "pipi_tensor_t_max")(20) #not used
set_param("48I", "measurement", "pipi_tensor_r_max")(24) #not used
set_param("48I", "measurement", "use_fsel_prop")(False)


################### CMD OPTIONS #######################
job_tag_list_default = [
        "test-4nt8-checker",
        ]
job_tag_list_str_default = ",".join(job_tag_list_default)
job_tag_list = q.get_arg("--job_tag_list", default=job_tag_list_str_default).split(",")

is_performing_inversion = not q.get_option("--no-inversion")

is_performing_contraction = not q.get_option("--no-contraction")

#######################################################


#This is a python file that I am going to use to load in propagators and gauge transform to 
#see about shapes, and how gauge fixing the source works. 
def gracefully_finish():
    q.displayln_info("Begin to gracefully_finish.")
    q.timer_display()
    if is_test():
        q.json_results_append(f"q.obtained_lock_history_list={q.obtained_lock_history_list}")
        q.check_log_json(__file__)
    qg.end_with_gpt()
    q.displayln_info("CHECK: finished successfully.")
    exit()

def try_gracefully_finish():
    """
    Call `gracefully_finish` if not test and if some work is done (q.obtained_lock_history_list != [])
    """
    if (not is_test()) and (len(q.obtained_lock_history_list) > 0):
        gracefully_finish()
    

if __name__ == "__main__":
    qg.begin_with_gpt()
    q.get_time_limit()

    job_tag_traj_list = []
    for job_tag in job_tag_list:
        run_params(job_tag)
        traj_list = get_param(job_tag, "traj_list")
        for traj in traj_list:
            job_tag_traj_list.append((job_tag, traj,))
    k_count = 0
    ncf = 1 
    for job_tag, traj, in job_tag_traj_list:
        if is_performing_contraction:
            q.get_time_limit()
            fill_prop_cache(job_tag, traj)
            q.clean_cache()
            k_count += 1
            if k_count >= ncf:
                try_gracefully_finish()
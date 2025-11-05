#!/usr/bin/env python3

import functools
import math
import os
import time
import importlib
import sys

import qlat_gpt as qg

from qlat_scripts.v1 import *
from auto_contractor.operators import *

is_cython = not is_test()

# ----

load_path_list[:] = [
        "results",
        "qcddata",
        "/lustre20/volatile/qcdqedta/qcddata",
        "/lustre20/volatile/decay0n2b/qcddata",
        "/lustre20/volatile/pqpdf/ljin/qcddata",
        "/data1/qcddata2",
        "/data1/qcddata3",
        "/data2/qcddata3-prop",
        ]

# ----
#Wall source meson correlator
# ----

#point source meson correlator
#cosine is symmetric. Any permutations of negative signs that change the overall sign
# in the cosine will be identical. 

#We average over specific fourier components that produce an identical energy or an equivalent momentum magnitude. 

def wave_function_mode_0(c12, size):
    return 1.0

#an average over the three directions for the (0,0,1) mode
#we should potentially average over (0,0,-1)'s ? 
def wave_function_mode_1(c12, size):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    w1 = np.cos(2.0 * np.pi * x / xs)
    w2 = np.cos(2.0 * np.pi * y / ys)
    w3 = np.cos(2.0 * np.pi * z / zs)
    w = (w1 + w2 + w3) / 3.0
    return w


#average over three positive momenta modes (0,1,1)
def wave_function_mode_2(c12, size):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    w1 = np.cos((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys))
    w2 = np.cos((2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
    w3 = np.cos((2.0 * np.pi * z / zs) + (2.0 * np.pi * x / xs))

    w4 = np.cos((2.0 * np.pi * x / zs) - (2.0 * np.pi * y / xs))
    w5 = np.cos((2.0 * np.pi * y / zs) - (2.0 * np.pi * z / xs))
    w6 = np.cos((2.0 * np.pi * z / zs) - (2.0 * np.pi * x / xs))
    w = (w1 + w2 + w3 + w4 + w5 + w6) / 6.0
    return w


#(1,1,1)
def wave_function_mode_3(c12, size):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple() 
    w1 = np.cos((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
    w2 = np.cos((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) - (2.0 * np.pi * z / zs))
    w3 = np.cos((2.0 * np.pi * x / xs) - (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
    w4 = np.cos(-(2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
    w = (w1 + w2 + w3 + w4) / 4.0
    return w

#exponential version of (1,1,1)
def wave_function_mode_4(c12, size):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple() 
    w1 = np.exp(-1j*((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs)))
    w2 = np.exp(-1j*((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) - (2.0 * np.pi * z / zs)))
    w3 = np.exp(-1j*((2.0 * np.pi * x / xs) - (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs)))
    w4 = np.exp(-1j*(-(2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs)))
    w = (w1 + w2 + w3 + w4) / 4.0
    return w




wave_function_mode_dict = dict()
wave_function_mode_dict[0] = wave_function_mode_0 #Identity momentum projection
#wave_function_mode_dict[1] = wave_function_mode_1
#wave_function_mode_dict[2] = wave_function_mode_2
#wave_function_mode_dict[3] = wave_function_mode_3
#wave_function_mode_dict[4] = wave_function_mode_4
#wave_function_mode_dict[5] = wave_function_mode_5
#wave_function_mode_dict[6] = wave_function_mode_6
#wave_function_mode_dict[7] = wave_function_mode_7

def wave_function(p1, p2, mode, size):
    p1_tag, c1 = p1
    p2_tag, c2 = p2
    c1 = q.Coordinate(c1)
    c2 = q.Coordinate(c2)
    c12 = q.smod_coordinate(c1 - c2, size)
    if mode not in wave_function_mode_dict:
        fname = q.get_fname()
        raise Exception(f"{fname}: {p1} {p2} {mode} {size}")
    wf = wave_function_mode_dict[mode]
    return wf(c12, size)

@q.timer
def get_cexpr_meson_corr_psnk_psrc():
    fn_base = "cache/auto_contract_cexpr_pos/get_cexpr_meson_corr_psnk_psrc"
    def calc_cexpr():
        diagram_type_dict = dict()
        diagram_type_dict[((('x_1', 'x_2'), 1), (('x_2', 'x_1'), 1))] = 'Type1'
        diagram_type_dict[((('x_1', 'x_1'), 1), (('x_2', 'x_2'), 1))] = None
        exprs = [
                mk_fac(1) + f"1",  mk_fac(1) * mk_pi_p("x_2", True)    * mk_pi_p("x_1")
                    + f" <1> * pi+^dag(0) * pi+(-tsep)", ]
       # for mode in [0]:
       #     exprs += [
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   #  mk_fac(1)
                   # * mk_pi_p("x_2", True)    * mk_pi_p("x_1")
                   # + f"wf({mode}) * pi+^dag(0) * pi+(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1,x_2, {mode}, size)")
                   # * mk_pi_m("x_2",True) * mk_pi_m("x_1")
                   # + f"wf({mode}) * pi-^dag(0) * pi-(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1,x_2, {mode}, size)")
                   # * mk_pi_0("x_2", True) * mk_pi_0("x_1")
                   # + f"wf({mode}) * pi0^dag(0) * pi0(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_j5pi_mu("x_2", 3)    * mk_pi_p("x_1")
                   # + f"wf({mode}) * j5pi_t(0) * pi+(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_pi_p("x_2", True)    * mk_j5pi_mu("x_1", 3, True)
                   # + f"wf({mode}) * pi+^dag(0) * j5pi_t^dag(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_j5pi_mu("x_2", 3)    * mk_j5pi_mu("x_1", 3, True)
                   # + f"wf({mode}) * j5pi_t(0) * j5pi_t^dag(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_k_0("x_2", True)     * mk_k_0("x_1")
                   # + f"wf({mode}) * K0^dag(0) * K0(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_j5k_mu("x_2", 3)     * mk_k_p("x_1")
                   # + f"wf({mode}) * j5k_t(0) * K+(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_k_p("x_2", True)     * mk_j5k_mu("x_1", 3, True)
                   # + f"wf({mode}) * K+^dag(0) * j5k_t^dag(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_j5k_mu("x_2", 3)     * mk_j5k_mu("x_1", 3, True)
                   # + f"wf({mode}) * j5k_t(0) * j5k_t^dag(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_a0_p("x_2", True)    * mk_a0_p("x_1")
                   # + f"wf({mode}) * a0+^dag(0) * a0+(-tsep)",
                    #
                   # mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                   # * mk_kappa_p("x_2", True) * mk_kappa_p("x_1")
                   # + f"wf({mode}) * kappa+^dag(0) * kappa+(-tsep)",
                    #
                    #]
        cexpr = contract_simplify_compile(
                *exprs,
                is_isospin_symmetric_limit=True,
                diagram_type_dict=diagram_type_dict,
                )
        return cexpr
    base_positions_dict = dict()
    base_positions_dict["wave_function"] = wave_function
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )

@q.timer(is_timer_fork=True)
def auto_contract_meson_corr_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract/traj-{traj}/meson_corr_psnk_psrc.lat"
    if get_load_path(fn) is not None:
        return
    cexpr = get_cexpr_meson_corr_psnk_psrc()
    expr_names = get_expr_names(cexpr)
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
    print(f"psel array size: {xg_psel_arr.shape}")
    xg_fsel_arr = fsel.to_psel_local()[:]
    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    def load_data():
        for pidx in range(len(xg_psel_arr)):
            yield pidx
    @q.timer
    def feval(args):
        pidx = args
        xg_src = q.Coordinate(xg_psel_arr[pidx])
        prob_src = psel_prob_arr[pidx]
        values = np.zeros((total_site[3], len(expr_names),), dtype=np.complex128)
        for idx in range(len(xg_fsel_arr)):
            xg_snk = q.Coordinate(xg_fsel_arr[idx])
            if xg_snk == xg_src:
                prob_snk = 1.0
            else:
                prob_snk = fsel_prob_arr[idx]
            prob = prob_src * prob_snk
            x_rel = q.smod_coordinate(xg_snk - xg_src, total_site)
            x_rel_t = x_rel[3]
            pd = {
                    "x_2" : ("point", xg_src.to_tuple(),),
                    "x_1" : ("point-snk", xg_snk.to_tuple(),),
                    "size" : total_site,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values[x_rel_t] += val / prob
        return values
    def sum_function(val_list):
        values = np.zeros((total_site[3], len(expr_names),), dtype=np.complex128)
        for val in val_list:
            values += val
        return values.transpose(1, 0)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 / (t_size * (total_volume / t_size))
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "t_sep", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

@q.timer(is_timer_fork=True)
def auto_contract_meson_corr_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pos-4/traj-{traj}/meson_corr_psnk_psrc_psel.lat"
    if get_load_path(fn) is not None:
        return
    cexpr = get_cexpr_meson_corr_psnk_psrc()
    expr_names = get_expr_names(cexpr)
    total_site = q.Coordinate(get_param(job_tag, "total_site"))
    x_size = total_site[0]
    y_size = total_site[1]
    z_size = total_site[2]
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
    print(f"psel array size: {xg_psel_arr.shape}")
    xg_fsel_arr = fsel.to_psel_local()[:]
    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    #generator function that creates an iterator that is used in parallel_map_sum. q.get_mpi_chunk breaks the full point array into subsets based on the number of nodes, 
    #so this generator creates an iterator over a subset of indices to be used in the feval calls on the current node. 
    def load_data():
        lsize = 16 #how many indices per task
        pidx_list = []
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):
            
            pidx_list.append(pidx) 
            #we yield the list of indices if the list is of our desired size. The list is then emptied so that we can do it again
            if len(pidx_list) == lsize:
                yield pidx_list
                pidx_list = []

        #if there are any leftover indices that werent yielded because the number of indices is not divisible by lsize, we yield those at the end. 
        if pidx_list:
            yield pidx_list


    @q.timer
    #currently feval is called once per worker node, and it is passed in a single index. We hit a huge bottleneck when doing the parallel map sum because 
    def feval(args):
        pidx_list = args #this is a list now. We must iteratre over it along with iterating over sinks
        #values_ind = np.zeros((total_site[0], total_site[1], total_site[2], total_site[3], len(expr_names),), dtype=np.complex128) #for each individual source 
        values_ind = np.zeros(((total_site[0]//2+1), (total_site[1]//2+1), (total_site[2]//2+1), total_site[3], len(expr_names),), dtype=np.complex128) #accumulation over all sources in this call

        for pidx in pidx_list: #iterate through the index list we passed in
            xg_src = q.Coordinate(xg_psel_arr[pidx]) 
            prob_src = psel_prob_arr[pidx]
            
            for idx in range(len(xg_psel_arr)): #use all randomly selected points as sinks
                xg_snk = q.Coordinate(xg_psel_arr[idx])
                if xg_snk == xg_src:
                    prob_snk = 1.0
                else:
                    prob_snk = psel_prob_arr[idx]
                prob = prob_src * prob_snk
                x_rel = q.smod_coordinate(xg_snk - xg_src, total_site)
                x_rel_t = x_rel[3]
                pd = {
                        "x_2" : ("point", xg_src.to_tuple(),),
                        "x_1" : ("point", xg_snk.to_tuple(),),
                        "size" : total_site,
                     }
                val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
           
                values_ind[abs(x_rel[0]),abs(x_rel[1]),abs(x_rel[2]),x_rel_t] += val

            #values += values_ind

        return values_ind

    def sum_function(val_list):
        values = np.zeros((total_site[0]//2+1,total_site[1]//2+1,total_site[2]//2+1,total_site[3], len(expr_names),), dtype=np.complex128)
        for val in val_list:
            values += val
           # valtp = np.transpose(values) #this probably takes a while
        return values
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 / (t_size * (total_volume / t_size))
    ld = q.mk_lat_data([
        [ "x_sep", x_size//2+1, [ str(x) for x in range(x_size//2+1) ], ],
        [ "y_sep", y_size//2+1, [ str(y) for y in range(y_size//2+1) ], ],
        [ "z_sep", z_size//2+1, [ str(z) for z in range(z_size//2+1) ], ],
        [ "t_sep", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        [ "expr_name", len(expr_names), expr_names, ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

# ----
#Wall source pion correlator
# ----


def pipi_wave_function_mode_0(c12, size, pipi_op_dis_4d_sqr_limit):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    dis_4d_sqr = c12.sqr()
    if dis_4d_sqr <= pipi_op_dis_4d_sqr_limit:
        return 0.0
    return 1.0

#momentum mode (0,0,1) and permutations
def pipi_wave_function_mode_1(c12,size, pipi_op_dis_4d_sqr_limit):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    dis_4d_sqr = c12.sqr()
    if dis_4d_sqr <= pipi_op_dis_4d_sqr_limit:
        return 0.0
    w1 = np.cos(2.0 * np.pi * x/xs)
    w2 = np.cos(2.0 * np.pi * y/ys)
    w3 = np.cos(2.0 * np.pi * z/zs)
    w = (w1 + w2 + w3)/3.0
    return w

def pipi_wave_function_mode_2(c12,size,pipi_op_dis_4d_sqr_limit):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    dis_4d_sqr = c12.sqr()
    if dis_4d_sqr <= pipi_op_dis_4d_sqr_limit:
        return 0.0
    else:   
        w1 = np.cos((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys))
        w2 = np.cos((2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
        w3 = np.cos((2.0 * np.pi * z / zs) + (2.0 * np.pi * x / xs))

        w4 = np.cos((2.0 * np.pi * x / zs) - (2.0 * np.pi * y / xs))
        w5 = np.cos((2.0 * np.pi * y / zs) - (2.0 * np.pi * z / xs))
        w6 = np.cos((2.0 * np.pi * z / zs) - (2.0 * np.pi * x / xs))
        w = (w1 + w2 + w3 + w4 + w5 + w6) / 6.0
        return w

def pipi_wave_function_mode_3(c12, size, pipi_op_dis_4d_sqr_limit):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    dis_4d_sqr = c12.sqr()
    if dis_4d_sqr <= pipi_op_dis_4d_sqr_limit:
        return 0.0
    else:
        w1 = np.cos((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
        w2 = np.cos((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) - (2.0 * np.pi * z / zs))
        w3 = np.cos((2.0 * np.pi * x / xs) - (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
        w4 = np.cos(-(2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
        w = (w1 + w2 + w3 + w4) / 4.0
        return w

#change this to include all modes we want
# pipi_op_tsep must change to the number of inlcuded modes
pipi_wave_function_mode_dict = dict()
pipi_wave_function_mode_dict[0] = pipi_wave_function_mode_0
pipi_wave_function_mode_dict[1] = pipi_wave_function_mode_1
pipi_wave_function_mode_dict[2] = pipi_wave_function_mode_2
pipi_wave_function_mode_dict[3] = pipi_wave_function_mode_3

def pipi_wave_function(p1, p2, mode, size, pipi_op_dis_4d_sqr_limit):
    p1_tag, c1 = p1
    p2_tag, c2 = p2
    c1 = q.Coordinate(c1)
    c2 = q.Coordinate(c2)
    c12 = q.smod_coordinate(c1 - c2, size)
    if mode not in pipi_wave_function_mode_dict:
        fname = q.get_fname()
        raise Exception(f"{fname}: {p1} {p2} {mode} {size}")
    wf = pipi_wave_function_mode_dict[mode]
    return wf(c12, size, pipi_op_dis_4d_sqr_limit)

@q.timer
def get_cexpr_pipi_corr_psnk_psrc():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_corr_psnk_psrc"
    def calc_cexpr():
        diagram_type_dict = dict()
        exprs = [
                mk_fac(1) + f"1",
                ]
        for mode_src in [ 0, 1, 2, 3]:
            for mode_snk in [ 0, 1, 2, 3]:
                exprs += [
                        #
                         #mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                         #* mk_fac(f"pipi_wave_function(src_1, src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                         #* mk_pipi_i22("snk_1", "snk_2", True)
                         #* mk_pipi_i22("src_1", "src_2")
                         #+ f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i22+^dag(0) * pipi_i22(-tsep)",
                        #
                        mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i20("snk_1", "snk_2", True) #true refers to the is_dagger boolean
                        * mk_pipi_i20("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i20^dag(0) * pipi_i20(-tsep)",
                        ]
        cexpr = contract_simplify_compile(
                *exprs,
                is_isospin_symmetric_limit=True,
                diagram_type_dict=diagram_type_dict,
                )
        return cexpr
    base_positions_dict = dict()
    base_positions_dict["pipi_wave_function"] = pipi_wave_function
    base_positions_dict["pipi_op_dis_4d_sqr_limit"] = 0.5 # default value, to be overrided by `pd`.
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )

@q.timer(is_timer_fork=True)
def auto_contract_pipi_corr_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract/traj-{traj}/pipi_corr_psnk_psrc.lat"
    if get_load_path(fn) is not None:
        return
    cexpr = get_cexpr_pipi_corr_psnk_psrc()
    expr_names = get_expr_names(cexpr)
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
    #
    # pidx_list_list[t_slice] == [ pidx1, pidx2, ... ]
    # xg_psel_arr[pidx1][3] == t_slice
    pidx_list_list = [ [] for i in range(t_size) ]
    for pidx in range(len(xg_psel_arr)):
        xg = xg_psel_arr[pidx]
        pidx_list_list[xg[3]].append(pidx)
    #
    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    pipi_op_t_sep = get_param(job_tag, "measurement", "pipi_op_t_sep")
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")
    pipi_corr_t_sep_list = get_param(job_tag, "measurement", "pipi_corr_t_sep_list")
    data_list = []
    # This bit iterates through all source locations, and finds each sink location based on the time separation we want, set in the parameter block
    # Each source location's timeslice is saved, and the sink ts is calculated from the t_sep of that current iteration, for which the full 
    # sink location is reconstructed, iterating through each possible sink given the sink ts. These pairs of locations are saved in data_list.
    for pidx_src in range(len(xg_psel_arr)):
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_src = xg_src[3]
        for t_sep_idx, t_sep in enumerate(pipi_corr_t_sep_list):
            assert t_sep > 0
            t_snk = (t_src + t_sep) % t_size
            for pidx_snk in pidx_list_list[t_snk]:
                xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
                assert xg_snk[3] == t_snk
                if pidx_snk == pidx_src:
                    continue
                data_list.append((pidx_snk, pidx_src, t_sep_idx,))
    def load_data():

        lsize = 8


        data_list_chunk = q.get_mpi_chunk(data_list)
        data_list_size = len(data_list_chunk)
        for data_list_idx, (pidx_snk, pidx_src, t_sep_idx,) in enumerate(data_list_chunk):
            yield data_list_idx, data_list_size, pidx_snk, pidx_src, t_sep_idx
    @q.timer
    def feval(args):
        data_list_idx, data_list_size, pidx_snk, pidx_src, t_sep_idx = args
        assert pidx_src != pidx_snk
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_src = xg_src[3]
        assert pidx_snk != pidx_src
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_src]
        pidx_snk_src_2_list = []
        #each two pion operator has some separation between pions within that operator, the input to feval in the two pion case is a source AND sink, and then based on the 
        # internal pion separation, two other source and sink locations are constructed in the same way as above, since one timeslice may have multiple suorce locations. 
        pipiop_tsep = [2,3]
        for pipi_op_t_sep_src in pipiop_tsep: #iterate over possible internal pion separations
            t_src_2 = (t_src - pipi_op_t_sep_src) % t_size
            for pipi_op_t_sep_snk in pipiop_tsep:
                t_snk_2 = (t_snk + pipi_op_t_sep_snk) % t_size
                for pidx_src_2 in pidx_list_list[t_src_2]:
                    if pidx_src_2 in [ pidx_snk, pidx_src, ]:
                        continue
                    for pidx_snk_2 in pidx_list_list[t_snk_2]:
                        if pidx_snk_2 in [ pidx_src_2, pidx_snk, pidx_src, ]:
                            continue
                        prob2 = psel_prob_arr[pidx_snk_2] * psel_prob_arr[pidx_src_2]
                        prob = prob1 * prob2
                        pidx_snk_src_2_list.append((pidx_snk_2, pidx_src_2, prob,)) #we then save a list of the second source and sink locations
        #values array holds the evaluation of a given expression for each internal pion separation for both initial and final pions for each expression
        values = np.zeros(
                (pipi_op_t_sep,
                 pipi_op_t_sep,
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        
        #iterating over the second sources and sinks, we evaluate each expression with the given source and sink locations, and assign them to the values array
        for pidx_snk_2, pidx_src_2, prob in pidx_snk_src_2_list:
            xg_snk_2 = q.Coordinate(xg_psel_arr[pidx_snk_2])
            xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
            t_snk_2 = xg_snk_2[3]
            t_src_2 = xg_src_2[3]
            pipi_op_t_sep_snk = (t_snk_2 - t_snk) % t_size
            pipi_op_t_sep_src = (t_src - t_src_2) % t_size
            pd = {
                    "snk_1": ("point", xg_snk.to_tuple(),),
                    "snk_2": ("point", xg_snk_2.to_tuple(),),
                    "src_1": ("point", xg_src.to_tuple(),),
                    "src_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values[pipi_op_t_sep_snk, pipi_op_t_sep_src] += val / prob
        return values, t_sep_idx, data_list_idx, data_list_size
    def sum_function(val_list):
        values = np.zeros(
                (len(pipi_corr_t_sep_list), 
                 pipi_op_t_sep,
                 pipi_op_t_sep,
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        for val, t_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size // 1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")
            values[t_sep_idx] += val
        return values.transpose(3, 0, 1, 2,)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 / (t_size * (total_volume / t_size) * (total_volume / t_size))
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "t_sep", len(pipi_corr_t_sep_list), pipi_corr_t_sep_list, ],
        [ "pipi_op_t_sep_snk", pipi_op_t_sep, [ str(i+2) for i in range(pipi_op_t_sep) ], ],
        [ "pipi_op_t_sep_src", pipi_op_t_sep, [ str(i+2) for i in range(pipi_op_t_sep) ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

# This contraction function returns a set of values for each two pion operator separation (set to [0,16] for now), each internal pion separation for the source and sink two pion operator
# and each expression (effectively momentum mode at this point). 

# ----
# pipi_jj wall source
# ----
# pipi_jj point source
# ----

@q.timer(is_timer_fork=True)
def run_auto_contraction(
        job_tag, traj,
        *,
        get_get_prop,
        get_psel_prob,
        get_fsel_prob,
        ):
    fname = q.get_fname()
    fn_checkpoint = f"{job_tag}/auto-contract/traj-{traj}/checkpoint.txt"
    if get_load_path(fn_checkpoint) is not None:
        q.displayln_info(0, f"{fname}: '{fn_checkpoint}' exists.")
        return
    if not q.obtain_lock(f"locks/{job_tag}-{traj}-{fname}"):
        return
    get_prop = get_get_prop()
    assert get_prop is not None
    use_fsel_prop = get_param(job_tag, "measurement", "use_fsel_prop", default=True)
    # ADJUST ME
   # auto_contract_meson_corr(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
   # if use_fsel_prop:
   #     auto_contract_meson_jj(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
   # auto_contract_pipi_corr(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
   # if use_fsel_prop:
   #     auto_contract_pipi_jj(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
   # if use_fsel_prop:
   #     auto_contract_meson_corr_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
    #auto_contract_meson_corr_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
    auto_contract_pipi_corr_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
    #
    q.qtouch_info(get_save_path(fn_checkpoint))
    q.release_lock()
    v = [ f"{fname} {job_tag} {traj} done", ]
    return v

### ------

@q.timer(is_timer_fork=True)
def run_job_inversion(job_tag, traj):
    #
    traj_gf = traj
    if is_test():
        # ADJUST ME
        traj_gf = 1000
        #
    #
    fns_produce = [
            (f"{job_tag}/prop-psrc-light/traj-{traj}.qar", f"{job_tag}/prop-psrc-light/traj-{traj}/geon-info.txt",),
            (f"{job_tag}/psel-prop-psrc-light/traj-{traj}.qar", f"{job_tag}/psel-prop-psrc-light/traj-{traj}/checkpoint.txt",),
            (f"{job_tag}/prop-psrc-strange/traj-{traj}.qar", f"{job_tag}/prop-psrc-strange/traj-{traj}/geon-info.txt",),
            (f"{job_tag}/psel-prop-psrc-strange/traj-{traj}.qar", f"{job_tag}/psel-prop-psrc-strange/traj-{traj}/checkpoint.txt",),
            #
            (f"{job_tag}/prop-wsrc-light/traj-{traj}.qar", f"{job_tag}/prop-wsrc-light/traj-{traj}/geon-info.txt",),
            (f"{job_tag}/psel-prop-wsrc-light/traj-{traj}.qar", f"{job_tag}/psel-prop-wsrc-light/traj-{traj}/checkpoint.txt",),
            (f"{job_tag}/prop-wsrc-strange/traj-{traj}.qar", f"{job_tag}/prop-wsrc-strange/traj-{traj}/geon-info.txt",),
            (f"{job_tag}/psel-prop-wsrc-strange/traj-{traj}.qar", f"{job_tag}/psel-prop-wsrc-strange/traj-{traj}/checkpoint.txt",),
            ]
    fns_need = [
            # f"{job_tag}/gauge-transform/traj-{traj}.field",
            # f"{job_tag}/points-selection/traj-{traj}.lati",
            # f"{job_tag}/field-selection/traj-{traj}.field",
            # f"{job_tag}/wall-src-info-light/traj-{traj}.txt",
            # f"{job_tag}/wall-src-info-strange/traj-{traj}.txt",
            # (f"{job_tag}/configs/ckpoint_lat.{traj}", f"{job_tag}/configs/ckpoint_lat.IEEE64BIG.{traj}",),
            ]
    if not check_job(job_tag, traj, fns_produce, fns_need):
        return
    #
    get_gf = run_gf(job_tag, traj_gf)
    get_gt = run_gt(job_tag, traj_gf, get_gf)
    #
    get_wi = run_wi(job_tag, traj)
    #
    get_eig_light = run_eig(job_tag, traj_gf, get_gf)
    get_eig_strange = run_eig_strange(job_tag, traj_gf, get_gf)
    #
    def run_wsrc_full():
        get_eig = get_eig_light
        # run_get_inverter(job_tag, traj, inv_type=0, get_gf=get_gf, get_gt=get_gt, get_eig=get_eig)
        run_prop_wsrc_full(job_tag, traj, inv_type=0, get_gf=get_gf, get_eig=get_eig, get_gt=get_gt, get_wi=get_wi)
        #
        get_eig = get_eig_strange
        # run_get_inverter(job_tag, traj, inv_type=1, get_gf=get_gf, get_gt=get_gt, get_eig=get_eig)
        run_prop_wsrc_full(job_tag, traj, inv_type=1, get_gf=get_gf, get_eig=get_eig, get_gt=get_gt, get_wi=get_wi)
    #
    run_wsrc_full()
    #
    get_f_weight = run_f_weight_from_wsrc_prop_full(job_tag, traj)
    get_f_rand_01 = run_f_rand_01(job_tag, traj)
    get_fsel_prob = run_fsel_prob(job_tag, traj, get_f_rand_01=get_f_rand_01, get_f_weight=get_f_weight)
    get_psel_prob = run_psel_prob(job_tag, traj, get_f_rand_01=get_f_rand_01, get_f_weight=get_f_weight)
    get_fsel = run_fsel_from_fsel_prob(get_fsel_prob)
    get_psel = run_psel_from_psel_prob(get_psel_prob)
    #
    get_fselc = run_fselc(job_tag, traj, get_fsel, get_psel)
    #
    get_eig = get_eig_light
    run_prop_wsrc_sparse(job_tag, traj, inv_type=0, get_gf=get_gf, get_eig=get_eig, get_gt=get_gt, get_psel=get_psel, get_fsel=get_fsel, get_wi=get_wi)
    get_eig = get_eig_strange
    run_prop_wsrc_sparse(job_tag, traj, inv_type=1, get_gf=get_gf, get_eig=get_eig, get_gt=get_gt, get_psel=get_psel, get_fsel=get_fsel, get_wi=get_wi)
    #
    def run_with_eig():
        get_eig = get_eig_light
        # run_get_inverter(job_tag, traj, inv_type=0, get_gf=get_gf, get_eig=get_eig)
        # run_prop_wsrc(job_tag, traj, inv_type=0, get_gf=get_gf, get_eig=get_eig, get_gt=get_gt, get_psel=get_psel, get_fsel=get_fselc, get_wi=get_wi)
        run_prop_psrc(job_tag, traj, inv_type=0, get_gf=get_gf, get_eig=get_eig, get_gt=get_gt, get_psel=get_psel, get_fsel=get_fselc, get_f_rand_01=get_f_rand_01)
        q.clean_cache(q.cache_inv)
    #
    def run_with_eig_strange():
        get_eig = get_eig_strange
        # run_get_inverter(job_tag, traj, inv_type=1, get_gf=get_gf, get_eig=get_eig)
        # run_prop_wsrc(job_tag, traj, inv_type=1, get_gf=get_gf, get_eig=get_eig, get_gt=get_gt, get_psel=get_psel, get_fsel=get_fselc, get_wi=get_wi)
        run_prop_psrc(job_tag, traj, inv_type=1, get_gf=get_gf, get_eig=get_eig, get_gt=get_gt, get_psel=get_psel, get_fsel=get_fselc, get_f_rand_01=get_f_rand_01)
        q.clean_cache(q.cache_inv)
    #
    def run_charm():
        # run_get_inverter(job_tag, traj, inv_type=2, get_gf=get_gf)
        q.clean_cache(q.cache_inv)
    #
    run_with_eig()
    run_with_eig_strange()
    run_charm()
    #
    q.clean_cache()
    if q.obtained_lock_history_list:
        q.timer_display()

@q.timer(is_timer_fork=True)
def run_job_contraction(job_tag, traj):
    #
    use_fsel_prop = get_param(job_tag, "measurement", "use_fsel_prop", default=True)
    #
    traj_gf = traj
    if is_test():
        # ADJUST ME
        traj_gf = 1000
        #
    #
    fns_produce = [
            f"{job_tag}/auto-contract/traj-{traj}/checkpoint.txt",
            #
            ]
    fns_need = [
            (f"{job_tag}/psel-prop-psrc-light/traj-{traj}.qar", f"{job_tag}/psel-prop-psrc-light/traj-{traj}/checkpoint.txt",),
            (f"{job_tag}/psel-prop-psrc-strange/traj-{traj}.qar", f"{job_tag}/psel-prop-psrc-strange/traj-{traj}/checkpoint.txt",),
            (f"{job_tag}/psel-prop-wsrc-light/traj-{traj}.qar", f"{job_tag}/psel-prop-wsrc-light/traj-{traj}/checkpoint.txt",),
            (f"{job_tag}/psel-prop-wsrc-strange/traj-{traj}.qar", f"{job_tag}/psel-prop-wsrc-strange/traj-{traj}/checkpoint.txt",),
            f"{job_tag}/gauge-transform/traj-{traj_gf}.field",
            f"{job_tag}/points-selection/traj-{traj}.lati",
            f"{job_tag}/field-selection/traj-{traj}.field",
            # f"{job_tag}/wall-src-info-light/traj-{traj}.txt",
            # f"{job_tag}/wall-src-info-strange/traj-{traj}.txt",
            # (f"{job_tag}/configs/ckpoint_lat.{traj}", f"{job_tag}/configs/ckpoint_lat.IEEE64BIG.{traj}",),
            ]
    if use_fsel_prop:
        fns_need += [
                (f"{job_tag}/prop-psrc-light/traj-{traj}.qar", f"{job_tag}/prop-psrc-light/traj-{traj}/geon-info.txt",),
                (f"{job_tag}/prop-psrc-strange/traj-{traj}.qar", f"{job_tag}/prop-psrc-strange/traj-{traj}/geon-info.txt",),
                (f"{job_tag}/prop-wsrc-light/traj-{traj}.qar", f"{job_tag}/prop-wsrc-light/traj-{traj}/geon-info.txt",),
                (f"{job_tag}/prop-wsrc-strange/traj-{traj}.qar", f"{job_tag}/prop-wsrc-strange/traj-{traj}/geon-info.txt",),
                ]
    if not check_job(job_tag, traj, fns_produce, fns_need):
        return
    #
    get_gf = None
    get_gt = run_gt(job_tag, traj_gf, get_gf)
    #
    get_f_weight = run_f_weight_uniform(job_tag, traj)
    get_f_rand_01 = run_f_rand_01(job_tag, traj)
    get_fsel_prob = run_fsel_prob(job_tag, traj, get_f_rand_01=get_f_rand_01, get_f_weight=get_f_weight)
    get_psel_prob = run_psel_prob(job_tag, traj, get_f_rand_01=get_f_rand_01, get_f_weight=get_f_weight)
    get_fsel = run_fsel_from_fsel_prob(get_fsel_prob)
    get_psel = run_psel_from_psel_prob(get_psel_prob)
    #
    prop_types = [
            "wsrc psel s",
            "wsrc psel l",
            "psrc psel s",
            "psrc psel l",
            # "rand_u1 fsel c",
            # "rand_u1 fsel s",
            # "rand_u1 fsel l",
            ]
    if use_fsel_prop:
        prop_types += [
                "wsrc fsel s",
                "wsrc fsel l",
                "psrc fsel s",
                "psrc fsel l",
                ]
    #
    get_get_prop = run_get_prop(
            job_tag, traj,
            get_gf = get_gf,
            get_gt = get_gt,
            get_psel = get_psel,
            get_fsel = get_fsel,
            prop_types = prop_types,
            )
    #
    run_r_list(job_tag)
    run_auto_contraction(job_tag, traj, get_get_prop=get_get_prop, get_psel_prob=get_psel_prob, get_fsel_prob=get_fsel_prob)
    #
    q.clean_cache()
    if q.obtained_lock_history_list:
        q.timer_display()

### ------

def get_all_cexpr():
   # benchmark_eval_cexpr(get_cexpr_meson_corr())
   # benchmark_eval_cexpr(get_cexpr_meson_jj())
   # benchmark_eval_cexpr(get_cexpr_pipi_corr())
   # benchmark_eval_cexpr(get_cexpr_pipi_jj())
    benchmark_eval_cexpr(get_cexpr_meson_corr_psnk_psrc())
   # benchmark_eval_cexpr(get_cexpr_pipi_corr_psnk_psrc())

### ------
set_param("48I", "traj_list")(list(range(1112,1493,10)) + list(range(1535,2176,10)))
set_param("48I", "measurement", "auto_contractor_chunk_size")(128)
set_param("48I", "measurement", "meson_tensor_t_sep")(12)
set_param("48I", "measurement", "pipi_op_t_sep")(2) #time separation between the two pions in a two pion operator. this is Delta
set_param("48I", "measurement", "pipi_op_dis_4d_sqr_limit")(6.0) #Minimum squared 4d distance between the two pion operators. We need to try with 9.0 and 16.0
set_param("48I", "measurement", "pipi_corr_t_sep_list")(list(range(1, 16))) #list of time separations between the two pion operators that we want to measure
set_param("48I", "measurement", "pipi_tensor_t_sep_list")([ 1, 2, ]) #not used
set_param("48I", "measurement", "pipi_tensor_t_max")(20) #not used
set_param("48I", "measurement", "pipi_tensor_r_max")(24) #not used
set_param("48I", "measurement", "use_fsel_prop")(False)

set_param("64I", "traj_list")(list(range(1200, 3000, 40)))
set_param("64I", "measurement", "meson_tensor_t_sep")(18)
set_param("64I", "measurement", "auto_contractor_chunk_size")(128)

# ----
#some more param settings for a test job
# ----

##################### CMD options #####################

job_tag_list_default = [
        "test-4nt8-checker",
        ]
job_tag_list_str_default = ",".join(job_tag_list_default)
job_tag_list = q.get_arg("--job_tag_list", default=job_tag_list_str_default).split(",")

is_performing_inversion = not q.get_option("--no-inversion")

is_performing_contraction = not q.get_option("--no-contraction")

#######################################################

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
    q.check_time_limit()
    get_all_cexpr()

    job_tag_traj_list = []
    for job_tag in job_tag_list:
        run_params(job_tag)
        traj_list = get_param(job_tag, "traj_list")
        for traj in traj_list:
            job_tag_traj_list.append((job_tag, traj,))
    if not is_test():
        # job_tag_traj_list = q.random_permute(job_tag_traj_list, q.RngState(f"{q.get_time()}"))
        job_tag_traj_list = q.get_comm().bcast(job_tag_traj_list)
    for job_tag, traj in job_tag_traj_list:
        if is_performing_inversion:
            q.check_time_limit()
            run_job_inversion(job_tag, traj)
            q.clean_cache()
            try_gracefully_finish()
    k_count = 0
    ncf = 10
    for job_tag, traj in job_tag_traj_list:
        if is_performing_contraction:
            q.check_time_limit()
            run_job_contraction(job_tag, traj)
            q.clean_cache()
            k_count += 1
            print(k_count)
            if k_count >= ncf:
                print(k_count)
                try_gracefully_finish()

    gracefully_finish()

# ----

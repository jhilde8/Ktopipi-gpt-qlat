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


# This script calculates all correlators necesary for pion scattering with the 48I sparsened propagators.
# For I=0, all four diagrams are active, along with including the sigma in the operator basis, meaning there are the 
# <sigma * sigma> terms, and all sigma pipi cross terms that must be evaluated. 
# This script outputs three .lat files for each configuration if all contraction functions are active
# one for the pipi-pipi, pipi-sigma, and sigma-sigma correlators, one for the vacuum expectation values, and one for the 3pt ATW correlators.  

#The layout of the file is as detailed below. 

#-------
# 1.  meson momentum projection functions
# 2.  point source meson correlator
#     + expression function
#     + momentum space contraction function
#     + position space contraction function
# 3.  wall source meson correlator
# 4.  pipi momentum projection functions
# 5.  pipi expression functions
#     + vev
#     + pipi-pipi and pipi-sigma 
# 6.  point source pipi correlator
#     + vev
#     + pipi-pipi and pipi-sigma
# 7.  ATW 3pt functions
#     + expressions
#     + contraction function
# 8.  run_auto_contraction
# 9.  run_job_inversion
# 10. run_job_contraction
# 11. ensemble params
# 12. gracefully_finish
#--------

#--------
#meson two point function momentum projections

def wave_function_mode_0(c12, size):
    return 1.0

def wave_function_mode_1(c12, size):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    w1 = np.cos(2.0 * np.pi * x / xs)
    w2 = np.cos(2.0 * np.pi * y / ys)
    w3 = np.cos(2.0 * np.pi * z / zs)
    w = (w1 + w2 + w3) / 3.0
    return w

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

def wave_function_mode_3(c12, size):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple() 
    w1 = np.cos((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
    w2 = np.cos((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) - (2.0 * np.pi * z / zs))
    w3 = np.cos((2.0 * np.pi * x / xs) - (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
    w4 = np.cos(-(2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs))
    w = (w1 + w2 + w3 + w4) / 4.0
    return w

#---------------

#---------------
#point source meson expressions

@q.timer
def get_cexpr_meson_corr_psnk_psrc():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_pos_psnk_psrc"
    def calc_cexpr():
        diagram_type_dict = dict()
        #diagram_type_dict[((('x_1', 'x_2'), 1), (('x_2', 'x_1'), 1))] = 'Type1'
        #diagram_type_dict[((('x_1', 'x_1'), 1), (('x_2', 'x_2'), 1))] = 'Type2'
        exprs = [
                mk_fac(1) + f"1",

                mk_pipi_i0('x_1','x_2') + f"pipi_i0(-tsep)",

                mk_pipi_i0('x_1','x_2',True) + f"pipi_i0^dag(0)",

                #(mk_sigma("x_2",True) * mk_sigma("x_1")
                #    + f"wf(0) * sigma^dag(0) * sigma_0(-tsep)",'Type1'),
                  
                #(mk_sigma("x_2",True) * mk_sigma("x_1")
                #+ f"wf(0) * sigma^dag(0) * sigma_0(-tsep)",'Type2'),
                
                #mk_sigma("x_2",True) * mk_sigma("x_1")
                #    + f"wf(0) * sigma^dag(0) * sigma(-tsep)",
            
                #mk_sigma("x_2",True) + f"sigma^dag(0)",
                #mk_sigma("x_1") + f"sigma(-tsep)",

                ]


        cexpr = contract_simplify_compile(
                *exprs,
                is_isospin_symmetric_limit=True,
                diagram_type_dict=diagram_type_dict,
                )
        return cexpr
    base_positions_dict = dict()
    #base_positions_dict["wave_function"] = wave_function
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )
#----------

#----------
#point source meson contractions (momentum space)
@q.timer(is_timer_fork=True)
def auto_contract_meson_corr_psnk_psrc_mom(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-I0D5-p/traj-{traj}/meson_corr_psnk_psrc_psel.lat"
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
    xg_fsel_arr = fsel.to_psel_local()[:]
    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    def load_data():
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):
            yield pidx
    @q.timer
    def feval(args):
        pidx = args
        xg_src = q.Coordinate(xg_psel_arr[pidx])
        prob_src = psel_prob_arr[pidx]
        values = np.zeros((total_site[3], len(expr_names),), dtype=np.complex128)
        for idx in range(len(xg_psel_arr)):
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
            
            #q.displayln_info(f"DEBUG: probabilities: {prob_list}")

            values[x_rel_t] += val/prob
        return values
    def sum_function(val_list):
        values = np.zeros((total_site[3], len(expr_names),), dtype=np.complex128)
        for val in val_list:
            values += val
        return values.transpose(1, 0)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    #q.displayln_info(f"DEBUG: normailzation factor: {total_volume}")
    res_sum *= 1.0 #/ (t_size * (total_volume / t_size))
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "t_sep", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))


#----------
#point source meson contractions (position space)
@q.timer(is_timer_fork=True)
def auto_contract_meson_corr_psnk_psrc_pos(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-avg-pos/traj-{traj}/meson_corr_psnk_psrc_psel.lat"
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
        #values_ind = np.zeros((total_site[0], total_site[1], total_site[2], total_site[3], len(expr_names),), dtype=np.complex128) 
        values_ind = np.zeros(((total_site[0]//2+1), (total_site[1]//2+1), (total_site[2]//2+1), total_site[3], len(expr_names),), dtype=np.complex128)

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
    res_sum *= 1.0 #/ (t_size * (total_volume / t_size))
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
#Wall source meson correlator

@q.timer
def get_cexpr_meson_corr():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_sigma_corr"
    def calc_cexpr():
        diagram_type_dict = dict()
        diagram_type_dict[((('x_1', 'x_2'), 1), (('x_2', 'x_1'), 1))] = 'Type1'
        diagram_type_dict[((('x_1', 'x_1'), 1), (('x_2', 'x_2'), 1))] = 'Type2'
        exprs = [
                mk_fac(1) + f"1",
                mk_sigma("x_2",True) * mk_sigma("x_1") + f"sigma^dag(0) * sigma(-tsep)",
                (mk_sigma("x_2",True) * mk_sigma("x_1") + f"sigma^dag(0) * sigma(-tsep)",'Type1'),
                (mk_sigma("x_2",True) * mk_sigma("x_1") + f"sigma^dag(0) * sigma(-tsep)",'Type2'),
                #mk_pi_0("x_2", True) * mk_pi_0("x_1")  + f"pi0^dag(0) * pi0(-tsep)",
                #(mk_eta("x_2", True)       * mk_eta("x_1")        + f"eta_dag(0) * eta(-tsep)",'Type2'),
                #(mk_eta_prime("x_2", True) * mk_eta_prime("x_1")  + f"eta_prime^dag(0) * eta_prime(-tsep)","Type2"),
                #(mk_eta("x_2",True)        * mk_eta_prime("x_1")  + f"eta_dag(0) * eta_prime(-tsep)","Type2"),
                #(mk_eta_prime("x_2",True)  * mk_eta("x_1")        + f"eta_prime^dag(0) * eta(-tsep)","Type2"),
                #mk_light("x_2",True) * mk_light("x_1") + f"light_meson^dag(0) * light_meson(-tsep)",
                #mk_strange("x_2",True) * mk_strange("x_1") + f"strage_meson^dag(0) * strange_meson(-tsep)",

                ]
        cexpr = contract_simplify_compile(*exprs, is_isospin_symmetric_limit=True, diagram_type_dict=diagram_type_dict)
        return cexpr
    return cache_compiled_cexpr(calc_cexpr, fn_base, is_cython=is_cython)

#wall-wall contraction
@q.timer(is_timer_fork=True)
def auto_contract_meson_corr(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-I0-D5/traj-{traj}/meson_corr.lat"
    if get_load_path(fn) is not None:
        return
    cexpr = get_cexpr_meson_corr()
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
    xg_fsel_arr = fsel.to_psel_local()[:]
    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    

    def load_data():
        t_t_list = q.get_mpi_chunk(
                [ (t_src, t_snk,) for t_snk in range(total_site[3]) for t_src in range(total_site[3]) ],
                rng_state=None)
        for t_src, t_snk in t_t_list:
            yield t_src, t_snk
    @q.timer
    def feval(args):
        t_src, t_snk = args
        t = (t_snk - t_src) % total_site[3]
        pd = {
                "x_2" : ("wall", t_snk,),
                "x_1" : ("wall", t_src,),
                "size" : total_site,
                }
        val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
        return val, t
    def sum_function(val_list):
        values = np.zeros((total_site[3], len(expr_names),), dtype=np.complex128)
        for val, t in val_list:
            values[t] += val
        return values.transpose(1, 0)
    auto_contractor_chunk_size = get_param(job_tag, "measurement", "auto_contractor_chunk_size", default=128)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=auto_contractor_chunk_size)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 / total_site[3]
    assert q.qnorm(res_sum[0] - 1.0) < 1e-10
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "t_sep", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

# ----

#----
#pipi wave function projections


@q.timer

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

fn_pipi_avg= "/home/jhildebrand28/ktopipi/analysis/pipi_avg_data.npy"
pipi_avg_arr = np.load(fn_pipi_avg)

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

def pipi_avg_sub(p1, p2, size):
    p1_tag, c1 = p1
    p2_tag, c2 = p2
    c1 = q.Coordinate(c1)
    c2 = q.Coordinate(c2)
    c12 = q.smod_coordinate(c1-c2,size)
    x_rel = c12[0]
    y_rel = c12[1]
    z_rel = c12[2]

    sub = pipi_avg_arr[abs(x_rel),abs(y_rel),abs(z_rel)]

    #we want to pick out the data that we want from the averaged term based on the positions in pd. 
    return sub

#-----
#pipi expressions, vacuum and 2 point function

@q.timer
def get_cexpr_pipi_vac():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_vev"
    def calc_cexpr():
        diagram_type_dict = dict()

        exprs = [
                mk_fac(1)+f"1",

                mk_sigma('x_1') + f"sigma(-tsep)",

                mk_sigma('x_1',True) + f"sigma^dag(0)",
                ]
        for mode in [0,1,2,3]:
            exprs += [
                        mk_fac(f"pipi_wave_function(x_1,x_2,{mode}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0('x_1','x_2') + f"wf_src({mode}) * pipi_i0(-tsep)",

                        mk_fac(f"pipi_wave_function(x_1,x_2,{mode}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0('x_1','x_2',True) + f"wf_snk({mode}) * pipi_i0^dag(0)",

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
 
@q.timer
def get_cexpr_pipi_dc_sub():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_dc_sub"
    def calc_cexpr():
        diagram_type_dict = dict()

        exprs = [
                mk_fac(1)+f"1",
                ]
        for mode in [0]:
            exprs += [
                        mk_fac(f"pipi_wave_function(x_1,x_2,{mode}, size, pipi_op_dis_4d_sqr_limit)")
                        * (mk_pipi_i0('x_1','x_2') - mk_fac(f"pipi_avg_sub(x_1,x_2,size)")) + f"wf_src({mode}) * pipi_i0(-tsep) - <pipi>",

                        mk_fac(f"pipi_wave_function(x_1,x_2,{mode}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0('x_1','x_2') + f"wf_snk({mode}) * pipi_i0^dag(0)",

                    ]

        cexpr = contract_simplify_compile(
                *exprs,
                is_isospin_symmetric_limit=True,
                diagram_type_dict=diagram_type_dict,
                )
        return cexpr
    base_positions_dict = dict()
    base_positions_dict["pipi_wave_function"] = pipi_wave_function
    base_positions_dict["pipi_avg_sub"] = pipi_avg_sub
    base_positions_dict["pipi_op_dis_4d_sqr_limit"] = 0.5 # default value, to be overrided by `pd`.
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )

def get_cexpr_pipi_vac_pos_avg():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_vev_pos_avg"
    def calc_cexpr():
        diagram_type_dict = dict()

        exprs = [
                mk_fac(1)+f"1",

                mk_pipi_i0('x_1','x_2') + f"pipi_i0(-tsep)",

                mk_pipi_i0('x_1','x_2',True) + f"pipi_i0^dag(0)",

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

@q.timer
def get_cexpr_pipi_corr_psnk_psrc():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_I0_psnk_psrc"
    def calc_cexpr():
        diagram_type_dict = dict() #the auto contractor deals with each term within each type, along with prefactors present in the sum. 
        #pipi-sigma
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT1_pps'
        diagram_type_dict[((('snk_1', 'snk_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'src_1'), 1))] = 'ADT2_pps'

        #sigma-pipi
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'snk_1'), 1), (('src_1', 'src_1'), 1))] = 'ADT1_spp'
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'src_1'), 1), (('src_1', 'snk_1'), 1))] = 'ADT2_spp'
        
        #pipi-pipi
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'snk_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'src_1'), 1))] = 'ADT1'
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'src_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT2'
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('snk_2', 'src_2'), 1), (('src_1', 'snk_1'), 1), (('src_2', 'snk_2'), 1))] = 'ADT3'
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('snk_2', 'src_2'), 1), (('src_1', 'snk_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT4'

        exprs = [
                mk_fac(1) + f"1",
                
                #sigma vevs
                #mk_sigma("src_1") + f"sigma(-tsep)",

                #mk_sigma("src_1",True) + f"sigma^dag(0)",

                #sigma correlator
                mk_sigma("src_1",True) * mk_sigma("snk_1") + f"sigma^dag(0) * sigma(-tsep)",

                ]
        for mode_src in [ 0, 1, 2, 3,]:
            exprs += [
                         #pipi-sigma cross terms
                         (mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_sigma('snk_1',True) * mk_pipi_i0('src_1','src_2') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i00(-tsep)",'ADT1_pps'),

                        (mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_sigma('snk_1',True) * mk_pipi_i0('src_1','src_2') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i00(-tsep)",'ADT2_pps'),

                        (mk_fac(f"pipi_wave_function(snk_1,snk_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0('snk_1','snk_2',True) * mk_sigma('src_1') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i00(-tsep)",'ADT1_spp'),
                   
                        (mk_fac(f"pipi_wave_function(snk_1,snk_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0('snk_1','snk_2',True) * mk_sigma('src_1') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i00(-tsep)",'ADT2_spp'),
                        ]
            for mode_snk in [ 0, 1, 2, 3,]:
                exprs += [
                        #I=0 pipi-pipi terms
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i00^dag(0) * pipi_i00(-tsep)",'ADT1'),

                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i00^dag(0) * pipi_i00(-tsep)",'ADT2'),
                        
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i00^dag(0) * pipi_i00(-tsep)",'ADT3'),
                        
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i00^dag(0) * pipi_i00(-tsep)",'ADT4'),
                        #---
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

#-----
#pipi vacuum contraction functions

def auto_contract_pipi_vev_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-dc-sub/traj-{traj}/pipi_vev.lat"
    if get_load_path(fn) is not None:
        return

    #cexpr = get_cexpr_pipi_vac()
    cexpr = get_cexpr_pipi_dc_sub()
    expr_names = get_expr_names(cexpr)
    total_site = q.Coordinate(get_param(job_tag, "total_site"))
    t_size = total_site[3]
    get_prop = get_get_prop()
    psel_prob = get_psel_prob()
    fsel_prob = get_fsel_prob()
    psel = psel_prob.psel
    fsel = fsel_prob.fsel
    if not fsel.is_containing(psel):
        q.display_info(-1, f"WARNING: fsel is not containing psel. The probability weighting may be wrong.")

    fsel_n_elems = fsel.n_elems
    fsel_prob_arr = fsel_prob[:].ravel()
    psel_prob_arr = psel_prob[:].ravel()
    xg_psel_arr = psel[:]
    xg_fsel_arr = fsel.to_psel_local()[:]
    pidx_list_list = [[] for i in range(t_size)]

    for pidx in range(len(xg_psel_arr)):
        xg = xg_psel_arr[pidx]
        pidx_list_list[xg[3]].append(pidx)

    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    pipiop_tsep = get_param(job_tag, "measurement", "pipi_op_t_sep")
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")

    def load_data_single():
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):
            yield pidx

    def load_data_block():
        lsize = 8
        pidx_list = []
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):
            pidx_list.append(pidx)
            if len(pidx_list) == lsize:
                yield pidx_list
                pidx_list = []

        if pidx_list:
            yield pidx_list

    def feval_single(args): 
        pidx = args
        xg_src = q.Coordinate(xg_psel_arr[pidx])
        t_src = xg_src[3]
        prob_src = psel_prob_arr[pidx]

        values = np.zeros((len(expr_names)),dtype=np.complex128)
        
        t_src_2 = (t_src + pipiop_tsep) % t_size #forward pipiop_tsep. This is important for constructing the subtraction term. 
        for pidx_src_2 in pidx_list_list[t_src_2]:
            xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
            prob = psel_prob_arr[pidx_src_2] * psel_prob_arr[pidx]

            pd = {
                    "x_1": ("point", xg_src.to_tuple(),),
                    "x_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }

            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

            values += val/prob

        return values, t_src

    def sum_function(val_list):
        values = np.zeros((t_size, len(expr_names),),dtype=np.complex128)
        for val, t_src in val_list:
            values[t_src] += val
        return values.transpose(1,0,)

    res_sum = q.parallel_map_sum(feval_single, load_data_single(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 #/ (t_size * (total_volume / t_size)) #normalization. change as needed.
    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names,],
        ["t_src", t_size, [str(t) for t in range(t_size)],],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))


def auto_contract_pipi_vev_pos_avg(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-pos-avg-t/traj-{traj}/pipi_vev_pos_avg.lat"
    if get_load_path(fn) is not None:
        return

    cexpr = get_cexpr_pipi_vac_pos_avg()
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
        q.display_info(-1, f"WARNING: fsel is not containing psel. The probability weighting may be wrong.")

    fsel_n_elems = fsel.n_elems
    fsel_prob_arr = fsel_prob[:].ravel()
    psel_prob_arr = psel_prob[:].ravel()
    xg_psel_arr = psel[:]
    xg_fsel_arr = fsel.to_psel_local()[:]
    pidx_list_list = [[] for i in range(t_size)]

    for pidx in range(len(xg_psel_arr)):
        xg = xg_psel_arr[pidx]
        pidx_list_list[xg[3]].append(pidx)

    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    pipiop_tsep = get_param(job_tag, "measurement", "pipi_op_t_sep")
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")

    def load_data_single():
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):
            yield pidx

    def load_data_block():
        lsize = 8
        pidx_list = []
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):
            pidx_list.append(pidx)
            if len(pidx_list) == lsize:
                yield pidx_list
                pidx_list = []

        if pidx_list:
            yield pidx_list

    def feval_single(args): 
        pidx = args
        xg_src = q.Coordinate(xg_psel_arr[pidx])
        t_src = xg_src[3]
        prob_src = psel_prob_arr[pidx]

        values = np.zeros((len(expr_names)),dtype=np.complex128)
        
        t_src_2 = (t_src + pipiop_tsep) % t_size #forward pipiop_tsep. This is important for constructing the subtraction term. 
        for pidx_src_2 in pidx_list_list[t_src_2]:
            xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
            prob = psel_prob_arr[pidx_src_2] * psel_prob_arr[pidx]

            x_rel = q.smod_coordinate(xg_src_2 - xg_src, total_site)
            assert x_rel[3] == pipiop_tsep 

            pd = {
                    "x_1": ("point", xg_src.to_tuple(),),
                    "x_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }

            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

            values += val/prob

        return values, x_rel, t_src

    def sum_function_avg(val_list):
        values = np.zeros((x_size//2+1, y_size//2+1, z_size//2+1,t_size, len(expr_names),),dtype=np.complex128)
        for val, x_rel, t_src in val_list:
            values[abs(x_rel[0]), abs(x_rel[1]), abs(x_rel[2]),t_src] += val
        return values.transpose(4,0,1,2,3)

    res_sum_avg = q.parallel_map_sum(feval_single, load_data_single(), sum_function=sum_function_avg, chunksize=1)
    res_sum_avg = q.glb_sum(res_sum_avg)
    res_sum_avg *= 1.0 #/ (t_size * (total_volume / t_size)) #normalization. change as needed.
   
    #constructs the average vev with the data we just global summed
    #counter = res_sum_avg[0,:,:,:]
    #mask = counter != 0
    #avg_pipi = np.zeros((len(expr_names), x_size//2+1, y_size//2+1, z_size//2+1),dtype=np.complex128)

    #avg_pipi[:,mask] = res_sum_avg[:,mask]/counter[mask] 

    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names,],
        ["x_rel", x_size//2+1, [str(x) for x in range(x_size//2+1)],],
        ["y_rel", y_size//2+1, [str(y) for y in range(y_size//2+1)],],
        ["z_rel", z_size//2+1, [str(z) for z in range(z_size//2+1)],],
        ["t_src", t_size, [str(t) for t in range(t_size)],],
        ])
    ld.from_numpy(res_sum_avg)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))



def auto_contract_pipi_vev_pos_sub(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-sub-abs/traj-{traj}/pipi_vev_pos.lat"
    if get_load_path(fn) is not None:
        return

    cexpr = get_cexpr_pipi_dc_sub()
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
        q.display_info(-1, f"WARNING: fsel is not containing psel. The probability weighting may be wrong.")

    fsel_n_elems = fsel.n_elems
    fsel_prob_arr = fsel_prob[:].ravel()
    psel_prob_arr = psel_prob[:].ravel()
    xg_psel_arr = psel[:]
    xg_fsel_arr = fsel.to_psel_local()[:]
    pidx_list_list = [[] for i in range(t_size)]

    for pidx in range(len(xg_psel_arr)):
        xg = xg_psel_arr[pidx]
        pidx_list_list[xg[3]].append(pidx)

    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    pipiop_tsep = get_param(job_tag, "measurement", "pipi_op_t_sep")
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")

    def load_data_single():
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):
            yield pidx

    def load_data_block():
        lsize = 8
        pidx_list = []
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):
            pidx_list.append(pidx)
            if len(pidx_list) == lsize:
                yield pidx_list
                pidx_list = []

        if pidx_list:
            yield pidx_list

    def feval_single(args): 
        pidx = args
        xg_src = q.Coordinate(xg_psel_arr[pidx])

        t_src = xg_src[3]
        prob_src = psel_prob_arr[pidx]

        values = np.zeros((len(expr_names)),dtype=np.complex128)
        
        t_src_2 = (t_src + pipiop_tsep) % t_size #forward pipiop_tsep. This is important for constructing the subtraction term. 
        for pidx_src_2 in pidx_list_list[t_src_2]:
            xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
            prob = psel_prob_arr[pidx_src_2] * psel_prob_arr[pidx]

            x_rel = q.smod_coordinate(xg_src_2 - xg_src, total_site)
            assert x_rel[3] == pipiop_tsep 

            pd = {
                    "x_1": ("point", xg_src.to_tuple(),),
                    "x_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }


            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

        
            #q.displayln_info(f"DEBUG: Subtracted bubble value: {val[1]}")

            q.displayln_info(f"DEBUG: unsubtracted bubble value: {val[2] - val[1]}")

            values += val/prob

        return values, x_rel, t_src

    def sum_function(val_list):
        values = np.zeros((x_size//2+1, y_size//2+1, z_size//2+1, t_size, len(expr_names),),dtype=np.complex128)
        for val, x_rel, t_src in val_list:
            values[abs(x_rel[0]), abs(x_rel[1]), abs(x_rel[2]), t_src] += val
        return values.transpose(4,0,1,2,3)

    res_sum = q.parallel_map_sum(feval_single, load_data_single(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 

    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names,],
        ["x_rel", x_size//2+1, [str(x) for x in range(x_size//2+1)],],
        ["y_rel", y_size//2+1, [str(y) for y in range(y_size//2+1)],],
        ["z_rel", z_size//2+1, [str(z) for z in range(z_size//2+1)],],
        ["t_src", t_size, [str(t) for t in range(t_size)],],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))
            

# ----- 
# two pion two point function contractions

@q.timer(is_timer_fork=True)
def auto_contract_pipi_corr_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-I0D5-p/traj-{traj}/pipi_corr_psnk_psrc.lat"
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
        pipiop_tsep = [5]
        for pipi_op_t_sep in pipiop_tsep: #iterate over possible internal pion separations, using the same internal sep for source and sink operator
            t_src_2 = (t_src - pipi_op_t_sep) % t_size
            #for pipi_op_t_sep_snk in pipiop_tsep:
            t_snk_2 = (t_snk + pipi_op_t_sep) % t_size
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
                (#len(pipiop_tsep),
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
            assert pipi_op_t_sep_snk == pipi_op_t_sep_src
            pd = {
                    "snk_1": ("point", xg_snk.to_tuple(),),
                    "snk_2": ("point", xg_snk_2.to_tuple(),),
                    "src_1": ("point", xg_src.to_tuple(),),
                    "src_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values += val/prob
        return values, t_sep_idx, data_list_idx, data_list_size
    def sum_function(val_list):
        pipiop_tsep = [5]
        values = np.zeros(
                (len(pipi_corr_t_sep_list), 
                 #len(pipiop_tsep),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        for val, t_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size // 1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")
            values[t_sep_idx] += val
        return values.transpose(1, 0,)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 #/ (t_size * (total_volume / t_size) * (total_volume / t_size))
    pipiop_tsep = [5]
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "t_sep", len(pipi_corr_t_sep_list), pipi_corr_t_sep_list, ],
        #[ "pipi_op_t_sep_snk", len(pipiop_tsep), [ str(i+1) for i in range(len(pipiop_tsep)) ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))


#----------
# 3 point ATW matrix element <pi | O_2pi | pi>

@q.timer
def get_cexpr_pipi_3ptATW_corr_psrc_psnk():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_3ptATW_corr_psnk_psrc"
    def calc_cexpr():
        diagram_type_dict = dict()
        exprs = [
                mk_fac(1) + f"1",
                ]
        for mode_src in [ 0, 1, 2, 3, ]:
            for mode_snk in [ 0, 1, 2, 3,]:
                exprs += [
                        # <pi0(t_1+t_2) * 2pi(t_2) * pi0(0)^dag>
                        mk_fac(f"wave_function(snk,src,{mode_snk},size)")
                        * mk_fac(f"pipi_wave_function(int_1,int_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pi_0("src", True) * mk_pipi_i0("int_1","int_2") * mk_pi_0("snk")
                        + f"wf({mode_snk}) * wf({mode_src}) * pi0^dag(0) * pipi_i20(-t_int) * pi0(-t)",
                        #
                        ]
        cexpr = contract_simplify_compile(
                *exprs,
                is_isospin_symmetric_limit=True,
                diagram_type_dict=diagram_type_dict,
                )
        return cexpr
    base_positions_dict = dict()
    base_positions_dict["wave_function"] = wave_function
    base_positions_dict["pipi_wave_function"] = pipi_wave_function
    base_positions_dict["pipi_op_dis_4d_sqr_limit"] = 0.5 # default value, to be overrided by `pd`.
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )

@q.timer(is_timer_fork=True)
def auto_contract_ATW3pt_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract/traj-{traj}/pipi_ATW_psnk_psrc.lat"
    if get_load_path(fn) is not None:
        return

    cexpr = get_cexpr_pipi_3ptATW_corr_psrc_psnk()
    expr_names = get_expr_names(cexpr)
    total_site = q.Coordinate(get_param(job_tag, "total_site"))
    t_size = total_site[3]
    get_prop = get_get_prop()
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

    pidx_list_list = [ [] for i in range(t_size) ] 
    for pidx in range(len(xg_psel_arr)):
        xg = xg_psel_arr[pidx]
        pidx_list_list[xg[3]].append(pidx) #time component of every point

    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    
    #params
    pipi_op_tsep = get_param(job_tag, "measurement", "pipi_op_t_sep")
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")
    ATW3pt_tsep = get_param(job_tag, "measurement", "ATW3pt_src_snk_t_sep")#24
    #pipi_int_tsep = list(range(0,ATW3pt_tsep)) #range from [1,23] for 2pi operator insertion
    pipi_corr_t_sep_list = get_param(job_tag, "measurement", "pipi_corr_t_sep_list") # list([1,24])
    pipi_int_tsep = pipi_corr_t_sep_list
    data_list = []
    pipiop_tsep = [1,2]

    #iterate through all source positions
    for pidx_src in range(len(xg_psel_arr)):
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_src = xg_src[3]
        #then iterate through all intermediate times, setting both sink and intermediate locations
        for t_sep_idx,t_sep in enumerate(pipi_int_tsep):
            assert t_sep > 0
            t_diff = (ATW3pt_tsep - t_sep) #time separation between operator insertion and sink with a set t_1 + t_2
            t_int = (t_src + t_sep) % t_size #time sep between source and operator insetion
            t_snk = (t_int + t_diff) % t_size

            #assert abs(t_snk - t_src) % t_size == ATW3pt_tsep #ensure that the distance between source and sink is a constant

            #iterating over all indices with t_int as their time separation
            for pidx_int in pidx_list_list[t_int]:
                xg_int = q.Coordinate(xg_psel_arr[pidx_int]) #full coordinate
                assert xg_int[3] == t_int
                if pidx_int == pidx_src:
                    continue

                for pidx_snk in pidx_list_list[t_snk]:
                    xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
                    assert xg_snk[3] == t_snk
                    if pidx_snk == pidx_src:
                        continue

                    #save the tuple of the source, intermediate, and sink location along with
                    #the time separation from source to int for this index config.
                    data_list.append((pidx_snk,pidx_int,pidx_src,t_sep_idx))

    #loads in the point data for each worker
    def load_data():
        data_list_chunk = q.get_mpi_chunk(data_list)
        data_list_size = len(data_list_chunk)
        for data_list_idx, (pidx_snk, pidx_int, pidx_src, t_sep_idx) in enumerate(data_list_chunk):
            yield data_list_idx, data_list_size, pidx_snk, pidx_int, pidx_src, t_sep_idx

    @q.timer
    def feval(args):
        data_list_idx, data_list_size, pidx_snk, pidx_int, pidx_src, t_sep_idx = args
        assert pidx_src != pidx_snk
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_int = q.Coordinate(xg_psel_arr[pidx_int])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_int = xg_int[3]
        t_src = xg_src[3]
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_int] * psel_prob_arr[pidx_src]
        pidx_int_2_list = []
        pipiop_tsep = [1,2]
        for pipi_op_t_sep in pipiop_tsep:
            t_int_2 = (t_int - pipi_op_t_sep) % t_size
            for pidx_int_2 in pidx_list_list[t_int_2]:
                if pidx_int_2 in  [pidx_snk,pidx_src,pidx_int]:
                    continue

                prob2 = psel_prob_arr[pidx_int_2]
                prob = prob1 * prob2
                pidx_int_2_list.append((pidx_int_2,prob,))

        values = np.zeros(
                (len(pipiop_tsep),
                 len(expr_names)), dtype=np.complex128,
                )

        for pidx_int_2, prob in pidx_int_2_list:
            xg_int_2 = q.Coordinate(xg_psel_arr[pidx_int_2])
            t_int_2 = xg_int_2[3]

            tsep_int = (t_int - t_int_2) % t_size

            pd = {
                    "snk": ("point", xg_snk.to_tuple(),),
                    "int_1": ("point", xg_int.to_tuple(),),
                    "int_2": ("point", xg_int_2.to_tuple(),),
                    "src": ("point", xg_src.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

            values[(tsep_int-1)] += val/prob
            #values += val
        return values, t_sep_idx, data_list_idx, data_list_size

    def sum_function(val_list):
        pipiop_tsep = [1,2]
        values = np.zeros(
                (len(pipi_int_tsep),
                 len(pipiop_tsep),
                 len(expr_names),
                ),
                dtype=np.complex128,)

        for val, t_sep_idx, data_list_idx, data_list_size in val_list:
            print(type(val))
            if data_list_idx % (data_list_size //1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")

            values[t_sep_idx] += val
        return values.transpose(2,0,1,)

    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 /(t_size * (total_volume / t_size) * (total_volume / t_size))
    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names, ],
        ["t2_sep", len(pipi_int_tsep), pipi_int_tsep, ],
        [ "pipi_op_t_sep", len(pipiop_tsep), [ str(i) for i in pipiop_tsep ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append("f{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ls '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

#----------
@q.timer(is_timer_fork=True)
def run_auto_contraction(
        job_tag, traj,
        *,
        get_get_prop,
        get_psel_prob,
        get_fsel_prob,
        ):
    fname = q.get_fname()
    fn_checkpoint = f"{job_tag}/auto-contract-pipi-pos-avg-t/traj-{traj}/checkpoint.txt"
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

    #meson contraction functions
    #auto_contract_meson_corr_psnk_psrc_pos(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob) 
    #auto_contract_meson_corr_psnk_psrc_mom(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob) 
    
    #pipi contraction functions
    #auto_contract_pipi_vev_psnk_psrc(job_tag, traj, get_get_prop,get_psel_prob, get_fsel_prob) 
    auto_contract_pipi_vev_pos_avg(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob) 
    #auto_contract_pipi_vev_pos_sub(job_tag, traj, get_get_prop,get_psel_prob, get_fsel_prob)
    #auto_contract_pipi_corr_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)

    #ATW 3pt function
    #auto_contract_ATW3pt_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob) 
    
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
            f"{job_tag}/auto-contract-pipi-pos-avg-t/traj-{traj}/checkpoint.txt",
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

#def get_all_cexpr():
   # benchmark_eval_cexpr(get_cexpr_meson_corr())
   # benchmark_eval_cexpr(get_cexpr_meson_jj())
   # benchmark_eval_cexpr(get_cexpr_pipi_corr())
   # benchmark_eval_cexpr(get_cexpr_pipi_jj())
   #benchmark_eval_cexpr(get_cexpr_meson_corr_psnk_psrc())
   # benchmark_eval_cexpr(get_cexpr_pipi_corr_psnk_psrc())

### ------
set_param("48I", "traj_list")(list(range(1865, 2176,10))) #list(range(1102,1492,10)) + list(range(1505, 1636, 10)) + list(range(1705, 2116,10)) + list(range(1005, 1096, 10)))
set_param("48I", "measurement", "auto_contractor_chunk_size")(128)
set_param("48I", "measurement", "meson_tensor_t_sep")(12)
set_param("48I", "measurement", "pipi_op_t_sep")(5) #time separation between the two pions in a two pion operator. this is Delta
set_param("48I", "measurement", "pipi_op_dis_4d_sqr_limit")(25.0) #Minimum squared 4d distance between the two pion operators. We need to try with 9.0 and 16.0
set_param("48I", "measurement", "pipi_corr_t_sep_list")(list(range(1, 21))) #list of time separations between the two pion operators that we want to measure
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
    #get_all_cexpr()

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
    ncf = 30
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

#!/usr/bin/env python3

import functools
import math
import os
import time
import importlib
import sys

#specific to webserver. I can just edit the code where I need to on BNL. 
#sys.path.insert(0,'/home/jhildebrand28')
#import auto_contractor
#import qlat_scripts

import qlat_gpt as qg

#from qlat_scripts import *
#from auto_contractor.operators import *
#---

#these imports will remain on BNL
from qlat_scripts.v1 import *
from auto_contractor.operators import *

is_cython = not is_test()

# ----

load_path_list[:] = [
        "results",
        "qcddata",
        "/data1/qcddata2",
        "/data2/qcddata3-prop",
        "/data1/qcddata3",
        "/hpcgpfs01/scratch/jhildebra/psrc_props",
        "/hpcgpfs01/work/lqcd/staging/RBC/qcddata/MDWF/2+1f/48nt96/IWASAKI/b2.13/ls24b+c2/M1.8/ms0.0362/mu0.00078/jhildebra",
        ]

# ----


# ----------------
# meson prefactors
# ----------------

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

def wave_function_mode_1_exp(c12, size):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    w1 = np.exp(-1j*(2.0 * np.pi * x / xs))
    w2 = np.exp(-1j*(2.0 * np.pi * y / xs))
    w3 = np.exp(-1j*(2.0 * np.pi * z / xs))
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

def wave_function_mode_2_exp(c12, size):
    x, y, z, t, = c12.to_tuple()
    xs, ys, zs, ts, = size.to_tuple()
    w1 = np.exp(-1j*((2.0 * np.pi * x / xs) + (2.0 * np.pi * y / ys)))
    w2 = np.exp(-1j*((2.0 * np.pi * y / ys) + (2.0 * np.pi * z / zs)))
    w3 = np.exp(-1j*((2.0 * np.pi * z / zs) + (2.0 * np.pi * x / xs)))

    w4 = np.exp(-1j*((2.0 * np.pi * x / zs) - (2.0 * np.pi * y / xs)))
    w5 = np.exp(-1j*((2.0 * np.pi * y / zs) - (2.0 * np.pi * z / xs)))
    w6 = np.exp(-1j*((2.0 * np.pi * z / zs) - (2.0 * np.pi * x / xs)))
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

def wave_function_mode_3_exp(c12, size):
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
wave_function_mode_dict[1] = wave_function_mode_1
wave_function_mode_dict[2] = wave_function_mode_2
wave_function_mode_dict[3] = wave_function_mode_3

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


def smearing_function_ident(c12,r):
    return 1.0

def smearing_function_hydrogen(c12,r):
    x, y, z, t = c12.to_tuple()
    s = np.exp(-(np.sqrt((x*x + y*y + z*z))/r))    
    return s


smearing_function_type_dict = dict()
smearing_function_type_dict["I"] = smearing_function_ident
smearing_function_type_dict["h"] = smearing_function_hydrogen

def smearing_function(p1, p2, rad, size, sm_type):
    p1_tag, c1 = p1
    p2_tag, c2 = p2
    c1 = q.Coordinate(c1)
    c2 = q.Coordinate(c2)
    c12 = q.smod_coordinate(c1 - c2, size)

    if sm_type not in smearing_function_type_dict:
        fname = q.get_fname()
        raise Exception(f"{fname}: {p1} {p2} {sm_type} {size}")
    sf = smearing_function_type_dict[sm_type]
    return sf(c12,rad)


# -------------------------
# meson expression function
# -------------------------

@q.timer
def get_cexpr_meson_corr_psnk_psrc():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_meson_corr_psnk_psrc"
    def calc_cexpr():
        diagram_type_dict = dict()
        diagram_type_dict[((('x_1', 'x_2'), 1), (('x_2', 'x_1'), 1))] = 'Type1'
        diagram_type_dict[((('x_1', 'x_1'), 1), (('x_2', 'x_2'), 1))] = None
        exprs = [
                mk_fac(1) + f"1",

                #mk_fac(f"wave_function(x_1, x_2, {mode}, size)")
                #mk_k_0("x_2", True)     * mk_k_0("x_1")
                #+ f"K0^dag(0) * K0(-tsep)",
            
                #mk_sigma("x_2",True) * mk_sigma("x_1")
                #+ f"sigma^dag(0) * sigma(-tsep)"
            
                ]
        for mode in [0,1,2,3]:
            exprs += [
                    mk_fac(f"wave_function(x_1,x_2,{mode},size)")
                    * mk_pi_0("x_2",True) * mk_pi_0("x_1")
                    + f"pi0^dag(0) * pi0(-tsep)",
                    
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
    base_positions_dict["pipi_op_dis_4d_sqr_limit"] = 0.0 # default value, to be overrided by `pd`.
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )

# ---------------
# meson corr fsel
# ---------------
@q.timer(is_timer_fork=True)
def auto_contract_meson_corr_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/meson_corr_psnk_psrc.lat"
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

    #if there is a bottleneck in the global sum, we can do multiple sources in feval at a time.
    def load_data_mult():
        lsize = 16 #how many indices per task
        pidx_list = []
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):

            pidx_list.append(pidx)
            if len(pidx_list) == lsize:
                yield pidx_list
                pidx_list = []

        if pidx_list:
            yield pidx_list

    @q.timer
    def feval_mult(args):
        pidx_list = args #this is a list now. We must iteratre over it along with iterating over sinks
        values = np.zeros((total_site[3], total_site[3], len(expr_names),), dtype=np.complex128) #accumulation over all sources in this call

        for pidx in pidx_list: #iterate through the index list we passed in
            xg_src = q.Coordinate(xg_psel_arr[pidx])
            prob_src = psel_prob_arr[pidx]

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


    def load_data():
        for pidx in range(len(xg_psel_arr)):
            yield pidx

    @q.timer
    def feval(args):
        pidx = args
        xg_src = q.Coordinate(xg_psel_arr[pidx])
        t_src = xg_src[3]
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
        return (values, t_src)

    def sum_function(val_list):
        values = np.zeros((total_site[3], total_site[3], len(expr_names),), dtype=np.complex128)
        k = 0
        for val in val_list:
            k+=1
            q.displayln_info(-1,f"sum element {k}")
            values[val[1]] += val[0]
        return values.transpose(2, 1, 0)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "t_sep", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        [ "t_src", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

# ---------------
# meson corr psel
# ---------------

@q.timer(is_timer_fork=True)
def auto_contract_meson_corr_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/meson_corr_psnk_psrc_psel.lat"
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
        t_src = xg_src[3]
        prob_src = psel_prob_arr[pidx]
        values = np.zeros((total_site[3], len(expr_names),), dtype=np.complex128)
        for idx in range(len(xg_psel_arr)):
            xg_snk = q.Coordinate(xg_psel_arr[idx])
            t_snk = xg_snk[3]
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
            values[x_rel_t] += val / prob
        return (values,t_src)
    
    def sum_function(val_list):
        values = np.zeros((total_site[3], total_site[3], len(expr_names),), dtype=np.complex128)
        k = 0
        for val in val_list:
            k+=1
            q.displayln_info(-1,f"sum element {k}")
            values[val[1]] += val[0]
        return values.transpose(2, 1, 0)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "t_sep", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        [ "t_src", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))
    
#-------

# -------
# position space meson correlator.
# -------

@q.timer(is_timer_fork=True)
def auto_contract_meson_corr_psnk_psrc_psel_pos(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-V/traj-{traj}/meson_corr_pos.lat"
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
    def load_data():
        lsize = 16 #how many indices per task
        pidx_list = []
        for pidx in q.get_mpi_chunk(list(range(len(xg_psel_arr)))):

            pidx_list.append(pidx)
            if len(pidx_list) == lsize:
                yield pidx_list
                pidx_list = []

        if pidx_list:
            yield pidx_list

    @q.timer
    def feval(args):
        values = np.zeros((total_site[0],total_site[1],total_site[2],total_site[3], len(expr_names),), dtype=np.complex128)
        pidx_list = args #this is a list now. We must iteratre over it along with iterating over sinks
        for pidx in pidx_list: #iterate through the index list we passed in
            xg_src = q.Coordinate(xg_psel_arr[pidx])
            prob_src = psel_prob_arr[pidx]

            for pidx in range(len(xg_psel_arr)): #use all randomly selected points as sinks
                xg_snk = q.Coordinate(xg_psel_arr[pidx])
                if xg_snk == xg_src:
                    prob_snk = 1.0
                else:
                    prob_snk = psel_prob_arr[pidx]
                prob = prob_src * prob_snk
                x_rel = q.smod_coordinate(xg_snk - xg_src, total_site)
                x_rel_t = x_rel[3]
                pd = {
                        "x_2" : ("point", xg_src.to_tuple(),),
                        "x_1" : ("point", xg_snk.to_tuple(),),
                        "size" : total_site,
                     }
                val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

                #averaging over +/- relative coordinate configurations. 
                values[x_rel[0],x_rel[1],x_rel[2],x_rel_t] += val

            #values += values_ind

        return values

    def sum_function(val_list):
        values = np.zeros((total_site[0],total_site[1],total_site[2],total_site[3], len(expr_names),), dtype=np.complex128)
        for val in val_list:
            values += val
           # valtp = np.transpose(values) #this probably takes a while
        return values.transpose(4,0,1,2,3)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 / (t_size * (total_volume / t_size))
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "x_sep", x_size, [ str(x) for x in range(x_size) ], ],
        [ "y_sep", y_size, [ str(y) for y in range(y_size) ], ],
        [ "z_sep", z_size, [ str(z) for z in range(z_size) ], ],
        [ "t_sep", t_size, [ str(q.rel_mod(t, t_size)) for t in range(t_size) ], ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

# -----
    
# ----------------------------
# two pion two point functions
# ----------------------------

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


# --------------------------------
#pipi expressions. I=2, I=0, and V
# --------------------------------

@q.timer
def get_cexpr_pipi_corr_psnk_psrc_psel():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_psnk_psrc_psel"
    def calc_cexpr():
        diagram_type_dict = dict() #the auto contractor deals with each term within each type, along with prefactors present in the sum. 
        #pipi-pipi
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'snk_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'src_1'), 1))] = 'ADT1' #V
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'src_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT2'
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('snk_2', 'src_2'), 1), (('src_1', 'snk_1'), 1), (('src_2', 'snk_2'), 1))] = 'ADT3'
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('snk_2', 'src_2'), 1), (('src_1', 'snk_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT4'

        #pipi-sigma
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT5_pps'
        diagram_type_dict[((('snk_1', 'snk_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'src_1'), 1))] = 'ADT6_pps' #V

        #sigma-pipi
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'src_1'), 1), (('src_1', 'snk_1'), 1))] = 'ADT5_spp' 
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'snk_1'), 1), (('src_1', 'src_1'), 1))] = 'ADT6_spp' #V

        exprs = [
                mk_fac(1) + f"1",
                
                ]
        for mode_src in [0,1,2,3]:
            
             #pipi-sigma cross terms
            (mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
            * mk_sigma('snk_1',True) * mk_pipi_i0('src_1','src_2') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i00(-tsep)",'ADT5_pps'),

            (mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
            * mk_sigma('snk_1',True) * mk_pipi_i0('src_1','src_2') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i00(-tsep)",'ADT6_pps'),#V

            (mk_fac(f"pipi_wave_function(snk_1,snk_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
            * mk_pipi_i0('snk_1','snk_2',True) * mk_sigma('src_1') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i00(-tsep)",'ADT5_spp'),
       
            (mk_fac(f"pipi_wave_function(snk_1,snk_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
            * mk_pipi_i0('snk_1','snk_2',True) * mk_sigma('src_1') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i00(-tsep)",'ADT6_spp'), #V
            
            for mode_snk in [0,1,2,3]:
                exprs += [
                        #Direct, I=2
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i20("snk_1", "snk_2", True)
                        * mk_pipi_i20("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i20^dag(0) * pipi_i20(-tsep)",'ADT3'),

                        #Cross, I=2
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i20("snk_1", "snk_2", True)
                        * mk_pipi_i20("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i20^dag(0) * pipi_i20(-tsep)",'ADT4'),
                        
                        #Total, I=2
                        mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i20("snk_1", "snk_2", True) #true refers to the is_dagger boolean
                        * mk_pipi_i20("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i20^dag(0) * pipi_i20(-tsep)",
                        
                        #Direct, I=0
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i20^dag(0) * pipi_i20(-tsep)",'ADT3'),

                        #Cross, I=0
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i20("snk_1", "snk_2", True)
                        * mk_pipi_i20("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i20^dag(0) * pipi_i20(-tsep)",'ADT4'),

                        #Rectangle
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i20^dag(0) * pipi_i20(-tsep)",'ADT2'),

                        #Vacuum
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i20^dag(0) * pipi_i20(-tsep)",'ADT1'),

                        #Total, I=0
                        mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True) #true refers to the is_dagger boolean
                        * mk_pipi_i0("src_1", "src_2")
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
    base_positions_dict["pipi_op_dis_4d_sqr_limit"] = 0.0 # default value, to be overrided by `pd`.
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )


# -----
# load in arrays for vacuum subtraction
# -----

# load in data for the improved subtraction. ADJUST ME
fn_pipi_avg = "/home/jhildebrand28/ktopipi/data/pipi_psel_vevs/"

#fn_pipi_avg = "/hpcgpfs01/scratch/jhildebra/vev_data/"

sigma_avg = np.load(fn_pipi_avg + "sigma_avg.npy")

pipi_vev_D5 = np.load(fn_pipi_avg + "pi_vev_avg_D5.npy") #shape (4,25,25,25)
pipi_vev_D7 = np.load(fn_pipi_avg + "pi_vev_avg_D7.npy") #shape (4,25,25,25)
pipi_vev_D9 = np.load(fn_pipi_avg + "pi_vev_avg_D9.npy") #shape (4,25,25,25)

pipi_vev_all = np.stack((pipi_vev_D5, pipi_vev_D7, pipi_vev_D9),axis=1)

#average over spatial coordinates
def sigma_avg_sub(p1):
    p1_tag,c1=p1
    c1 = q.Coordinate(c1)
    t = c1[3]

    sub = sigma_avg[t]
    return sub
    
def pipi_avg_sub(p1, p2, size, mode, Delta_idx):
    p1_tag, c1 = p1
    p2_tag, c2 = p2
    c1 = q.Coordinate(c1)
    c2 = q.Coordinate(c2)
    c12 = q.smod_coordinate(c1-c2,size)
    x_rel = c12[0]
    y_rel = c12[1]
    z_rel = c12[2]
    
    #this should always be the first term, the second source and sink are calculated such that the actual tsep is between the first source and sink. 
    t_src = c1[3]

    if mode > 3 or mode < 0:
        q.displayln_info(-1,f"ERROR: invalid momentum mode")

    #q.displayln_info(f"DEBUG: time==0 if statement entered, time={t2}")
    return pipi_vev_all[mode, Delta_idx, abs(x_rel),abs(y_rel),abs(z_rel)]

# -----
    

@q.timer
def get_cexpr_pipi_corr_psnk_psrc_V():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_corr_psnk_psrc_V"
    def calc_cexpr():
        diagram_type_dict = dict()
        diagram_type_dict[((('x_1', 'x_2'), 1), (('x_2', 'x_1'), 1))] = 'Type1'
        diagram_type_dict[((('x_1', 'x_1'), 1), (('x_2', 'x_2'), 1))] = None
        exprs = [
                mk_fac(1) + f"1",

                #improved vacuum subtraction with sigma using psel->psel vev
                mk_sigma('x_1') - mk_fac(f"sigma_avg_sub(x_1)") + f"sigma(-tsep) - <sigma>",
            
                ]
        for mode in [0,1,2,3]:
            exprs += [
                        # I=0 V term
                        mk_fac(f"pipi_wave_function(x_1,x_2,{mode}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0('x_1','x_2') + f"wf_src({mode}) * pipi_i0(-tsep)",
                
                        #improved vac sub using psel -> psel vev.
                        (mk_fac(f"pipi_wave_function(x_1,x_2,{mode}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0('x_1','x_2')) - mk_fac(f"pipi_avg_sub(x_1,x_2,size,{mode},Delta_idx)") + f"wf_src({mode}) * pipi_i0(-tsep) - <pipi>",        

                
                    ]
        cexpr = contract_simplify_compile(
                *exprs,
                is_isospin_symmetric_limit=True,
                diagram_type_dict=diagram_type_dict,
                )
        return cexpr
    base_positions_dict = dict()
    base_positions_dict["pipi_wave_function"] = pipi_wave_function
    base_positions_dict["sigma_avg_sub"]= sigma_avg_sub
    base_positions_dict["pipi_avg_sub"] = pipi_avg_sub
    base_positions_dict["pipi_op_dis_4d_sqr_limit"] = 0.5 # default value, to be overrided by `pd`.
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )

@q.timer
def get_cexpr_pipi_3ptATW_corr_psrc_psnk():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_3ptATW_corr_psnk_psrc"
    def calc_cexpr():
        diagram_type_dict = dict()
        diagram_type_dict[()] = 'ADT0'
        diagram_type_dict[((('int_1', 'snk'), 1), (('int_2', 'src'), 1), (('snk', 'int_1'), 1), (('src', 'int_2'), 1))] = 'ADT1'
        diagram_type_dict[((('int_1', 'snk'), 1), (('int_2', 'src'), 1), (('snk', 'int_2'), 1), (('src', 'int_1'), 1))] = 'ADT2'
        diagram_type_dict[((('int_1', 'int_2'), 1), (('int_2', 'int_1'), 1), (('snk', 'src'), 1), (('src', 'snk'), 1))] = 'ADT3' #V
        diagram_type_dict[((('int_1', 'int_2'), 1), (('int_2', 'snk'), 1), (('snk', 'src'), 1), (('src', 'int_1'), 1))] = 'ADT4'
        diagram_type_dict[((('int_1', 'int_2'), 1), (('int_2', 'src'), 1), (('snk', 'int_1'), 1), (('src', 'snk'), 1))] = 'ADT5'
        diagram_type_dict[((('snk', 'src'), 1), (('src', 'snk'), 1))] = 'ADT6'
        exprs = [
                mk_fac(1) + f"1",
                ]
        for mode_src in [ 0, 1,2,3,]:
            for mode_snk in [ 0, 1, 2, 3,]:
                exprs += [
                        # <pi0(t_1+t_2) * 2pi(t_2)_I2 * pi0(0)^dag>
                        #no vacuum contam, so we just look at the total.
                        mk_fac(f"wave_function(snk,src,{mode_snk},size)")
                        * mk_fac(f"pipi_wave_function(int_1,int_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pi_0("src", True) * mk_pipi_i20("int_1","int_2") * mk_pi_0("snk")
                        + f"wf({mode_snk}) * wf({mode_src}) * pi0^dag(0) * pipi_i20(-t_int) * pi0(-t)",

                        # <pi0(t_1+t_2) * 2pi(t_2)_I0 * pi0(0)^dag>
                        mk_fac(f"wave_function(snk,src,{mode_snk},size)")
                        * mk_fac(f"pipi_wave_function(int_1,int_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pi_0("src", True) * mk_pipi_i0("int_1","int_2") * mk_pi_0("snk")
                        + f"wf({mode_snk}) * wf({mode_src}) * pi0^dag(0) * pipi_i0(-t_int) * pi0(-t)",

                        # <pi0(t_1+t_2) * 2pi(t_2)_I0 * pi0(0)^dag>
                        (mk_fac(f"wave_function(snk,src,{mode_snk},size)")
                        * mk_fac(f"pipi_wave_function(int_1,int_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pi_0("src", True) * mk_pipi_i0("int_1","int_2") * mk_pi_0("snk")
                        + f"wf({mode_snk}) * wf({mode_src}) * pi0^dag(0) * pipi_i0(-t_int) * pi0(-t)",['ADT1','ADT2','ADT4','ADT5']),

                        # <pi0(t_1+t_2) * 2pi(t_2)_I0 * pi0(0)^dag>
                        #(mk_fac(f"wave_function(snk,src,{mode_snk},size)")
                        #* mk_fac(f"pipi_wave_function(int_1,int_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        #* mk_pi_0("src", True) * mk_pipi_i0("int_1","int_2") * mk_pi_0("snk")
                        #+ f"wf({mode_snk}) * wf({mode_src}) * pi0^dag(0) * pipi_i0(-t_int) * pi0(-t)",'ADT3'),

                        # <pi0(t_1+t_2) * 2pi(t_2)_I0 * pi0(0)^dag>
                        #(mk_fac(f"wave_function(snk,src,{mode_snk},size)") * mk_pi_0("src", True)
                        #* ((mk_fac(f"pipi_wave_function(int_1,int_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        #* mk_pipi_i0("int_1","int_2")) - mk_fac(f"pipi_avg_sub(int_1, int_2, size, {mode_src},Delta_idx)"))
                        #* mk_pi_0("snk")
                        #+ f"wf({mode_snk}) * wf({mode_src}) * pi0^dag(0) * pipi_i0(-t_int) * pi0(-t)",['ADT3','ADT6']),
                    
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
    base_positions_dict["pipi_avg_sub"] = pipi_avg_sub
    base_positions_dict["pipi_op_dis_4d_sqr_limit"] = 0.5 # default value, to be overrided by `pd`.
    return cache_compiled_cexpr(
            calc_cexpr,
            fn_base,
            is_cython=is_cython,
            base_positions_dict=base_positions_dict,
            )

#-----

# -----------------------------------------
# two pion two point function contractions.
# -----------------------------------------

# Direct, Rectangle, and Cross diagrams will have a single fsel point at a time, and we will average
# over the results where the second sink is the fsel point, and the second source is the fsel point.
# Vacuum has two fsel points since it is the noisiest and can be reconstructed from vacuum bubbles. 

# ----
# fsel 
# ----

#contraction function for vacuum diagram. This will be done one bubble at a time, with the correlation happening after. 
def auto_contract_pipi_corr_psnk_psrc_V(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/pipi_vev.lat"
    if get_load_path(fn) is not None:
        return

    cexpr = get_cexpr_pipi_corr_psnk_psrc_V()
    #cexpr = get_cexpr_pipi_dc_sub()
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
    fidx_list_list = [[] for i in range(t_size)]

    for pidx in range(len(xg_psel_arr)):
        xg = xg_psel_arr[pidx]
        pidx_list_list[xg[3]].append(pidx)

    for idx in range(len(xg_fsel_arr)):
        xg = xg_fsel_arr[idx]
        fidx_list_list[xg[3]].append(idx)

    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")
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

        values = np.zeros((len(pipiop_tsep_list), len(expr_names)),dtype=np.complex128)

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
        
            t_src_2 = (t_src + pipiop_tsep) % t_size #forward pipiop_tsep. This is important for constructing the subtraction term. 
            for idx_src_2 in fidx_list_list[t_src_2]:
                xg_src_2 = q.Coordinate(xg_fsel_arr[idx_src_2])
                prob = fsel_prob_arr[idx_src_2] * psel_prob_arr[pidx]
                x_rel = q.smod_coordinate(xg_src_2 - xg_src, total_site)
    
                pd = {
                        "x_1": ("point", xg_src.to_tuple(),),
                        "x_2": ("point-snk", xg_src_2.to_tuple(),),
                        "size": total_site,
                        "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                        }
    
                val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
    
                values[pipiop_tsep_idx] += val/prob

        return values, t_src

    def sum_function(val_list):
        values = np.zeros((t_size, len(pipiop_tsep_list), len(expr_names),),dtype=np.complex128)
        for val, t_src in val_list:
            values[t_src] += val
        return values.transpose(2,1,0,)

    res_sum = q.parallel_map_sum(feval_single, load_data_single(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 #/ (t_size * (total_volume / t_size)) #normalization. change as needed.
    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names,],
        ["pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list],
        ["t_src", t_size, [str(t) for t in range(t_size)],],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))


@q.timer(is_timer_fork=True)
def auto_contract_ATW3pt_psnk_psrc1(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/pipi_ATW_psnk_psrc1.lat"
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

    fidx_list_list = [ [] for i in range(t_size) ] 
    for idx in range(len(xg_fsel_arr)):
        xg = xg_fsel_arr[idx]
        fidx_list_list[xg[3]].append(idx) #time component of every point

    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    
    #params
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")#Delta. this is a list
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")
    snk_src_tsep_list = get_param(job_tag, "measurement", "tsep_snk_src_3pt")#list of fixed source-sink separations. eg. [12,16,20]

    data_list = []

    #iterate through all source positions
    for pidx_src in range(len(xg_psel_arr)):
        
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_src = xg_src[3]

        pipi_int_tsep = list(range(1,max(snk_src_tsep_list))) 
        #then iterate through all intermediate times, setting both sink and intermediate locations
        for t_sep_idx,t_sep in enumerate(pipi_int_tsep):
            assert t_sep > 0
            t_int = (t_src + t_sep) % t_size # timeslice of operator insertion

            #iterating over all indices with t_int as their time separation
            for pidx_int in pidx_list_list[t_int]:
                xg_int = q.Coordinate(xg_psel_arr[pidx_int]) #full coordinate
                assert xg_int[3] == t_int
                if pidx_int == pidx_src:
                    continue

                #we are iterating through a list like [12,16,20,24]. Since we are finding t_int first
                #based on the max number in this list, there will be cases where t_int > t_snk, which we do not want
                #so we only find sink points where t_int + Delta (full extent of two pion operator) is before t_snk. 

                #put this in feval?? Delta isn't defined yet and I don't think i want to iterate over it here.
                for ss_sep_idx, ss_sep in enumerate(snk_src_tsep_list): 
                    t_snk = (t_src + ss_sep) % t_size
                    
                    if t_sep >= (ss_sep-1):
                        continue
                    
                    
                    for pidx_snk in pidx_list_list[t_snk]:
                        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
                        assert xg_snk[3] == t_snk
                        if pidx_snk == pidx_src or pidx_snk == pidx_int:
                            continue

                        #save the tuple of the source, intermediate, and sink location along with
                        #the time separation from source to int for this index config.
                        data_list.append((pidx_snk,pidx_int,pidx_src,t_sep_idx, ss_sep_idx))

    #loads in the point data for each worker
    def load_data():
        data_list_chunk = q.get_mpi_chunk(data_list)
        data_list_size = len(data_list_chunk)
        for data_list_idx, (pidx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx) in enumerate(data_list_chunk):
            yield data_list_idx, data_list_size, pidx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx

    @q.timer
    def feval(args):
        data_list_idx, data_list_size, pidx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx = args
        assert pidx_src != pidx_snk
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_int = q.Coordinate(xg_psel_arr[pidx_int])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_int = xg_int[3]
        t_src = xg_src[3]
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_int] * psel_prob_arr[pidx_src]
        idx_int_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_int_2 = (t_int + pipiop_tsep) % t_size

            #tsep = t_int1 - t_src
            tsep_tint2 = (pipi_int_tsep[t_sep_idx] + pipiop_tsep) % t_size
            tint2_limit = (snk_src_tsep_list[ss_sep_idx] - 1) % t_size
            if tsep_tint2 > tint2_limit:
                continue
        
            for idx_int_2 in fidx_list_list[t_int_2]:
                if idx_int_2 in  [pidx_snk,pidx_src,pidx_int]:
                    continue
    
                prob2 = fsel_prob_arr[idx_int_2]
                prob = prob1 * prob2
                idx_int_2_list.append((idx_int_2,prob,pipiop_tsep_idx))

        values = np.zeros((
                 len(pipiop_tsep_list), len(expr_names)), dtype=np.complex128,
                )

        for idx_int_2, prob, pipiop_tsep_idx in idx_int_2_list:
            xg_int_2 = q.Coordinate(xg_fsel_arr[idx_int_2])
            t_int_2 = xg_int_2[3]

            tsep_int = (t_int - t_int_2) % t_size
            assert tsep_int == pipiop_tsep_list[pipiop_tsep_idx] or tsep_int == t_size - pipiop_tsep_list[pipiop_tsep_idx] 

            pd = {
                    "snk": ("point", xg_snk.to_tuple(),),
                    "int_1": ("point", xg_int.to_tuple(),),
                    "int_2": ("point-snk", xg_int_2.to_tuple(),),
                    "src": ("point", xg_src.to_tuple(),),
                    "size": total_site,
                    "Delta_idx": pipiop_tsep_idx,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

            values[pipiop_tsep_idx] += val/prob
            #values += val
        return values, t_sep_idx, ss_sep_idx, data_list_idx, data_list_size

    def sum_function(val_list):
        values = np.zeros(
                (len(snk_src_tsep_list), #t1 + t2, indexed by ss_tsep_idx
                 max(snk_src_tsep_list), #t2, indexed by t_sep_idx
                 len(pipiop_tsep_list),
                 len(expr_names),
                ),
                dtype=np.complex128,)

        for val, t_sep_idx, ss_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size //1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")

            values[ss_sep_idx,t_sep_idx] += val
        return values.transpose(3,2,0,1) #(nexpr, nDelta, n_ss_tsep, n_int_tsep)

    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names, ],
        ["pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        ["snk_src_tsep", len(snk_src_tsep_list), snk_src_tsep_list, ],
        ["t2_sep", max(snk_src_tsep_list), list(range(1,max(snk_src_tsep_list))), ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append("f{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ls '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

    
def auto_contract_ATW3pt_psnk_psrc2(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/pipi_ATW_psnk_psrc2.lat"
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

    fidx_list_list = [ [] for i in range(t_size) ] 
    for idx in range(len(xg_fsel_arr)):
        xg = xg_fsel_arr[idx]
        fidx_list_list[xg[3]].append(idx) #time component of every point

    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    
    #params
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")#Delta. this is a list
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")
    snk_src_tsep_list = get_param(job_tag, "measurement", "tsep_snk_src_3pt")#list of fixed source-sink separations. eg. [12,16,20]

    data_list = []

    #iterate through all source positions
    for pidx_src in range(len(xg_psel_arr)):
        
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_src = xg_src[3]

        pipi_int_tsep = list(range(1,max(snk_src_tsep_list))) 
        #then iterate through all intermediate times, setting both sink and intermediate locations
        for t_sep_idx,t_sep in enumerate(pipi_int_tsep):
            assert t_sep > 0
            t_int = (t_src + t_sep) % t_size # timeslice of operator insertion

            #iterating over all indices with t_int as their time separation
            for idx_int in fidx_list_list[t_int]:
                xg_int = q.Coordinate(xg_fsel_arr[idx_int]) #full coordinate
                assert xg_int[3] == t_int
                if idx_int == pidx_src:
                    continue

                for ss_sep_idx, ss_sep in enumerate(snk_src_tsep_list): 
                    t_snk = (t_src + ss_sep) % t_size
                    
                    if t_sep >= (ss_sep-1):
                        continue
                    
                    for pidx_snk in pidx_list_list[t_snk]:
                        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
                        assert xg_snk[3] == t_snk
                        if pidx_snk == pidx_src or pidx_snk == idx_int:
                            continue

                        data_list.append((pidx_snk,idx_int,pidx_src,t_sep_idx, ss_sep_idx))

    def load_data():
        data_list_chunk = q.get_mpi_chunk(data_list)
        data_list_size = len(data_list_chunk)
        for data_list_idx, (pidx_snk, idx_int, pidx_src, t_sep_idx, ss_sep_idx) in enumerate(data_list_chunk):
            yield data_list_idx, data_list_size, pidx_snk, idx_int, pidx_src, t_sep_idx, ss_sep_idx

    @q.timer
    def feval(args):
        data_list_idx, data_list_size, pidx_snk, idx_int, pidx_src, t_sep_idx, ss_sep_idx = args
        assert pidx_src != pidx_snk
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_int = q.Coordinate(xg_fsel_arr[idx_int])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_int = xg_int[3]
        t_src = xg_src[3]
        prob1 = psel_prob_arr[pidx_snk] * fsel_prob_arr[idx_int] * psel_prob_arr[pidx_src]
        pidx_int_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_int_2 = (t_int + pipiop_tsep) % t_size

            #tsep = t_int1 - t_src
            tsep_tint2 = (pipi_int_tsep[t_sep_idx] + pipiop_tsep) % t_size
            tint2_limit = (snk_src_tsep_list[ss_sep_idx] - 1) % t_size
            if tsep_tint2 > tint2_limit:
                continue
        
            for pidx_int_2 in pidx_list_list[t_int_2]:
                if pidx_int_2 in  [pidx_snk,pidx_src,idx_int]:
                    continue
    
                prob2 = psel_prob_arr[pidx_int_2]
                prob = prob1 * prob2
                pidx_int_2_list.append((pidx_int_2,prob,pipiop_tsep_idx))

        values = np.zeros((
                 len(pipiop_tsep_list), len(expr_names)), dtype=np.complex128,
                )

        for pidx_int_2, prob, pipiop_tsep_idx in pidx_int_2_list:
            xg_int_2 = q.Coordinate(xg_psel_arr[pidx_int_2])
            t_int_2 = xg_int_2[3]

            tsep_int = (t_int - t_int_2) % t_size
            assert tsep_int == pipiop_tsep_list[pipiop_tsep_idx] or tsep_int == t_size - pipiop_tsep_list[pipiop_tsep_idx] 

            pd = {
                    "snk": ("point", xg_snk.to_tuple(),),
                    "int_1": ("point-snk", xg_int.to_tuple(),),
                    "int_2": ("point", xg_int_2.to_tuple(),),
                    "src": ("point", xg_src.to_tuple(),),
                    "size": total_site,
                    "Delta_idx": pipiop_tsep_idx,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

            values[pipiop_tsep_idx] += val/prob
            #values += val
        return values, t_sep_idx, ss_sep_idx, data_list_idx, data_list_size

    def sum_function(val_list):
        values = np.zeros(
                (len(snk_src_tsep_list), #t1 + t2, indexed by ss_tsep_idx
                 max(snk_src_tsep_list), #t2, indexed by t_sep_idx
                 len(pipiop_tsep_list),
                 len(expr_names),
                ),
                dtype=np.complex128,)

        for val, t_sep_idx, ss_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size //1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")

            values[ss_sep_idx,t_sep_idx] += val
        return values.transpose(3,2,0,1) #(nexpr, nDelta, n_ss_tsep, n_int_tsep)

    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names, ],
        ["pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        ["snk_src_tsep", len(snk_src_tsep_list), snk_src_tsep_list, ],
        ["t2_sep", max(snk_src_tsep_list), list(range(1,max(snk_src_tsep_list))), ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append("f{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ls '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))


@q.timer(is_timer_fork=True)
def auto_contract_ATW3pt_psnk_psrc3(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-ATW-test/traj-{traj}/pipi_ATW_psnk_psrc3.lat"
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

    fidx_list_list = [ [] for i in range(t_size) ] 
    for idx in range(len(xg_fsel_arr)):
        xg = xg_fsel_arr[idx]
        fidx_list_list[xg[3]].append(idx) #time component of every point

    geo = q.Geometry(total_site)
    total_volume = geo.total_volume
    
    #params
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")#Delta. this is a list
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")
    snk_src_tsep_list = get_param(job_tag, "measurement", "tsep_snk_src_3pt")#list of fixed source-sink separations. eg. [12,16,20]

    data_list = []

    for pidx_src in range(len(xg_psel_arr)):
        
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_src = xg_src[3]

        pipi_int_tsep = list(range(1,max(snk_src_tsep_list))) 
        for t_sep_idx,t_sep in enumerate(pipi_int_tsep):
            assert t_sep > 0
            t_int = (t_src + t_sep) % t_size # timeslice of operator insertion

            for pidx_int in pidx_list_list[t_int]:
                xg_int = q.Coordinate(xg_psel_arr[pidx_int]) #full coordinate
                assert xg_int[3] == t_int
                if pidx_int == pidx_src:
                    continue

                for ss_sep_idx, ss_sep in enumerate(snk_src_tsep_list): 
                    t_snk = (t_src + ss_sep) % t_size
                    
                    if t_sep >= (ss_sep-1):
                        continue
                    
                    
                    for idx_snk in fidx_list_list[t_snk]:
                        xg_snk = q.Coordinate(xg_fsel_arr[idx_snk])
                        assert xg_snk[3] == t_snk
                        if idx_snk == pidx_src or pidx_snk == pidx_int:
                            continue

                        #save the tuple of the source, intermediate, and sink location along with
                        #the time separation from source to int for this index config.
                        data_list.append((idx_snk,pidx_int,pidx_src,t_sep_idx, ss_sep_idx))

    #loads in the point data for each worker
    def load_data():
        data_list_chunk = q.get_mpi_chunk(data_list)
        data_list_size = len(data_list_chunk)
        for data_list_idx, (idx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx) in enumerate(data_list_chunk):
            yield data_list_idx, data_list_size, idx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx

    @q.timer
    def feval(args):
        data_list_idx, data_list_size, idx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx = args
        assert pidx_src != pidx_snk
        xg_snk = q.Coordinate(xg_fsel_arr[idx_snk])
        xg_int = q.Coordinate(xg_psel_arr[pidx_int])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_int = xg_int[3]
        t_src = xg_src[3]
        prob1 = fsel_prob_arr[idx_snk] * psel_prob_arr[pidx_int] * psel_prob_arr[pidx_src]
        pidx_int_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_int_2 = (t_int + pipiop_tsep) % t_size

            #tsep = t_int1 - t_src
            tsep_tint2 = (pipi_int_tsep[t_sep_idx] + pipiop_tsep) % t_size
            tint2_limit = (snk_src_tsep_list[ss_sep_idx] - 1) % t_size
            if tsep_tint2 > tint2_limit:
                continue
        
            for pidx_int_2 in pidx_list_list[t_int_2]:
                if pidx_int_2 in  [pidx_snk,pidx_src,pidx_int]:
                    continue
    
                prob2 = psel_prob_arr[pidx_int_2]
                prob = prob1 * prob2
                pidx_int_2_list.append((pidx_int_2,prob,pipiop_tsep_idx))

        values = np.zeros((
                 len(pipiop_tsep_list), len(expr_names)), dtype=np.complex128,
                )

        for pidx_int_2, prob, pipiop_tsep_idx in pidx_int_2_list:
            xg_int_2 = q.Coordinate(xg_psel_arr[pidx_int_2])
            t_int_2 = xg_int_2[3]

            tsep_int = (t_int - t_int_2) % t_size
            assert tsep_int == pipiop_tsep_list[pipiop_tsep_idx] or tsep_int == t_size - pipiop_tsep_list[pipiop_tsep_idx] 

            pd = {
                    "snk": ("point-snk", xg_snk.to_tuple(),),
                    "int_1": ("point", xg_int.to_tuple(),),
                    "int_2": ("point", xg_int_2.to_tuple(),),
                    "src": ("point", xg_src.to_tuple(),),
                    "size": total_site,
                    "Delta_idx": pipiop_tsep_idx,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

            values[pipiop_tsep_idx] += val/prob
            #values += val
        return values, t_sep_idx, ss_sep_idx, data_list_idx, data_list_size

    def sum_function(val_list):
        values = np.zeros(
                (len(snk_src_tsep_list), #t1 + t2, indexed by ss_tsep_idx
                 max(snk_src_tsep_list), #t2, indexed by t_sep_idx
                 len(pipiop_tsep_list),
                 len(expr_names),
                ),
                dtype=np.complex128,)

        for val, t_sep_idx, ss_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size //1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")

            values[ss_sep_idx,t_sep_idx] += val
        return values.transpose(3,2,0,1) #(nexpr, nDelta, n_ss_tsep, n_int_tsep)

    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names, ],
        ["pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        ["snk_src_tsep", len(snk_src_tsep_list), snk_src_tsep_list, ],
        ["t2_sep", max(snk_src_tsep_list), list(range(1,max(snk_src_tsep_list))), ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append("f{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ls '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

# ----

# ----
# psel
# ----

def auto_contract_pipi_corr_psnk_psrc_psel_V(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/pipi_vev_psel.lat"
    if get_load_path(fn) is not None:
        return

    cexpr = get_cexpr_pipi_corr_psnk_psrc_V()
    #cexpr = get_cexpr_pipi_dc_sub()
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
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")
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

        values = np.zeros((len(pipiop_tsep_list), len(expr_names)),dtype=np.complex128)

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
        
            t_src_2 = (t_src + pipiop_tsep) % t_size #forward pipiop_tsep. This is important for constructing the subtraction term. 
            for pidx_src_2 in pidx_list_list[t_src_2]:
                xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
                prob = psel_prob_arr[pidx_src_2] * psel_prob_arr[pidx]
                x_rel = q.smod_coordinate(xg_src_2 - xg_src, total_site)
    
                pd = {
                        "x_1": ("point", xg_src.to_tuple(),),
                        "x_2": ("point", xg_src_2.to_tuple(),),
                        "size": total_site,
                        "Delta_idx": pipiop_tsep_idx,
                        "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                        }
    
                val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
    
                values[pipiop_tsep_idx] += val/prob

        return values, t_src

    def sum_function(val_list):
        values = np.zeros((t_size, len(pipiop_tsep_list), len(expr_names),),dtype=np.complex128)
        for val, t_src in val_list:
            values[t_src] += val
        return values.transpose(2,1,0,)

    res_sum = q.parallel_map_sum(feval_single, load_data_single(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    res_sum *= 1.0 #/ (t_size * (total_volume / t_size)) #normalization. change as needed.
    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names,],
        ["pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list],
        ["t_src", t_size, [str(t) for t in range(t_size)],],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))
    

#full pipi contractions are done in the psel case, no reconstruction needed. 
@q.timer(is_timer_fork=True)
def auto_contract_pipi_corr_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-V/traj-{traj}/pipi_corr_psnk_psrc_psel.lat"
    if get_load_path(fn) is not None:
        return
    cexpr = get_cexpr_pipi_corr_psnk_psrc_psel()
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

        values = np.zeros(
                (#len(pipiop_tsep),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        
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
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "t_sep", len(pipi_corr_t_sep_list), pipi_corr_t_sep_list, ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))


@q.timer(is_timer_fork=True)
def auto_contract_ATW3pt_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/pipi_ATW_psnk_psrc_psel.lat"
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
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")#Delta. this is a list
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")
    snk_src_tsep_list = get_param(job_tag, "measurement", "tsep_snk_src_3pt")#list of fixed source-sink separations. eg. [12,16,20]
    #pipi_int_tsep = list(range(1,ATW3pt_tsep)) #range from [1,23] for 2pi operator insertion

    data_list = []

    #iterate through all source positions
    for pidx_src in range(len(xg_psel_arr)):
        
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_src = xg_src[3]

        pipi_int_tsep = list(range(1,max(snk_src_tsep_list))) 
        #then iterate through all intermediate times, setting both sink and intermediate locations
        for t_sep_idx,t_sep in enumerate(pipi_int_tsep):
            assert t_sep > 0
            t_int = (t_src + t_sep) % t_size # timeslice of operator insertion

            #iterating over all indices with t_int as their time separation
            for pidx_int in pidx_list_list[t_int]:
                xg_int = q.Coordinate(xg_psel_arr[pidx_int]) #full coordinate
                assert xg_int[3] == t_int
                if pidx_int == pidx_src:
                    continue

                #we are iterating through a list like [12,16,20,24]. Since we are finding t_int first
                #based on the max number in this list, there will be cases where t_int > t_snk, which we do not want
                #so we only find sink points where t_int + Delta (full extent of two pion operator) is before t_snk. 

                #put this in feval?? Delta isn't defined yet and I don't think i want to iterate over it here.
                for ss_sep_idx, ss_sep in enumerate(snk_src_tsep_list): 
                    t_snk = (t_src + ss_sep) % t_size
                    
                    if t_sep >= (ss_sep-1):
                        continue
                    
                    
                    for pidx_snk in pidx_list_list[t_snk]:
                        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
                        assert xg_snk[3] == t_snk
                        if pidx_snk == pidx_src or pidx_snk == pidx_int:
                            continue

                        #save the tuple of the source, intermediate, and sink location along with
                        #the time separation from source to int for this index config.
                        data_list.append((pidx_snk,pidx_int,pidx_src,t_sep_idx, ss_sep_idx))

    #loads in the point data for each worker
    def load_data():
        data_list_chunk = q.get_mpi_chunk(data_list)
        data_list_size = len(data_list_chunk)
        for data_list_idx, (pidx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx) in enumerate(data_list_chunk):
            yield data_list_idx, data_list_size, pidx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx

    @q.timer
    def feval(args):
        data_list_idx, data_list_size, pidx_snk, pidx_int, pidx_src, t_sep_idx, ss_sep_idx = args
        assert pidx_src != pidx_snk
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_int = q.Coordinate(xg_psel_arr[pidx_int])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_int = xg_int[3]
        t_src = xg_src[3]
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_int] * psel_prob_arr[pidx_src]
        pidx_int_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_int_2 = (t_int + pipiop_tsep) % t_size

            #tsep = t_int1 - t_src
            tsep_tint2 = (pipi_int_tsep[t_sep_idx] + pipiop_tsep) % t_size
            tint2_limit = (snk_src_tsep_list[ss_sep_idx] - 1) % t_size
            if tsep_tint2 > tint2_limit:
                continue
        
            for pidx_int_2 in pidx_list_list[t_int_2]:
                if pidx_int_2 in  [pidx_snk,pidx_src,pidx_int]:
                    continue
    
                prob2 = psel_prob_arr[pidx_int_2]
                prob = prob1 * prob2
                pidx_int_2_list.append((pidx_int_2,prob,pipiop_tsep_idx))

        values = np.zeros((
                 len(pipiop_tsep_list), len(expr_names)), dtype=np.complex128,
                )

        for pidx_int_2, prob, pipiop_tsep_idx in pidx_int_2_list:
            xg_int_2 = q.Coordinate(xg_psel_arr[pidx_int_2])
            t_int_2 = xg_int_2[3]

            tsep_int = (t_int - t_int_2) % t_size
            assert tsep_int == pipiop_tsep_list[pipiop_tsep_idx] or tsep_int == t_size - pipiop_tsep_list[pipiop_tsep_idx] 

            pd = {
                    "snk": ("point", xg_snk.to_tuple(),),
                    "int_1": ("point", xg_int.to_tuple(),),
                    "int_2": ("point", xg_int_2.to_tuple(),),
                    "src": ("point", xg_src.to_tuple(),),
                    "size": total_site,
                    "Delta_idx": pipiop_tsep_idx,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)

            values[pipiop_tsep_idx] += val/prob
            #values += val
        return values, t_sep_idx, ss_sep_idx, data_list_idx, data_list_size

    def sum_function(val_list):
        values = np.zeros(
                (len(snk_src_tsep_list), #t1 + t2, indexed by ss_tsep_idx
                 max(snk_src_tsep_list), #t2, indexed by t_sep_idx
                 len(pipiop_tsep_list),
                 len(expr_names),
                ),
                dtype=np.complex128,)

        for val, t_sep_idx, ss_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size //1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")

            values[ss_sep_idx,t_sep_idx] += val
        return values.transpose(3,2,0,1) #(nexpr, nDelta, n_ss_tsep, n_int_tsep)

    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum)
    ld = q.mk_lat_data([
        ["expr_name", len(expr_names), expr_names, ],
        ["pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        ["snk_src_tsep", len(snk_src_tsep_list), snk_src_tsep_list, ],
        ["t2_sep", max(snk_src_tsep_list), list(range(1,max(snk_src_tsep_list))), ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append("f{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ls '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

    
# ----
#
#----

@q.timer(is_timer_fork=True)
def run_auto_contraction(
        job_tag, traj,
        *,
        get_get_prop,
        get_psel_prob,
        get_fsel_prob,
        ):
    fname = q.get_fname()
    fn_checkpoint = f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/checkpoint.txt"
    if get_load_path(fn_checkpoint) is not None:
        q.displayln_info(0, f"{fname}: '{fn_checkpoint}' exists.")
        return
    if not q.obtain_lock(f"locks/{job_tag}-{traj}-{fname}"):
        return
    get_prop = get_get_prop()
    assert get_prop is not None
    use_fsel_prop = get_param(job_tag, "measurement", "use_fsel_prop", default=True)
    # ADJUST ME
    if use_fsel_prop:
        #meson, psrc. Includes pion and sigma two point function.
        auto_contract_meson_corr_psnk_psrc(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)

        #pipi fsel.
        auto_contract_pipi_corr_psnk_psrc_V(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob) 

        #ATW fsel
        auto_contract_ATW3pt_psnk_psrc1(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
        auto_contract_ATW3pt_psnk_psrc2(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
        #auto_contract_ATW3pt_psnk_psrc3(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)

        

    #meson psrc psel
    #auto_contract_meson_corr_psnk_psrc_psel_pos(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob) 
    auto_contract_meson_corr_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)  
    
    #pipi psrc psel
    auto_contract_pipi_corr_psnk_psrc_psel_V(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
    #auto_contract_pipi_corr_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)

    #ATW three point function, psel
    auto_contract_ATW3pt_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)

    
    q.qtouch_info(get_save_path(fn_checkpoint))
    q.release_lock()
    v = [ f"{fname} {job_tag} {traj} done", ]
    return v

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
            f"{job_tag}/auto-contract-48I-pipi-dc/traj-{traj}/checkpoint.txt",
            #
            ]
    fns_need = [
            (f"{job_tag}/psel-prop-psrc-light/traj-{traj}.qar", f"{job_tag}/psel-prop-psrc-light/traj-{traj}/checkpoint.txt",),
            #(f"{job_tag}/psel-prop-psrc-strange/traj-{traj}.qar", f"{job_tag}/psel-prop-psrc-strange/traj-{traj}/checkpoint.txt",),
            #(f"{job_tag}/psel-prop-wsrc-light/traj-{traj}.qar", f"{job_tag}/psel-prop-wsrc-light/traj-{traj}/checkpoint.txt",),
            #(f"{job_tag}/psel-prop-wsrc-strange/traj-{traj}.qar", f"{job_tag}/psel-prop-wsrc-strange/traj-{traj}/checkpoint.txt",),
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
                #(f"{job_tag}/prop-psrc-strange/traj-{traj}.qar", f"{job_tag}/prop-psrc-strange/traj-{traj}/geon-info.txt",),
                #(f"{job_tag}/prop-wsrc-light/traj-{traj}.qar", f"{job_tag}/prop-wsrc-light/traj-{traj}/geon-info.txt",),
                #(f"{job_tag}/prop-wsrc-strange/traj-{traj}.qar", f"{job_tag}/prop-wsrc-strange/traj-{traj}/geon-info.txt",),
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
            #"wsrc psel s",
            #"wsrc psel l",
            #"psrc psel s",
            "psrc psel l",
            #"gauge transform" #necessary for smeared operators. 
            # "rand_u1 fsel c",
            # "rand_u1 fsel s",
            # "rand_u1 fsel l",
            ]
    if use_fsel_prop:
        prop_types += [
                #"wsrc fsel s",
                #"wsrc fsel l",
                #"psrc fsel s",
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



### ------ params
set_param("16IH2", "traj_list")(list(range(1400,2101,10)))#list(range(1000,1501,10)))
set_param("16IH2", "measurement", "auto_contractor_chunk_size")(128)
set_param("16IH2", "measurement", "meson_tensor_t_sep")(12)
set_param("16IH2", "measurement", "pipi_op_t_sep")([1,3]) #Delta
set_param("16IH2", "measurement", "pipi_op_dis_4d_sqr_limit")(0.0) #minimum squared distance for single pions
set_param("16IH2", "measurement", "pipi_corr_t_sep_list")(list(range(1, 11))) # pipi corr tsep list
set_param("16IH2", "measurement", "tsep_snk_src_3pt")([16,20]) #constant source-sink separation in 3pt function
set_param("16IH2", "measurement", "use_fsel_prop")(True)

set_param("48I", "traj_list")([2165])
set_param("48I", "measurement", "auto_contractor_chunk_size")(128)
set_param("48I", "measurement", "meson_tensor_t_sep")(12)
set_param("48I", "measurement", "pipi_op_t_sep")([5,7,9]) #Delta
set_param("48I", "measurement", "tsep_snk_src_3pt")([20,24,28,32])
set_param("48I", "measurement", "pipi_op_dis_4d_sqr_limit")(25.0) #Minimum squared 4d distance between the two pion operators. We need to try with 9.0 and 16.0
set_param("48I", "measurement", "pipi_corr_t_sep_list")(list(range(1, 24))) #list of time separations between the two pion operators that we want to measure

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
    c_count = 0
    ncf = 1
    for job_tag, traj in job_tag_traj_list:
        if is_performing_contraction:
            q.check_time_limit()
            run_job_contraction(job_tag, traj)
            q.clean_cache()
            c_count += 1
            print(f"DEBUG: c_count: {c_count}")
            if c_count >= ncf:
                try_gracefully_finish()

    gracefully_finish()

# ----












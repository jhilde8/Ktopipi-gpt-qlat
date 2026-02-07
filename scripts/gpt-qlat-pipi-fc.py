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
        #"/data1/qcddata2",
        #"/data1/qcddata3",
        #"/data2/qcddata3-prop",
        "/hpcgpfs01/scratch/jhildebra/psrc_props",
        "/hpcgpfs01/work/lqcd/staging/RBC/qcddata/MDWF/2+1f/48nt96/IWASAKI/b2.13/ls24b+c2/M1.8/ms0.0362/mu0.00078/jhildebra",
        #"/data1/qcddata2",
        #"/data1/qcddata3",
        #"/data2/qcddata3-prop",
        ]

# ----

    
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

@q.timer
def get_cexpr_pipi_corr_psnk_psrc():
    fn_base = "cache/auto_contract_cexpr/get_cexpr_pipi_psnk_psrc_fc"
    def calc_cexpr():
        diagram_type_dict = dict() #the auto contractor deals with each term within each type, along with prefactors present in the sum. 
        #pipi-pipi
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'snk_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'src_1'), 1))] = 'ADT1' #V
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'src_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT2' #R      
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('snk_2', 'src_2'), 1), (('src_1', 'snk_1'), 1), (('src_2', 'snk_2'), 1))] = 'ADT3' #D
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('snk_2', 'src_2'), 1), (('src_1', 'snk_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT4' #C
        #sigma-pipi. only connected piece, disconnected piece will be reconstructed from meson data
        #pipi-sigma
        diagram_type_dict[((('snk_1', 'src_1'), 1), (('src_1', 'src_2'), 1), (('src_2', 'snk_1'), 1))] = 'ADT5_pps'
        #sigma-pipi
        diagram_type_dict[((('snk_1', 'snk_2'), 1), (('snk_2', 'src_1'), 1), (('src_1', 'snk_1'), 1))] = 'ADT5_spp'

        exprs = [
                mk_fac(1) + f"1",
                
                ]
        for mode_src in [0,1,2,3]:

            #pipi-sigma cross terms
            (mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
            * mk_sigma('snk_1',True) * mk_pipi_i0('src_1','src_2') + f"wf_src({mode_src}) * sigma^dag(0) * pipi_i0(-tsep)",'ADT5_pps'),
            
            (mk_fac(f"pipi_wave_function(snk_1,snk_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
            * mk_pipi_i0('snk_1','snk_2',True) * mk_sigma('src_1') + f"wf_src({mode_src}) * pipi_i0^dag(0) * sigma(-tsep)",'ADT5_spp'),
            
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
                    
                        #Direct, I=0
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i0^dag(0) * pipi_i0(-tsep)",'ADT3'),
                    
                        #Cross, I=0
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i0^dag(0) * pipi_i0(-tsep)",'ADT4'),

                        #Rectangle
                        (mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        * mk_pipi_i0("snk_1", "snk_2", True)
                        * mk_pipi_i0("src_1", "src_2")
                        + f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i0^dag(0) * pipi_i0(-tsep)",'ADT2'),

                        #Total
                        #mk_fac(f"pipi_wave_function(snk_1, snk_2, {mode_snk}, size, pipi_op_dis_4d_sqr_limit)")
                        #* mk_fac(f"pipi_wave_function(src_1,src_2, {mode_src}, size, pipi_op_dis_4d_sqr_limit)")
                        #* mk_pipi_i0("snk_1", "snk_2", True) #true refers to the is_dagger boolean
                        #* mk_pipi_i0("src_1", "src_2")
                        #+ f"wf_snk({mode_snk}) * wf_src({mode_src}) * pipi_i0^dag(0) * pipi_i0(-tsep)",
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

# -----------------------------------------
# two pion two point function contractions.
# -----------------------------------------
 
#this will be used for both fully connected diagrams (R and C). 
@q.timer(is_timer_fork=True)
def auto_contract_pipi_corr_psnk_psrc_DCR(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-avg-test/traj-{traj}/pipi_corr_psnk_psrc.lat"
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
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")
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
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_src = xg_src[3]
        assert pidx_snk != pidx_src
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_src]
        idx_snk_src_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_src_2 = (t_src - pipiop_tsep) % t_size
            t_snk_2 = (t_snk + pipiop_tsep) % t_size
            for pidx_src_2 in pidx_list_list[t_src_2]:
                if pidx_src_2 in [ pidx_snk, pidx_src, ]:
                    continue
                for idx_snk_2 in fidx_list_list[t_snk_2]:
                    if idx_snk_2 in [ pidx_src_2, pidx_snk, pidx_src, ]:
                        continue
                    prob2 = fsel_prob_arr[idx_snk_2] * psel_prob_arr[pidx_src_2]
                    prob = prob1 * prob2
                    idx_snk_src_2_list.append((idx_snk_2, pidx_src_2, prob, pipiop_tsep_idx)) #we then save a list of the second source and sink locations
        #values array holds the evaluation of a given expression for each internal pion separation for both initial and final pions for each expression
        values = np.zeros(
                (len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        
        #iterating over the second sources and sinks, we evaluate each expression with the given source and sink locations, and assign them to the values array
        for idx_snk_2, pidx_src_2, prob, pipiop_tsep_idx in idx_snk_src_2_list:
            xg_snk_2 = q.Coordinate(xg_fsel_arr[idx_snk_2])
            xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
            t_snk_2 = xg_snk_2[3]
            t_src_2 = xg_src_2[3]
            pipi_op_t_sep_snk = (t_snk_2 - t_snk) % t_size
            pipi_op_t_sep_src = (t_src - t_src_2) % t_size
            assert pipi_op_t_sep_snk == pipi_op_t_sep_src
            pd = {
                    "snk_1": ("point", xg_snk.to_tuple(),),
                    "snk_2": ("point-snk", xg_snk_2.to_tuple(),),
                    "src_1": ("point", xg_src.to_tuple(),),
                    "src_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values[pipiop_tsep_idx] += val/prob
        return values, t_sep_idx, data_list_idx, data_list_size

    @q.timer
    def feval_2(args):
        data_list_idx, data_list_size, pidx_snk, pidx_src, t_sep_idx = args
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_src = xg_src[3]
        assert pidx_snk != pidx_src
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_src]
        idx_snk_src_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_src_2 = (t_src - pipiop_tsep) % t_size
            t_snk_2 = (t_snk + pipiop_tsep) % t_size
            for idx_src_2 in fidx_list_list[t_src_2]:
                if idx_src_2 in [ pidx_snk, pidx_src, ]:
                    continue
                for pidx_snk_2 in pidx_list_list[t_snk_2]:
                    if pidx_snk_2 in [ idx_src_2, pidx_snk, pidx_src, ]:
                        continue
                    prob2 = psel_prob_arr[pidx_snk_2] * fsel_prob_arr[idx_src_2]
                    prob = prob1 * prob2
                    idx_snk_src_2_list.append((pidx_snk_2, idx_src_2, prob,pipiop_tsep_idx)) #we then save a list of the second source and sink locations
        #values array holds the evaluation of a given expression for each internal pion separation for both initial and final pions for each expression
        values = np.zeros(
                (len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        
        #iterating over the second sources and sinks, we evaluate each expression with the given source and sink locations, and assign them to the values array
        for pidx_snk_2, idx_src_2, prob, pipiop_tsep_idx in idx_snk_src_2_list:
            xg_snk_2 = q.Coordinate(xg_psel_arr[pidx_snk_2])
            xg_src_2 = q.Coordinate(xg_fsel_arr[idx_src_2])
            t_snk_2 = xg_snk_2[3]
            t_src_2 = xg_src_2[3]
            pipi_op_t_sep_snk = (t_snk_2 - t_snk) % t_size
            pipi_op_t_sep_src = (t_src - t_src_2) % t_size
            assert pipi_op_t_sep_snk == pipi_op_t_sep_src
            pd = {
                    "snk_1": ("point", xg_snk.to_tuple(),),
                    "snk_2": ("point", xg_snk_2.to_tuple(),),
                    "src_1": ("point", xg_src.to_tuple(),),
                    "src_2": ("point-snk", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values[pipiop_tsep_idx] += val/prob
        return values, t_sep_idx, data_list_idx, data_list_size
        
    def sum_function(val_list):
        values = np.zeros(
                (len(pipi_corr_t_sep_list),
                 len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        for val, t_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size // 1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")
            values[t_sep_idx] += val
        return values.transpose(2, 1, 0)
    res_sum1 = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum1 = q.glb_sum(res_sum1)
    
    res_sum2 = q.parallel_map_sum(feval_2, load_data(), sum_function=sum_function, chunksize=1)
    res_sum2 = q.glb_sum(res_sum2)  

    res_sum = 0.5*(res_sum1 + res_sum2)
    
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        [ "t_sep", len(pipi_corr_t_sep_list), pipi_corr_t_sep_list, ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

# ----
#calculate individually then average

#this will be used for both fully connected diagrams (R and C). 
@q.timer(is_timer_fork=True)
def auto_contract_pipi_corr_psnk_psrc_DCR1(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-fc/traj-{traj}/pipi_corr_psnk_psrc1.lat"
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
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")
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
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_src = xg_src[3]
        assert pidx_snk != pidx_src
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_src]
        idx_snk_src_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_src_2 = (t_src - pipiop_tsep) % t_size
            t_snk_2 = (t_snk + pipiop_tsep) % t_size
            for pidx_src_2 in pidx_list_list[t_src_2]:
                if pidx_src_2 in [ pidx_snk, pidx_src, ]:
                    continue
                for idx_snk_2 in fidx_list_list[t_snk_2]:
                    if idx_snk_2 in [ pidx_src_2, pidx_snk, pidx_src, ]:
                        continue
                    prob2 = fsel_prob_arr[idx_snk_2] * psel_prob_arr[pidx_src_2]
                    prob = prob1 * prob2
                    idx_snk_src_2_list.append((idx_snk_2, pidx_src_2, prob, pipiop_tsep_idx)) #we then save a list of the second source and sink locations
        values = np.zeros(
                (len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        
        for idx_snk_2, pidx_src_2, prob, pipiop_tsep_idx in idx_snk_src_2_list:
            xg_snk_2 = q.Coordinate(xg_fsel_arr[idx_snk_2])
            xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
            t_snk_2 = xg_snk_2[3]
            t_src_2 = xg_src_2[3]
            pipi_op_t_sep_snk = (t_snk_2 - t_snk) % t_size
            pipi_op_t_sep_src = (t_src - t_src_2) % t_size
            assert pipi_op_t_sep_snk == pipi_op_t_sep_src
            pd = {
                    "snk_1": ("point", xg_snk.to_tuple(),),
                    "snk_2": ("point-snk", xg_snk_2.to_tuple(),),
                    "src_1": ("point", xg_src.to_tuple(),),
                    "src_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values[pipiop_tsep_idx] += val/prob
        return values, t_sep_idx, data_list_idx, data_list_size
    
    def sum_function(val_list):
        values = np.zeros(
                (len(pipi_corr_t_sep_list),
                 len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        for val, t_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size // 1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")
            values[t_sep_idx] += val
        return values.transpose(2, 1, 0)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum) 
    
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        [ "t_sep", len(pipi_corr_t_sep_list), pipi_corr_t_sep_list, ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))


#this will be used for both fully connected diagrams (R and C). 
@q.timer(is_timer_fork=True)
def auto_contract_pipi_corr_psnk_psrc_DCR2(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-fc/traj-{traj}/pipi_corr_psnk_psrc2.lat"
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
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")
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
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_src = xg_src[3]
        assert pidx_snk != pidx_src
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_src]
        idx_snk_src_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_src_2 = (t_src - pipiop_tsep) % t_size
            t_snk_2 = (t_snk + pipiop_tsep) % t_size
            for idx_src_2 in fidx_list_list[t_src_2]:
                if idx_src_2 in [ pidx_snk, pidx_src, ]:
                    continue
                for pidx_snk_2 in pidx_list_list[t_snk_2]:
                    if pidx_snk_2 in [ idx_src_2, pidx_snk, pidx_src, ]:
                        continue
                    prob2 = psel_prob_arr[pidx_snk_2] * fsel_prob_arr[idx_src_2]
                    prob = prob1 * prob2
                    idx_snk_src_2_list.append((pidx_snk_2, idx_src_2, prob,pipiop_tsep_idx)) #we then save a list of the second source and sink locations
        values = np.zeros(
                (len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        
        for pidx_snk_2, idx_src_2, prob, pipiop_tsep_idx in idx_snk_src_2_list:
            xg_snk_2 = q.Coordinate(xg_psel_arr[pidx_snk_2])
            xg_src_2 = q.Coordinate(xg_fsel_arr[idx_src_2])
            t_snk_2 = xg_snk_2[3]
            t_src_2 = xg_src_2[3]
            pipi_op_t_sep_snk = (t_snk_2 - t_snk) % t_size
            pipi_op_t_sep_src = (t_src - t_src_2) % t_size
            assert pipi_op_t_sep_snk == pipi_op_t_sep_src
            pd = {
                    "snk_1": ("point", xg_snk.to_tuple(),),
                    "snk_2": ("point", xg_snk_2.to_tuple(),),
                    "src_1": ("point", xg_src.to_tuple(),),
                    "src_2": ("point-snk", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values[pipiop_tsep_idx] += val/prob
        return values, t_sep_idx, data_list_idx, data_list_size
    
    def sum_function(val_list):
        values = np.zeros(
                (len(pipi_corr_t_sep_list),
                 len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        for val, t_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size // 1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")
            values[t_sep_idx] += val
        return values.transpose(2, 1, 0)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum) 
    
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        [ "t_sep", len(pipi_corr_t_sep_list), pipi_corr_t_sep_list, ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))


@q.timer(is_timer_fork=True)
def auto_contract_pipi_corr_psnk_psrc_DCR3(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-fc/traj-{traj}/pipi_corr_psnk_psrc3.lat"
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
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")
    pipi_op_dis_4d_sqr_limit = get_param(job_tag, "measurement", "pipi_op_dis_4d_sqr_limit")
    pipi_corr_t_sep_list = get_param(job_tag, "measurement", "pipi_corr_t_sep_list")
    data_list = []
    for pidx_src in range(len(xg_psel_arr)):
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_src = xg_src[3]
        for t_sep_idx, t_sep in enumerate(pipi_corr_t_sep_list):
            assert t_sep > 0
            t_snk = (t_src + t_sep) % t_size
            for idx_snk in fidx_list_list[t_snk]:
                xg_snk = q.Coordinate(xg_fsel_arr[idx_snk])
                assert xg_snk[3] == t_snk
                if idx_snk == pidx_src:
                    continue
                data_list.append((idx_snk, pidx_src, t_sep_idx,))
    def load_data():
        data_list_chunk = q.get_mpi_chunk(data_list)
        data_list_size = len(data_list_chunk)
        for data_list_idx, (idx_snk, pidx_src, t_sep_idx,) in enumerate(data_list_chunk):
            yield data_list_idx, data_list_size, idx_snk, pidx_src, t_sep_idx
    @q.timer
    def feval(args):
        data_list_idx, data_list_size, idx_snk, pidx_src, t_sep_idx = args
        xg_snk = q.Coordinate(xg_fsel_arr[idx_snk])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_src = xg_src[3]
        assert idx_snk != pidx_src
        prob1 = fsel_prob_arr[idx_snk] * psel_prob_arr[pidx_src]
        idx_snk_src_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_src_2 = (t_src - pipiop_tsep) % t_size
            t_snk_2 = (t_snk + pipiop_tsep) % t_size
            for pidx_src_2 in pidx_list_list[t_src_2]:
                if pidx_src_2 in [ idx_snk, pidx_src, ]:
                    continue
                for pidx_snk_2 in pidx_list_list[t_snk_2]:
                    if pidx_snk_2 in [ pidx_src_2, idx_snk, pidx_src, ]:
                        continue
                    prob2 = psel_prob_arr[pidx_snk_2] * psel_prob_arr[pidx_src_2]
                    prob = prob1 * prob2
                    idx_snk_src_2_list.append((pidx_snk_2, pidx_src_2, prob, pipiop_tsep_idx)) #we then save a list of the second source and sink locations
        #values array holds the evaluation of a given expression for each internal pion separation for both initial and final pions for each expression
        values = np.zeros(
                (len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        
        #iterating over the second sources and sinks, we evaluate each expression with the given source and sink locations, and assign them to the values array
        for pidx_snk_2, pidx_src_2, prob, pipiop_tsep_idx in idx_snk_src_2_list:
            xg_snk_2 = q.Coordinate(xg_psel_arr[pidx_snk_2])
            xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
            t_snk_2 = xg_snk_2[3]
            t_src_2 = xg_src_2[3]
            pipi_op_t_sep_snk = (t_snk_2 - t_snk) % t_size
            pipi_op_t_sep_src = (t_src - t_src_2) % t_size
            assert pipi_op_t_sep_snk == pipi_op_t_sep_src
            pd = {
                    "snk_1": ("point-snk", xg_snk.to_tuple(),),
                    "snk_2": ("point", xg_snk_2.to_tuple(),),
                    "src_1": ("point", xg_src.to_tuple(),),
                    "src_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values[pipiop_tsep_idx] += val/prob
        return values, t_sep_idx, data_list_idx, data_list_size
    
    def sum_function(val_list):
        values = np.zeros(
                (len(pipi_corr_t_sep_list),
                 len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        for val, t_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size // 1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")
            values[t_sep_idx] += val
        return values.transpose(2, 1, 0)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum) 
    
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        [ "t_sep", len(pipi_corr_t_sep_list), pipi_corr_t_sep_list, ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))

    
# ----

#full pipi contractions are done in the psel case, no reconstruction needed. 
@q.timer(is_timer_fork=True)
def auto_contract_pipi_corr_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob):
    fname = q.get_fname()
    fn = f"{job_tag}/auto-contract-pipi-fc/traj-{traj}/pipi_corr_psnk_psrc_psel.lat"
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
    pipiop_tsep_list = get_param(job_tag, "measurement", "pipi_op_t_sep")
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
        xg_snk = q.Coordinate(xg_psel_arr[pidx_snk])
        xg_src = q.Coordinate(xg_psel_arr[pidx_src])
        t_snk = xg_snk[3]
        t_src = xg_src[3]
        assert pidx_snk != pidx_src
        prob1 = psel_prob_arr[pidx_snk] * psel_prob_arr[pidx_src]
        idx_snk_src_2_list = []

        for pipiop_tsep_idx, pipiop_tsep in enumerate(pipiop_tsep_list):
            t_src_2 = (t_src - pipiop_tsep) % t_size
            t_snk_2 = (t_snk + pipiop_tsep) % t_size
            for pidx_src_2 in pidx_list_list[t_src_2]:
                if pidx_src_2 in [ pidx_snk, pidx_src, ]:
                    continue
                for pidx_snk_2 in pidx_list_list[t_snk_2]:
                    if pidx_snk_2 in [ pidx_src_2, pidx_snk, pidx_src, ]:
                        continue
                    prob2 = psel_prob_arr[pidx_snk_2] * psel_prob_arr[pidx_src_2]
                    prob = prob1 * prob2
                    idx_snk_src_2_list.append((pidx_snk_2, pidx_src_2, prob, pipiop_tsep_idx)) #we then save a list of the second source and sink locations
        #values array holds the evaluation of a given expression for each internal pion separation for both initial and final pions for each expression
        values = np.zeros(
                (len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        
        #iterating over the second sources and sinks, we evaluate each expression with the given source and sink locations, and assign them to the values array
        for pidx_snk_2, pidx_src_2, prob, pipiop_tsep_idx in idx_snk_src_2_list:
            xg_snk_2 = q.Coordinate(xg_psel_arr[pidx_snk_2])
            xg_src_2 = q.Coordinate(xg_psel_arr[pidx_src_2])
            t_snk_2 = xg_snk_2[3]
            t_src_2 = xg_src_2[3]
            pipi_op_t_sep_snk = (t_snk_2 - t_snk) % t_size
            pipi_op_t_sep_src = (t_src - t_src_2) % t_size
            assert pipi_op_t_sep_snk == pipi_op_t_sep_src
            pd = {
                    "snk_1": ("point-snk", xg_snk.to_tuple(),),
                    "snk_2": ("point", xg_snk_2.to_tuple(),),
                    "src_1": ("point", xg_src.to_tuple(),),
                    "src_2": ("point", xg_src_2.to_tuple(),),
                    "size": total_site,
                    "pipi_op_dis_4d_sqr_limit": pipi_op_dis_4d_sqr_limit,
                    }
            val = eval_cexpr(cexpr, positions_dict=pd, get_prop=get_prop)
            values[pipiop_tsep_idx] += val/prob
        return values, t_sep_idx, data_list_idx, data_list_size
    def sum_function(val_list):
        values = np.zeros(
                (len(pipi_corr_t_sep_list),
                 len(pipiop_tsep_list),
                 len(expr_names),
                 ),
                dtype=np.complex128,
                )
        for val, t_sep_idx, data_list_idx, data_list_size in val_list:
            if data_list_idx % (data_list_size // 1024 + 4) == 0:
                q.displayln_info(0, f"{fname}: {data_list_idx}/{data_list_size}")
            values[t_sep_idx] += val
        return values.transpose(2, 1, 0)
    res_sum = q.parallel_map_sum(feval, load_data(), sum_function=sum_function, chunksize=1)
    res_sum = q.glb_sum(res_sum) 
    
    ld = q.mk_lat_data([
        [ "expr_name", len(expr_names), expr_names, ],
        [ "pipiop_tsep", len(pipiop_tsep_list), pipiop_tsep_list, ],
        [ "t_sep", len(pipi_corr_t_sep_list), pipi_corr_t_sep_list, ],
        ])
    ld.from_numpy(res_sum)
    ld.save(get_save_path(fn))
    q.json_results_append(f"{fname}: ld sig", q.get_data_sig_arr(ld, q.RngState(), 4))
    for i, en in enumerate(expr_names):
        q.json_results_append(f"{fname}: ld '{en}' sig", q.get_data_sig_arr(ld[i], q.RngState(), 4))
    
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
    fn_checkpoint = f"{job_tag}/auto-contract-pipisc-fc/traj-{traj}/checkpoint.txt"
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

        #two different fevals, does the average in this script
        #auto_contract_pipi_corr_psnk_psrc_DCR(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob) 

        #one function for eachfsel permutation (memory concerns)
        auto_contract_pipi_corr_psnk_psrc_DCR1(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
        auto_contract_pipi_corr_psnk_psrc_DCR2(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
        auto_contract_pipi_corr_psnk_psrc_DCR3(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
    
    #pipi psrc psel
    #auto_contract_pipi_corr_psnk_psrc_psel_V(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
    auto_contract_pipi_corr_psnk_psrc_psel(job_tag, traj, get_get_prop, get_psel_prob, get_fsel_prob)
    
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
            f"{job_tag}/auto-contract-pipisc-fc/traj-{traj}/checkpoint.txt",
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
set_param("16IH2", "traj_list")(list(range(1100,2101,10)))#list(range(1000,1501,10)))
set_param("16IH2", "measurement", "auto_contractor_chunk_size")(128)
set_param("16IH2", "measurement", "meson_tensor_t_sep")(12)
set_param("16IH2", "measurement", "pipi_op_t_sep")([1,3]) #Delta
set_param("16IH2", "measurement", "pipi_op_dis_4d_sqr_limit")(0.0) #minimum squared distance for single pions
set_param("16IH2", "measurement", "pipi_corr_t_sep_list")(list(range(1, 11))) # pipi corr tsep list
set_param("16IH2", "measurement", "tsep_snk_src_3pt")([12]) #constant source-sink separation in 3pt function
set_param("16IH2", "measurement", "use_fsel_prop")(True)

set_param("48I", "traj_list")([2165])
set_param("48I", "measurement", "auto_contractor_chunk_size")(128)
set_param("48I", "measurement", "meson_tensor_t_sep")(12)
set_param("48I", "measurement", "pipi_op_t_sep")([5,7,9]) #Delta
set_param("48I", "measurement", "pipi_op_dis_4d_sqr_limit")(25.0) #Minimum squared 4d distance between the two pion operators. We need to try with 9.0 and 16.0
set_param("48I", "measurement", "pipi_corr_t_sep_list")(list(range(1, 24))) #list of time separations between the two pion operators that we want to measure
set_param("48I", "measurement", "tsep_snk_src_3pt")([12,18,24,30]) #list of fixed time separations between source and sink for a three point function

set_param("48I", "measurement", "pipi_tensor_t_sep_list")([ 1, 2, ]) #not used
set_param("48I", "measurement", "pipi_tensor_t_max")(20) #not used
set_param("48I", "measurement", "pipi_tensor_r_max")(24) #not used
set_param("48I", "measurement", "use_fsel_prop")(True)

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












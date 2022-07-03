import os

def sub_command(cmd,identifier, hours=4):
   # add_path = 'export PATH="/cluster/home/amollers/software/anaconda/bin:${PATH}"'
    setup_cmd = 'source activate /cluster/home/amollers/software/anaconda/envs/base_env'
    # TODO: adjust this to needs with -n {cpus} and mem as param
    sub_cmd = (f'bsub -W {hours}:00 -n 4 -R "rusage[mem=40000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=15240]" '
               f'-oo logs/gs_hs_cpath/hs_logs/{identifier}.log -J {identifier} -eo logs/gs_hs_cpath/err_logs/{identifier}.err "{setup_cmd}; {cmd}"')
    os.system(sub_cmd)

learning_rates = [0.1]
gaussianprior_variances = [0.01]
global_cauchy_scale = [0.1,1.,2.]
weight_cauchy_scale = [1.,1.5,2.]
epochs = [70]
num_mcs = [300]


for learning_rate in learning_rates:
    for gp_var in gaussianprior_variances:
        for ghs_scale in global_cauchy_scale:
            for whs_scale in weight_cauchy_scale:
                for epoch in epochs:
                    for num_mc in num_mcs:

                        cmd =(f'/cluster/home/amollers/software/anaconda/envs/base_env/bin/python hs_cpath_ex.py --epochs {epoch} '
                              f'--num-mc {num_mc} --lr {learning_rate} --gp-var {gp_var} --hs-glob {ghs_scale} --hs-group {whs_scale} '
                              f'--save-dir "/cluster/home/amollers/Github/survival_analysis/run_pipeline/logs/gs_hs_cpath/model_checkpoints" '
                              f'--arch "cpath_hs_model_{learning_rate}_{gp_var}_{ghs_scale}_{whs_scale}" '
                              f'--log-dir "/cluster/home/amollers/Github/survival_analysis/run_pipeline/logs/gs_hs_cpath/hs_output" ')

                        sub_command(cmd,f'"cpathhs_model_{learning_rate}_{gp_var}_{ghs_scale}_{whs_scale}"',hours=4)

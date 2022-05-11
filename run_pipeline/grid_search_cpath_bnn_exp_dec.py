import os

def sub_command(cmd,identifier, hours=4):
   # add_path = 'export PATH="/cluster/home/amollers/software/anaconda/bin:${PATH}"'
    setup_cmd = 'source activate /cluster/home/amollers/software/anaconda/envs/base_env'
    # TODO: adjust this to needs with -n {cpus} and mem as param
    sub_cmd = (f'bsub -W {hours}:00 -n 6 -R "rusage[mem=20000,ngpus_excl_p=1]" '
               f'-oo /cluster/home/amollers/Github/survival_analysis/run_pipeline/logs/gs_cp_bnn_cwarmann/outputs/{identifier}.log -J {identifier} -eo /cluster/home/amollers/Github/survival_analysis/run_pipeline/logs/gs_cp_bnn_cwarmann/err_logs/{identifier}.err "{setup_cmd}; {cmd}"')
    os.system(sub_cmd)

learning_rates = [0.1]
gaussianprior_variances = [0.01,0.5]
epochs = [1]
num_mcs = [300]


for learning_rate in learning_rates:
    for gp_var in gaussianprior_variances:
        for epoch in epochs:
            for num_mc in num_mcs:

                cmd =(f'/cluster/home/amollers/software/anaconda/envs/base_env/bin/python cpath_mdrop_lingau_bnn_wcosann_decay.py --epochs {epoch} '
                      f'--num-mc {num_mc} --lr {learning_rate} --gp-var {gp_var} '
                      f'--save-dir "/cluster/home/amollers/Github/survival_analysis/run_pipeline/logs/gs_cp_bnn_cwarmann/model_checkpoints"'
                      f'--arch "cpath_model_{learning_rate}_{gp_var}" '
                      f'--log-dir "/cluster/home/amollers/Github/survival_analysis/run_pipeline/logs/gs_cp_bnn_cwarmann/logs"'
                      f'--save-best-model="False"')

                sub_command(cmd,f'"cpath_wcosann_{learning_rate}_{gp_var}"',hours=4)
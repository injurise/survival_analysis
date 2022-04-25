import os

def sub_command(cmd,identifier, hours=4):
    setup_cmd = 'conda activate /cluster/home/amollers/software/anaconda/envs/base_env'
    # TODO: adjust this to needs with -n {cpus} and mem as param
    sub_cmd = (f'bsub -W {hours}:00 -n 4 -R "rusage[mem=2000]" '
               f'-o logs/{identifier}.log -J {identifier} "{setup_cmd}; {cmd}"')
    os.system(sub_cmd)

learning_rates = [0.05,0.6]
gaussianprior_variances = [0.07]
epochs = [100]
num_mcs = [200]


for learning_rate in learning_rates:
    for gp_var in gaussianprior_variances:
        for epoch in epochs:
            for num_mc in num_mcs:

                cmd =(f'cpath_mdrop_lingau_bnn_bsub.py --epochs {epoch} ' 
                      f'--num-mc {num_mc} --lr {learning_rate} --gp-var {gp_var}' 
                      f'--savedir "/cluster/home/amollers/Github/survival_analysis/run_pipeline/model_checkpoints"' 
                      f'--arch "cpath_model_{learning_rate}_{gp_var}" '
                      f'--log-dir {"/cluster/home/amollers/Github/survival_analysis/run_pipeline/model_checkpoints"} ')

                sub_command(cmd,f'--arch "cpath_model_{learning_rate}_{gp_var}"',hours=4)

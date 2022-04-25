import os

def sub_command(cmd, hours=4):
    setup_cmd = 'conda activate /cluster/home/amollers/software/anaconda/envs/base_env'
    # TODO: adjust this to needs with -n {cpus} and mem as param
    sub_cmd = f'bsub -W {hours}:00 -n 4 -R "rusage[mem=2000]" "{setup_cmd}; {cmd}"'
    os.system(sub_cmd)

learning_rates = [0.01,0.1]
gaussianprior_variances = [0.01]
epochs = [100]
num_mcs = [200]


for lr in learning_rates:
    for gp_var in gaussianprior_variances:
        for epoch in epochs:
            for num_mc in num_mcs:

                cmd = f'cpath_mdrop_lingau_bnn.py --epochs {epoch} ' \
                      f'--num-mc {num_mc} --lr {lr} --gp-var {gp_var}' \
                      f'--savedir "/Users/alexandermollers/Documents/GitHub/survival_analysis/run_pipeline/model_checkpoints"' \
                      f'--arch "cpath_model_{lr}_{gp_var}" '\
                      f'--log-dir {"/Users/alexandermollers/Documents/GitHub/survival_analysis/run_pipeline/model_checkpoints"} '

                sub_command(cmd, hours=4)
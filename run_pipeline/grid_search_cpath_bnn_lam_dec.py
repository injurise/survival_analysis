import os

def sub_command(cmd,identifier, hours=4):
   # add_path = 'export PATH="/cluster/home/amollers/software/anaconda/bin:${PATH}"'
    setup_cmd = 'source activate /cluster/home/amollers/software/anaconda/envs/base_env'
    # TODO: adjust this to needs with -n {cpus} and mem as param
    sub_cmd = (f'bsub -W {hours}:00 -n 4 -R "rusage[mem=20000]" '
               f'-oo logs/{identifier}.log -J {identifier} -eo err_logs/{identifier}.err "{setup_cmd}; {cmd}"')
    os.system(sub_cmd)

learning_rates = [0.001,0.01,0.1,0.2]
gaussianprior_variances = [0.01,0.05,0.1,0.2,0.5]
epochs = [70]
num_mcs = [300]


for learning_rate in learning_rates:
    for gp_var in gaussianprior_variances:
        for epoch in epochs:
            for num_mc in num_mcs:

                cmd =(f'/cluster/home/amollers/software/anaconda/envs/base_env/bin/python cpath_mdrop_lingau_bnn_bsub.py --epochs {epoch} '
                      f'--num-mc {num_mc} --lr {learning_rate} --gp-var {gp_var} '
                      f'--save-dir "/cluster/home/amollers/Github/survival_analysis/run_pipeline/model_checkpoints" '
                      f'--arch "cpath_model_{learning_rate}_{gp_var}" '
                      f'--log-dir "/cluster/home/amollers/Github/survival_analysis/run_pipeline/logs"')

                sub_command(cmd,f'"cpath_model_{learning_rate}_{gp_var}"',hours=4)

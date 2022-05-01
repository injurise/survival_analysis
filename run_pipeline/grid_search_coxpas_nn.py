import os

def sub_command(cmd,identifier, hours=4):
   # add_path = 'export PATH="/cluster/home/amollers/software/anaconda/bin:${PATH}"'
    setup_cmd = 'source activate /cluster/home/amollers/software/anaconda/envs/base_env'
    # TODO: adjust this to needs with -n {cpus} and mem as param
    sub_cmd = (f'bsub -W {hours}:00 -n 6 -R "rusage[mem=20000]" '
               f'-oo logs/{identifier}.log -J {identifier} -eo err_logs/{identifier}.err "{setup_cmd}; {cmd}"')
    os.system(sub_cmd)

learning_rates = [0.001,0.01,0.1,0.2]
l2s = [0.01,0.05,0.1,0.2,0.5]
epochs = [2]


for learning_rate in learning_rates:
    for l2 in l2s:
        for epoch in epochs:

                cmd =(f'/cluster/home/amollers/software/anaconda/envs/base_env/bin/python coxpas_nn_bsub.py --epochs {epoch} '
                      f' --lr {learning_rate} --l2 {l2} '
                      f'--arch "coxpas_nn_{learning_rate}_{l2}" '
                      f'--log-dir "/cluster/home/amollers/Github/survival_analysis/run_pipeline/logs"')

                sub_command(cmd,f'"coxpas_nn_{learning_rate}_{l2}"',hours=4)

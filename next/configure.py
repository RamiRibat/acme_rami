import os, yaml, json, subprocess

path = os.path.join(os.getcwd()+'/config_v.yaml')
config = yaml.safe_load(open(path))

suites, levels, tasks = [], [], []


for suite in config.keys():
    print('suite: ', suite)
    suites.append(suite)
    for level in config[suite].keys():
        print('level: ', level)
        levels.append(level)
        num_steps = config[suite][level]['run']['steps']
        # tasks = tuple(config[suite][level]['tasks'])
        tasks = '(1 2 3)'
        subprocess.run(
            f"echo declare -a steps={num_steps} >> script_tmp.sh",
            shell=True)
        subprocess.run(
            f"echo declare -a {level}='(1 2 3 4 5 6 7)' >> script_tmp.sh",
            shell=True)
    txt = "[0t]=123 [1e]=321 [2m]=456 [3h]=654"
    subprocess.run(
        f"echo declare -A {suite}='({txt})' >> script_tmp.sh",
        shell=True)
subprocess.run(
    f"echo declare -a config=control >> script_tmp.sh",
    shell=True)

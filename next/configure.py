import os, yaml, json, subprocess

path = os.path.join(os.getcwd()+'/config.yaml')
config = yaml.safe_load(open(path))

suites, levels, tasks = [], [], []

# subprocess.run('echo declare -a config >> script_tmp.sh',shell=True)
for suite in config.keys():
    print('suite: ', suite)
    suites.append(suite)
    # subprocess.run(
    #     f"echo declare -a {suite} >> script_tmp.sh",
    #     shell=True)
    for level in config[suite].keys():
        print('level: ', level)
        levels.append(level)
        num_steps = config[suite][level]['run']['steps']
        # tasks = tuple(config[suite][level]['tasks'])
        tasks = '(1 2 3)'
        # subprocess.run(
        #     f"echo declare -a {level} >> script_tmp.sh",
        #     shell=True)
        subprocess.run(
            f"echo declare -a steps={num_steps} >> script_tmp.sh",
            shell=True)
        # subprocess.run(
        #     f"echo tasks='{tasks}' >> script_tmp.sh",
        #     shell=True)
        # subprocess.run(
        #     f"echo {level}[steps]={num_steps} >> script_tmp.sh",
        #     shell=True)
        # subprocess.run(
        #     f"echo {level}[tasks]='{'${tasks[*]}'}' >> script_tmp.sh",
        #     shell=True)
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

# subprocess.run('echo A=10 >> script_tmp.sh', shell=True)
import subprocess


def run_command(cmd):
    run = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    result = run.stdout
    result_str = result.decode().strip('\n').strip()

    return result_str


def get_device_info(device_id):
    sys_info = {}
    fields = ['Dev Name', 'Ver', 'GCU CLK', 'Mem CLK']

    for field in fields:
        col = 3 if field == 'Ver' else 4

        cmd = f"efsmi --q -i 0 | grep '{field}' | awk " + "'{print $" + str(col) + "}'"
        result_str = run_command(cmd)

        if 'CLK' in field:
            result_str += 'MHz'

        if field == 'Ver':
            field == 'KMD Ver'

        sys_info[field] = result_str

    return sys_info


def get_deb_pkg_info():
    deb_info = {}
    deb_names = ['topsruntime', 'tops-sdk', 'tops-inference', 'topsaten']

    for deb in deb_names:
        cmd = f"dpkg -l | grep '{deb}' | awk " + "'{print $3}'"
        result_str = run_command(cmd)
        deb_info[deb] = result_str

    return deb_info


def get_python_pkg_info():
    python_info = {}

    result_str = run_command('python3 -V')
    python_info['python'] = result_str

    deb_names = ['TopsInference', 'torch', 'diffusers', 'transformers', 'torch-gcu', 'Pillow', 'opencv-python']

    for deb in deb_names:
        cmd = f"python3 -m pip list | grep '{deb}' | awk " + "'{print $2}' | head -n 1"
        result_str = run_command(cmd)
        python_info[deb] = result_str


    return python_info


def get_os_info():
    os_info = {}

    cmds = ['lsb_release -sd', 'uname -r']
    keys = ['os_distrb_name', 'os_kernel_version']

    for cmd, key in zip(cmds, keys):
        result_str = run_command(cmd)
        os_info[key] = result_str

    return os_info


def get_cpu_info():
    cpu_info = {}

    cmds = [
        "uname -m",
        'cat /proc/cpuinfo | grep "model name" | uniq | awk \'{str=""; for (i=4; i<=NF; i++) str=str $i " "; print str}\'',
        "cat /proc/cpuinfo | grep \"vendor_id\" | uniq | awk '{print $3}'",
    ]

    keys = ['cpu_arch', 'cpu_model_name', 'cpu_vendor']

    for key, cmd in zip(keys, cmds):
        result_str = run_command(cmd)
        cpu_info[key] = result_str

    return cpu_info


def get_disk_info():
    disk_info = {}
    cmds = [
        "lsblk -o NAME,MODEL",
    ]

    keys = ['disk_model']

    for key, cmd in zip(keys, cmds):
        result_str = run_command(cmd)
        disk_info[key] = result_str

    return disk_info


def get_host_info():
    host_info = {}
    cmds = [
        "hostname",
        "hostname -I | awk '{print $1}'"
    ]

    keys = ['host_name', 'host_ip']

    for key, cmd in zip(keys, cmds):
        result_str = run_command(cmd)
        host_info[key] = result_str

    return host_info


def get_meta_info(device_id=None):
    meta_info = {}
    meta_info['device_info'] = get_device_info(device_id)
    meta_info['deb_info'] = get_deb_pkg_info()
    meta_info['python_info'] = get_python_pkg_info()
    meta_info['os_info'] = get_os_info()
    meta_info['cpu_info'] = get_cpu_info()
    meta_info['disk_info'] = get_disk_info()
    meta_info['host_info'] = get_host_info()

    return meta_info


if __name__ == '__main__':
    import json

    meta_info = get_meta_info(device_id=0)
    meta = json.dumps(meta_info, indent=4)
    print(meta)

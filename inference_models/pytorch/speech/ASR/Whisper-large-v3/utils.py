#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Enflame. All Rights Reserved.
#
import sys
import subprocess
import traceback
import socket
from collections import OrderedDict


def get_python_version():
    v = sys.version_info
    return (int(v.major), int(v.minor), int(v.micro))


def get_local_ip():
    try:
        proc = subprocess.Popen(["ip -o -4 route show to default | awk '{print $5}'"],
                                stdout=subprocess.PIPE, shell=True)
        iface_name = proc.communicate()[0].strip()
        iface_name = iface_name.decode('utf-8')
        # if many iface name, only get first
        iface_name = iface_name.split("\n")[0]
        proc = subprocess.Popen(
            ["ip -4 addr show {} | awk '/inet/ {{print $2}}' | cut -d/ -f1".format(
                iface_name)],
            stdout=subprocess.PIPE, shell=True)
        node_ip = proc.communicate()[0].strip()
        local_ip = node_ip.decode('utf-8').split("\n")[0]
    except Exception as ex:
        traceback.print_exc()
        local_ip = "unknown_ip"
    return str(local_ip.strip())


def get_host_mem():
    try:
        proc = subprocess.Popen(["free -m |grep Mem |awk -F\" \" '{print $2}'"],
                                stdout=subprocess.PIPE, shell=True)
        host_mem = proc.communicate()[0].strip()
        host_mem = host_mem.decode('utf-8').split("\n")[0]
    except Exception as ex:
        traceback.print_exc()
        host_mem = ""
    return str(host_mem.strip())


def get_host_model():
    try:
        proc = subprocess.Popen(
            ["cat /proc/cpuinfo |grep \"model name\" |uniq |awk -F\":\" '{print $2}'"],
            stdout=subprocess.PIPE, shell=True)
        cpu_model = proc.communicate()[0].strip()
        cpu_model = cpu_model.decode('utf-8').split("\n")[0]
    except Exception as ex:
        traceback.print_exc()
        cpu_model = ""
    return str(cpu_model.strip())


def get_host_name():
    try:
        host_name = socket.gethostname()
    except Exception as ex:
        traceback.print_exc()
        host_name = ""
    return str(host_name.strip())


def get_host_cores():
    try:
        proc = subprocess.Popen(
            ["cat /proc/cpuinfo |grep \"cpu cores\" |uniq |awk -F\":\" '{print $2}'"],
            stdout=subprocess.PIPE, shell=True)
        cpu_cores = proc.communicate()[0].strip()
        cpu_cores = cpu_cores.decode('utf-8').split("\n")[0]
    except Exception as ex:
        traceback.print_exc()
        cpu_cores = 1
    return cpu_cores


def get_os_name():
    try:
        proc = subprocess.Popen(
            ["cat /etc/os-release | grep PRETTY_NAME |awk -F\"=\" '{print$2}'"],
            stdout=subprocess.PIPE, shell=True)
        os = proc.communicate()[0].strip()
        os = os.decode('utf-8').split("\n")[0].replace('"', '')
    except Exception as ex:
        traceback.print_exc()
        os = 'unknown'
    return os


def get_gcu_clock_freq():
    try:
        proc = subprocess.Popen(
            ["efsmt -clock list |grep GCU |awk -F' ' '{print $5}'"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        gcu_clock = proc.communicate()[0].strip()
        gcu_clock = gcu_clock.decode('utf-8')
    except Exception as ex:
        # traceback.print_exc()
        gcu_clock = None
    return gcu_clock


def get_hbm_phy():
    try:
        proc = subprocess.Popen(
            ["efsmt -clock list |grep 'HBM PHY' |awk -F' ' '{print $6}'"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        hbm_phy = proc.communicate()[0].strip()
        hbm_phy = hbm_phy.decode('utf-8')
    except Exception as ex:
        # traceback.print_exc()
        hbm_phy = None
    return hbm_phy


def get_environment_info():
    base_dict = OrderedDict([('python', "%i.%i.%i" % get_python_version()),
                             ('cpu_cores', get_host_cores()),
                             ('cpu_model_name', get_host_model()),
                             ('total_mem', "{}MB".format(get_host_mem())),
                             ('host_name', get_host_name()),
                             ('host_ip', get_local_ip()),
                             ('os_name', get_os_name()),
                             ])
    freq = get_gcu_clock_freq()
    if freq:
        base_dict.update({'gcu_clock': '{}MHz'.format(freq)})
    hbm = get_hbm_phy()
    if hbm:
        base_dict.update({'HBM_PHY': '{}Mbps'.format(hbm)})
    return base_dict

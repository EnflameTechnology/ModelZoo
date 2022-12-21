#!/usr/bin/env bash
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1 --tf_xla_min_cluster_size=4"
export ENFLAME_UMD_FLAGS="ib_pool_size=134217728"
export ENFLAME_MEMUSE_POLICY=performance

export ENFLAME_DEVICE_MODE=ONEDEVICE_EX
export ENFLAME_COMPILE_OPTIONS_HLIR="hlir-pipeline{tensor-split=true}"
export ENFLAME_CLUSTER_PARALLEL=true
export ENFLAME_ENABLE_TF32=true
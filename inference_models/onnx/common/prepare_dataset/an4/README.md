# AN4 数据集准备

## 步骤

1、将`an4`数据集的压缩包下载至本地

```bash
wget "https://huggingface.co/datasets/espnet/an4/resolve/main/an4_sphere.tar.gz"
```

2、安装需要的依赖库

```bash
pip3 install -r requirements.txt
```

3、数据预处理

```bash
python3 sph2wav.py --dir_path=<path/to/the/directory/you/containing/an4_sphere.tar.gz>
python3 build_mainfest.py --dataset_path=<path/to/the/directory/you/containing/an4test_clstk> --dir_path=<path/to/the/directory/you/output/test_manifest.json>
```

## 准备好之后的目录结构

```bash
data/
└── an4
     ├── etc
     |     ├──an4_test.transcription
     |     └── ...
     ├── wav
     |    ├── an4_clstk
     |    └── an4test_clstk
     └──  test_manifest.json
```


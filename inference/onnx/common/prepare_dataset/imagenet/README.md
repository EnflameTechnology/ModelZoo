# Imagenet 数据集准备
## 步骤
1. 从 https://image-net.org/challenges/LSVRC/2012/ 下载 ILSVRC2012_img_val.tar （需要注册）
2. 解压

    ```bash
    mkdir val
    tar -xvf ILSVRC2012_img_val.tar -C val/
    ```

3. 下载标签

    ```bash
    wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
    wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
    ```

4. 将图片放到分类文件夹中（可选）

    ```bash
    python3 preprocess_imagenet_validation_data.py val/ imagenet_2012_validation_synset_labels.txt imagenet_lsvrc_2015_synsets.txt
    cp imagenet_2012_validation_synset_labels.txt val/synset_labels.txt
    ```

5. 生成 val_map.txt

    ```bash
    python3 convert_imagenet.py val/ imagenet_2012_validation_synset_labels.txt imagenet_lsvrc_2015_synsets.txt val/val_map.txt
    ```

6. 重命名

    ```bash
    mv val data
    ```

## 准备好之后的目录结构
```
data
   ├── n01440764
   │   ├── ILSVRC2012_val_00000293.JPEG
   │   ├── ILSVRC2012_val_00002138.JPEG
   |   └── ……
   ……
   └── val_map.txt
```

val_map.txt 包括图片路径和分类id的对应关系

```
./n01751748/ILSVRC2012_val_00000001.JPEG 65
./n09193705/ILSVRC2012_val_00000002.JPEG 970
./n02105855/ILSVRC2012_val_00000003.JPEG 230
./n04263257/ILSVRC2012_val_00000004.JPEG 809
……
```
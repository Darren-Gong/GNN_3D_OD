## 环境配置

### 先决条件
在Python=3.6的环境下
```
conda install tensorflow-gpu==1.15.0
pip install tensorflow-estimator==1.15.0
```

安装其他依赖项: 
```
conda install py-opencv
pip install open3d-python==0.7.0.0
pip install scikit-learn
pip install tqdm
pip install shapely
```

### KITTI Dataset

使用KITTI 3D物体检测数据集 [here](https://blog.csdn.net/qq_16137569/article/details/118873033). 其中Velodyne点云文件、左视RGB图像链接[夸克网盘](https://pan.quark.cn/s/9f59ca7b5628)。提供了额外的拆分文件。 [splits/](splits). 文件结构:

    DATASET_ROOT_DIR
    ├── image                    #  Left color images
    │   ├── training
    |   |   └── image_2            
    │   └── testing
    |       └── image_2 
    ├── velodyne                 # Velodyne point cloud files
    │   ├── training
    |   |   └── velodyne            
    │   └── testing
    |       └── velodyne 
    ├── calib                    # Calibration files
    │   ├── training
    |   |   └──calib            
    │   └── testing
    |       └── calib 
    ├── labels                   # Training labels
    │   └── training
    |       └── label_2
    └── 3DOP_splits              # split files.
        ├── train.txt
        ├── train_car.txt
        └── ...


## Inference
### Run a checkpoint
在验证分割上进行测试:
```
python run.py checkpoints/car_auto_T3_train/ --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```
在测试数据集上进行测试:
```
python run.py checkpoints/car_auto_T3_trainval/ --test --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```

```
usage: run.py [-h] [-l LEVEL] [--test] [--no-box-merge] [--no-box-score]
              [--dataset_root_dir DATASET_ROOT_DIR]
              [--dataset_split_file DATASET_SPLIT_FILE]
              [--output_dir OUTPUT_DIR]
              checkpoint_path

Point-GNN inference on KITTI

positional arguments:
  checkpoint_path       Path to checkpoint

optional arguments:
  -h, --help            show this help message and exit
  -l LEVEL, --level LEVEL
                        Visualization level, 0 to disable,1 to nonblocking
                        visualization, 2 to block.Default=0
  --test                Enable test model
  --no-box-merge        Disable box merge.
  --no-box-score        Disable box score.
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split
                        file.Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"
  --output_dir OUTPUT_DIR
                        Path to save the detection
                        resultsDefault="CHECKPOINT_PATH/eval/"
```
### Performance
安装kitti_native_evaluation离线评估:
```
cd kitti_native_evaluation
cmake ./
make
```
在验证分割上评估输出结果:
```
evaluate_object_offline DATASET_ROOT_DIR/labels/training/label_2/ DIR_TO_SAVE_RESULTS
```

## Training
将训练参数放在train_config文件中。要开始训练，需要train_config和config。
```
usage: train.py [-h] [--dataset_root_dir DATASET_ROOT_DIR]
                [--dataset_split_file DATASET_SPLIT_FILE]
                train_config_path config_path

Training of PointGNN

positional arguments:
  train_config_path     Path to train_config
  config_path           Path to config

optional arguments:
  -h, --help            show this help message and exit
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split file.Default="DATASET_ROOT
                        _DIR/3DOP_splits/train_config["train_dataset"]"
```
For example:
```
python3 train.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config
```
强烈建议在开始训练之前查看train_config。可能想要首先更改的一些常见参数:
```
train_dir     The directory where checkpoints and logs are stored.
train_dataset The dataset split file for training. 
NUM_GPU       The number of GPUs to use. We used two GPUs for the reference model. 
              If you want to use a single GPU, you might also need to reduce the batch size by half to save GPU memory.
              Similarly, you might want to increase the batch size if you want to utilize more GPUs. 
              Check the train.py for details.               
```
提供了一个评估脚本来定期评估检查点. For example:
```
python3 eval.py configs/car_auto_T3_train_eval_config 
```
可以使用tensorboard查看培训和评估状态。
```
tensorboard --logdir=./train_dir
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details



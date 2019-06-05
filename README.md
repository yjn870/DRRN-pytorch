# DRRN

This repository is implementation of the ["Image Super-Resolution via Deep Recursive Residual Network"](http://cvlab.cse.msu.edu/project-super-resolution.html).

## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Prepare

The images for creating a dataset used for training (**291-image**) or evaluation (**Set5**) can be downloaded from the paper author's [implementation](https://github.com/tyshiwo/DRRN_CVPR17/tree/master/data).

You can also use pre-created dataset files with same settings as the paper.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 291-image | 2, 3, 4 | Train | [Download](https://www.dropbox.com/s/w67yqju1suxejxn/291-image_x234.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/b4a48onyqedx8dz/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/if01dprb3tzc8jr/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/cdoxdgz99imy9ik/Set5_x4.h5?dl=0) |

### Generate training dataset

```bash
python generate_trainset.py --images-dir "BLAH_BLAH/Train_291" \
                            --output-path "BLAH_BLAH/Train_291_x234.h5" \
                            --patch-size 31 \
                            --stride 21
```

### Generate test dataset

```bash
python generate_testset.py --images-dir "BLAH_BLAH/Set5" \
                           --output-path "BLAH_BLAH/Set5_x2.h5" \
                           --scale 2
```

## Train

Model weights will be stored in the `--outputs-dir` after every epoch.

```bash
python train.py --train-file "BLAH_BLAH/Train_291_x234.h5" \
                --outputs-dir "BLAH_BLAH/DRRN_B1U9" \
                --B 1 \
                --U 9 \
                --num-features 128 \
                --lr 0.1 \
                --clip-grad 0.01 \
                --batch-size 128 \
                --num-epochs 50 \
                --num-workers 8 \
                --seed 123
```

You can also evaluate using `--eval-file`, `--eval-scale` options during training after every epoch. In addition, the best weights file will be stored in the `--outputs-dir` as a `best.pth`.

```bash
python train.py --train-file "BLAH_BLAH/Train_291_x234.h5" \
                --outputs-dir "BLAH_BLAH/DRRN_B1U9" \
                --eval-file "BLAH_BLAH/Set5_x2.h5" \
                --eval-scale 2 \
                --B 1 \
                --U 9 \
                --num-features 128 \
                --lr 0.1 \
                --clip-grad 0.01 \
                --batch-size 128 \
                --num-epochs 50 \
                --num-workers 8 \
                --seed 123
```

## Evaluate

The pre-trained weights can be downloaded from the following links.

| Model | Link |
|-------|------|
| DRRN_B1U9 | [Download](https://www.dropbox.com/s/1ozete9panliycb/drrn_x234.pth?dl=0) |

```bash
python eval.py --weights-file "BLAH_BLAH/DRRN_B1U9/best.pth" \
               --eval-file "BLAH_BLAH/Set5_x2.h5" \
               --eval-scale 2 \
               --B 1 \
               --U 9 \
               --num-features 128               
```

## Results

The our model was learned and evaluated on the **Y(luminance) channel**.

For performance, we modified the original implementation as follows. 

- **Batch normalization** was removed from the residual unit.
- **No bias** was used in the convolution layer.

### Performance comparision on the Set5

| Eval. Mat | Scale | DRRN_B1U9 (Paper) | DRRN_B1U9 (Ours) |
|-----------|-------|-------|-----------------|
| PSNR | 2 | 37.66 | **37.62** |
| PSNR | 3 | 33.93 | **33.86** |
| PSNR | 4 | 31.58 | **31.52** |

## References

1. [https://github.com/tyshiwo/DRRN_CVPR17](https://github.com/tyshiwo/DRRN_CVPR17)
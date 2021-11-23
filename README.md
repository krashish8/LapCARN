# Laplacian Cascading Networks for Image Super-Resolution

### Dataset

- Training: DIV2K dataset, download and place in dataset/ directory. Convert to h5 format by running the `dataset/div2h5.py` file.
- Testing: Set5, Set14, B100, Urban100 and Manga100 dataset.

### Testing

We provide the models in `checkpoint` directory. For testing, run the [eval.ipynb](./eval.ipynb) file using Jupyter Notebook and adjust accordingly.

### Training

```shell

python3 carn/train.py --patch_size 64
                      --batch_size 8
                      --max_steps 600000
                      --decay 400000
                      --model lapcarn
                      --ckpt_name lapcarn
                      --ckpt_dir checkpoint/lapcarn
                      --scale 8
                      --num_gpu 2
                      --print_interval 100
```

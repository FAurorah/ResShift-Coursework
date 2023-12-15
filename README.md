# ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting (NeurIPS 2023, Spotlight) 

## Inference
#### :tiger: Real-world image super-resolution
```
CUDA_VISIBLE_DEVICES=gpu_id python inference_resshift.py -i [image folder/image path] -o [result folder] --scale 4 --task realsrx4 --chop_size 512
```
#### :lion: Bicubic (resize by Opencv) image super-resolution
```
CUDA_VISIBLE_DEVICES=gpu_id python inference_resshift.py -i [image folder/image path] -o [result folder] --scale 4 --task bicsrx4_opencv --chop_size 512
```
#### :lion: Bicubic (resize by Matlab) image super-resolution
```
CUDA_VISIBLE_DEVICES=gpu_id python inference_resshift.py -i [image folder/image path] -o [result folder] --scale 4 --task bicsrx4_matlab --chop_size 512
```

## Training
#### :turtle: Prepare data
Download the training data and add the data path to the config file (data.train.params.dir_path or data.train.params.txt_file_path). To synthesize the testing dataset utilized in our paper, please refer to these [scripts](./scripts/).
* Real-world and Bicubic image super-resolution: [ImageNet](https://www.image-net.org/) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) (resized to 256x256) 
#### :dolphin: Real-world Image Super-resolution
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --cfg_path configs/realsr_swinunet_realesrgan256.yaml --save_dir [Logging Folder] --steps 15
```
#### :whale: Bicubic Image Super-resolution
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --cfg_path configs/bicubic_swinunet_bicubic256.yaml --save_dir [Logging Folder]  --steps 15
```

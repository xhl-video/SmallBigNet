# SmallBigNet


This repo is the official implementation of our paper ["SmallBigNet: Integrating Core and Contextual Views for Video Classification (CVPR2020)"](https://arxiv.org/abs/2006.14582).

## Citation


```
@inproceedings{li2020smallbignet,
  title={Smallbignet: Integrating core and contextual views for video classification},
  author={Li, Xianhang and Wang, Yali and Zhou, Zhipeng and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

```



## Usage

### Data Preparation
First, please follow the [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README.md) to prepare data. Note that our codebase only supports the **RGB** frames. Thus you may need to decord the video dataset offline and store it in SSD.
If you need the Kinetics-400 dataset, please feel free to email me.
(Tips: if you want to use video online decode, highly recommend you to use the mmaction2. Our idea is simple so only a few codes need to change in [resnet3d.py](https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/backbones/resnet3d.py) ) 


### K400 Training Scripts

After you prepare the dataset, edit the parameters in  ``scripts/kinectis.sh``.

``
--half indicates using mix precision
``

``
--root_path the path you store the whole dataset(RGB)
``

``
--train_list_file the train list file (video_name num_frames label)
``


``
--val_list_file the val list file (video_name num_frames label)
``

``
--model_name  [res50, slowonly50, slowonly50_extra, smallbig50_no_extra,smallbig50_extra, smallbig101_no_extra]
``

``
--image_tmpl  the format of the name you store the RGB frames like img_{:05d}.jpg
``


----------------

If you have any question about the code and data, please contact us directly.



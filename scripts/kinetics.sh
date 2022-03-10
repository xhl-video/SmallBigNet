




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8  python -m torch.distributed.launch --nproc_per_node=8  main.py \
--distribute --half  --num_classes 400 --batch_size 64 --t_length 8 --t_stride 8 --image_tmpl img_{:05d}.jpg \
--dataset kinetics --root_path /data2/data/kinetics_400/rawframes_320  --val_list_file /data1/data/kinetics_400/RGB_val_videofolder.txt  --train_list_file /data1/data/kinetics_400/RGB_train_videofolder.txt \
--model_name smallbig50_no_extra --lr 0.01 --num_epochs 100 --check_dir ./checkpoint  --log_dir  ./logs


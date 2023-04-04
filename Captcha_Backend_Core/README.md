# Reactionary Models


| Backbone| Link |
|--------------|-------|
| vgg_seq2seq | https://drive.google.com/file/d/1-2ukNhfd_Wpg6qpmlFzKy6eXq1bqQgGz/view?usp=sharing|
| vgg_transformer |  https://drive.google.com/file/d/1-O7FnkW10WbHmXl51y2p-C9Q0b_kJ9Jw/view?usp=sharing|

# Capcha Models


| Backbone| Link |
|--------------|-------|
| vgg_seq2seq | https://drive.google.com/file/d/1tTVpjYEb46ZkxrX_JztmSQbXEzKim4HO/view?usp=sharing|
| vgg_transformer |  https://drive.google.com/file/d/1KFBlcJxZQ2u8uULyPIFRo-Y9-9CPPVqI/view?usp=sharing|

## Train 

```
!python3 train_ocr.py --config ${Path_to_your_config_yml_file} \
                      --data-root ${Path_to_your_dataset_folder} \
                      --train ${Path_to_your_train_file} \
                      --test ${Path_to_your_test_file} \
                      --num-epochs ${Num_epochs} \
                      --batch-size ${Batch_size} \
                      --max-lr ${Learning_rate} \
                      --export {Path_to_your_weight} \
 ```
 
 ### Example

```
!python3 train_ocr.py --config 'config_vgg_transformer.yml' \
                      --data-root './dataset/ocr/data_line/' \
                      --train 'train_line_annotation.txt' \
                      --test 'train_line_annotation.txt' \
                      --num-epochs 20000 \
                      --batch-size 32 \
                      --max-lr 0.0003 \
                      --export './weights/transformerocr.pth' \
 ```
 ## Test
 
  ```
!python3 train_ocr.py --config ${Path_to_your_config_yml_file} \
                      --data-root ${Path_to_your_dataset_folder} \
                      --test ${Path_to_your_test_file} \
                      --weight {Path_to_your_weight}
 ```
 
 ### Example
 
 ```
!python3 train_ocr.py --config 'vgg_transformer' \
                      --data-root './dataset/ocr/data_line/' \
                      --test 'train_line_annotation.txt' \
                      --weight './vietocr/weights/transformerocr.pth'
 ```
 ## Infer
 
 ```
 python3 demo_ocr.py\
        --img ${Path_to_your_image}\
        --config ${Path_to_your_config_yml_file} \
        --weight ${Path_to_your_weight}
 ```

# Deploy

```

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
RUN pip3 install einops
RUN pip3 install gdown
RUN pip3 install matplotlib
RUN pip3 install imgaug
RUN pip3 install PyYAML
RUN git clone -b ocr https://github.com/TrinhThiBaoAnh/Reaction.git
WORKDIR /Reaction
RUN pip install -r requirements.txt

####  Adding OCR models ####
RUN mkdir -p /Reaction/vietocr/weights
ADD https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4 /Reaction/vietocr/weights
ADD https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA /Reaction/vietocr/weights
```

# OOSS
Omnisupervised Omnidirectional Semantic Segmentation

## Datasets
[**PASS Datast**](https://drive.google.com/file/d/1A_P2u5HUbrHZnKJYAOL2f7JLxxj69LqB/view?usp=sharing)
Panoramic Annular Semantic Segmentation Dataset with pixel-wise labels (400 images).

[**Chengyuan Dataset**](https://drive.google.com/file/d/1xMUeptlceWpjLmqUKeOasRmGg1J9QF-h/view?usp=sharing)
Panoramas captured with an instrumented vehicle (650 images).

[**Streetview Dataset**](https://drive.google.com/file/d/1_tZiYdRCQASJhNiR6MAPC_P_F1EbpEw0/view?usp=sharing)
Panoramas collected in different cities including New York, Beijing, Shanghai, Changsha, Hangzhou, Huddersfield, Madrid, Karlsruhe and Sydney.

![Example segmentation](figure_psv.jpg?raw=true "Example segmentation")

## Codes
Training:
```
CUDA_VISIBLE_DEVICES=0,1,2,3
python3 segment.py
--basedir /home/kyang/Downloads/
--num-epochs 200
--batch-size 12
--savedir /erfpsp
--datasets 'MAP' 'IDD20K'
--num-samples 18000
--alpha 0
--beta 0
--model erfnet_pspnet
```

Evaluation:
```
python3 eval_color.py
--datadir /home/kyang/Downloads/Mapillary/
--subset val
--loadDir ./trained/
--loadWeights model_best.pth
--loadModel erfnet_pspnet.py
--basedir /home/kyang/Downloads/
--datasets 'MAP' 'IDD20K'
```

# Training
## Pase
pase 공식 github 따라 하면 돼.
wavlm 학습: python -m train.train_wavlm -C configs/cfg_train_wavlm.yaml 
vocos 학습: python -m train.train_vocoder_dual -C configs/cfg_train_vocoder_dual.yaml
vocos 학습시킬 때 cfg_train_vocoder_dual.yaml에 wavlm_ckpt_path, cfg_path 를 wavlm 학습 후 생성된 checkpoint와 학습시 사용한 config 주소 넣기.

## VAE
python train.py 
학습 중간에 끊기면, 
python train.py --load output/checkpoint_epoch88.pth --start-epoch 89 
이렇게 epoch 설정 해야함.

## LCTGAN
python train_dns_la1n.py
학습 중간에 끊기면,
python train_dns_la1n.py --resume-epoch 171

# Inference
## Pase
python -m inference.infer_vocoder_dual --output_dir pase_result_e27_best
python -m inference.infer_vocoder_dual_dns --output_dir pase_result_e27_best_dns
configs/cfg_infer.yaml에서 exp_path에 vocos 경로 넣기. 

## VAE
python test_teacher_batch.py
python test_teacher_dns.py

## LCTGAN
python eva_dns_a1n_mos_gan_test.py
python eva_dns_challenge_test.py

pretrained models: https://drive.google.com/drive/u/0/folders/1ADP32jXesY0xyopbIPGYHCOInZi3UdLQ

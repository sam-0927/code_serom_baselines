# Training
## Pase
pase 공식 github 따라 하면 돼.
wavlm 학습: python -m train.train_wavlm -C configs/cfg_train_wavlm.yaml 
vocos 학습: python -m train.train_vocoder_dual -C configs/cfg_train_vocoder_dual.yaml
vocos 학습시킬 때 cfg_train_vocoder_dual.yaml에 wavlm_ckpt_path, cfg_path 를 wavlm 학습 후 생성된 checkpoint와 학습시 사용한 config 주소 넣기.

## VAE
* 이게 teacher model 중에 predictive model 만 학습시키는 것. 그리고 모델 사이즈도 논문과 달랐음.
python train.py 
학습 중간에 끊기면, 
python train.py --load output/checkpoint_epoch88.pth --start-epoch 89 
이렇게 epoch 설정 해야함.

* 아래가 진짜 논문과 동일한 모델.
python train_predicive_model.py
먼저 하고, checkpoint_200.pth를 best 다신 사용해서 cvae 학습에 사용. 어차피 별 차이 없었음.
python train_cvae_model.py
학습 중간에 끊기면,
python train.py --load output_cvae_model/checkpoint_epoch165.pth 까지만 해도 알아서 감.

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
* 이전 모델
python test_teacher_batch.py
python test_teacher_dns.py
* 찐 모델. 이게 논문에 서술.
python test_cvae.py
python test_cvae_dns.py

## LCTGAN
python eva_dns_a1n_mos_gan_test.py
python eva_dns_challenge_test.py

pretrained models: https://drive.google.com/drive/u/0/folders/1ADP32jXesY0xyopbIPGYHCOInZi3UdLQ

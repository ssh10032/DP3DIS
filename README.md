## Instance Position Encoding and Dual Denoising Tasks for Efficient 3D Instance Segmentation
## 효율적인 3차원 개체 분할을 위한 개체 위치 인코딩과 이중 노이즈 제거 작업들


## 논문 링크
- Version 1 : <a href="https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11621139&googleIPSandBox=false&mark=0&ipRange=false&b2cLoginYN=false&aiChatView=B&readTime=15-20&isPDFSizeAllowed=true&accessgl=Y&language=ko_KR&hasTopBanner=true">제어로봇시스템학회 논문지 T3DIS</a>
- Version 2 : <a href="https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11758934&googleIPSandBox=false&mark=0&ipRange=false&b2cLoginYN=false&aiChatView=B&readTime=15-20&isPDFSizeAllowed=true&accessgl=Y&language=ko_KR&hasTopBanner=true">멀티미디어학회 논문지 DP3DIS</a>


## 개발 환경
- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
- OS : Ubuntu 18.04.6LTS
- GPU : GeForce GTX 3090 GPU X2


## 가상 환경 설정
Python, Cuda 버전 
```yaml
python: 3.10.9
cuda: 11.3
```
가상 환경 생성 및 라이브러리 설치
```
conda env create -f environment.yml

conda activate DP3IDS_cuda113

pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

mkdir third_party
cd third_party

git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

cd ..
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make

cd ../../pointnet2
python setup.py install

cd ../../
pip3 install pytorch-lightning==1.7.2
```
### 데이터 전처리 :hammer:


#### ScanNet / ScanNet200
세부 사항 참고: [original repository](https://github.com/ScanNet/ScanNet/tree/master/Segmentator)
```
python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="PATH_TO_RAW_SCANNET_DATASET" \
--save_dir="data/processed/scannet" \
--git_repo="PATH_TO_SCANNET_GIT_REPO" \
--scannet200=false/true
```
### 학습 및 평가 :train2:
학습
```bash
python main_instance_segmentation.py
```
추론 및 평가
```bash
python main_instance_segmentation.py \
general.checkpoint='가중치_이름.ckpt' \
general.train_mode=false
```

# dacon study 1

 * 참여일: 23.04.18 ~ 23.04.24
 * 목적: 가볍게 스타트를 끊어보자
 * 주제: 생체 데이터로부터 칼로리 소모량을 예측하는 AI 알고리즘 개발
 * 경진대회 링크: [dacon link](https://dacon.io/competitions/official/236097/overview/description)
 
## description
.
├── README.md
├── best_results                     ## 모델링 결과
│   ├── mlpregressor-23-04-24-3.csv
│   ├── mlpregressor-23-04-24-4.csv
│   └── mlpregressor-23-04-24-5.csv
├── data-mining.ipynb                ## 데이터 탐구
├── mlpregressor.ipynb               ## mlp 모델링
├── open                             ## 공개 데이터
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
└── various-approaches.ipynb         ## 다양한 모델 시도

3 directories, 10 files
 
## best_results (test RMSE)
* mlpregressor-23-04-24-5.csv: 0.3641213005
* mlpregressor-23-04-24-4.csv: 0.3195398415 ## best 31st
* gradientboost-23-04-23-3.csv: 0.4793217297
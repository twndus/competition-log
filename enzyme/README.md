# Explore Multi-Label Classification with an Enzyme Substrate Dataset

# Memo
- 새로이 추가한 feature는 효과가 있었음.

# Submissions
1. knn_baseline	- train:0.6170, public:0.5647
1. mlp_pure	- train:0.6341, public:0.5703
1. xgboost_pure	- train:0.5686, public:0.5676
1. rf_pure	- train:1.0000, public:0.5689
1. extra_pure	- train:0.5586, public:0.5526
1. mlp_fe	- train:0.5692, public:0.5747

# best public scores
1. xgboostmlp	- train:0.985,	public:0.5975 # smote, 10 fold
2. xgboostmlp	- train:0.985,	public:0.5965


# TODO

아래 두 방법은 실제로 성능을 올리는 데에 효과가 있었음
1. ensemble with library mlp 10 - OK
2. augmentation ! oversampling! - OK

python ../train_evaluate.py --dataset 'SIDER' --epoch 200 --batch_size 64 --drop_rate 0.4 --dist 0.08 --hid_dim 96 --attention True --step 4 --agg_op mean --learning_rate 0.0001 --folds 10 --device cuda:0 --heads 4

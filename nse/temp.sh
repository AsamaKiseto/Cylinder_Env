# bash gen_data.sh
# cd data/test_data
# python3 merge_data.py
# cd ..
# cd ..

# python3 4_test.py

bash gen_data.sh
python3 2_nse_operator_grid_rbc.py --phys_gap 500 --phys_epochs 0 -lf data_based
python3 2_nse_operator_grid_rbc.py -lf phys_inc
python3 2_nse_operator_grid_rbc.py --phys_random_select True --phys_scale 0.001 -lf random_select
python3 2_nse_operator_grid_rbc.py --phys_scale 0 -lf no_random

python3 4_test.py

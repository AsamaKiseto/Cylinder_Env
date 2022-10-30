# bash gen_data.sh
# cd data/test_data
# python3 merge_data.py
# cd ..
# cd ..

# python3 4_test.py

python3 2_nse_operator_grid_rbc.py --phys_gap 500 --phys_epochs 0 -lf data_based
python3 2_nse_operator_grid_rbc.py -lf phys_inc

python3 2_nse_operator_grid_rbc.py --phys_gap 40 --phys_epochs 5 -lf pe_5
python3 2_nse_operator_grid_rbc.py --phys_gap 40 --phys_epochs 15 -lf pe_15

python3 2_nse_operator_grid_rbc.py --phys_scale 0.01 -lf ps_0.01
python3 2_nse_operator_grid_rbc.py --phys_scale 0.05 -lf ps_0.05
python3 2_nse_operator_grid_rbc.py --phys_scale 0.2 -lf ps_0.2

python3 2_nse_operator_grid_rbc.py --phys_random_select True --phys_scale 0.001 -lf random_select_0.001
python3 2_nse_operator_grid_rbc.py --phys_random_select True --phys_scale 0.0001 -lf random_select_0.0001
python3 2_nse_operator_grid_rbc.py --phys_scale 0 -lf no_random
python3 2_nse_operator_grid_rbc_test.py -lf pre_phys

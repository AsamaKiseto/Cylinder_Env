# python3 1_data_gen.py -fr 4.0

# for fb in 2.0 3.0 4.0 5.0; do
#     python3 1_data_test_gen.py -fb $fb -s 1.0
# done

# python3 1_data_gen_rbc.py
python3 4_test.py
python3 1_data_gen_rbc.py
python3 1_data_gen_rbc2.py
python3 1_data_test_gen_rbc.py

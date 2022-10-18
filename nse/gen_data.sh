for s in 0.1 0.5 1.0; do
    python3 1_data_test_gen.py -fb 0.0 -s $s 
done

cd data/test_data
python3 merge_data.py
cd ..
cd ..
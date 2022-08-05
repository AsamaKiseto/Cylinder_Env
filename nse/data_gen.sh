for f1 in -2 0; do
    for f2 in -2 0; do
        python3 -u 1_data_gen.py --f1 $f1 --f2 $f2 &
    done
done
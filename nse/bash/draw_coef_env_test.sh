for scale in 0.1 0.5; do
    python3 4_draw_coef_env.py -s $scale -nt 80 -ts 5 -f 1 -lf compare
done
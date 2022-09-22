
python3 4_draw_coef_env.py -s 0.0 -nt 80 -ts 5 -f 0.14285714285714285 -lf 1
python3 4_draw_coef_env.py -s 0.0 -nt 80 -ts 5 -f 1 -lf 2

for k in {1..10}; do
    bash draw_coef_env.sh
done

echo "Start train......"
nohup python -u train.py --number_of_segments=1 > train.log 2>&1 &
wait
echo "Finished......"

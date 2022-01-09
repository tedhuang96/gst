for dataset in eth hotel univ zara1 zara2; do
    python -u scripts/create_datasets.py --dataset $dataset | tee -a logs/create_datasets.txt
done
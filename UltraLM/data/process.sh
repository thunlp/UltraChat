python -u split_long.py --in-file ./raw/part1.json --out-file ./processed/part1.json --begin 0 --model-name-or-path huggyllama/llama-7b --max-length 2048
python -u split_long.py --in-file ./raw/part2_1.json --out-file ./processed/part2_1.json --begin 0 --model-name-or-path huggyllama/llama-7b --max-length 2048
python -u split_long.py --in-file ./raw/part2_2.json --out-file ./processed/part2_2.json --begin 0 --model-name-or-path huggyllama/llama-7b --max-length 2048
python -u split_long.py --in-file ./raw/part3.json --out-file ./processed/part3.json --begin 0 --model-name-or-path huggyllama/llama-7b --max-length 2048

python3 testAdvBench.py --num_samples=520 --model_type=llama-3 > terminal.log
python3 LLM_evalutils.py --input_csv=results/responses.csv > LLMRefusal_evalPerf.txt
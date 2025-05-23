#!/bin/bash

python ~/elarablate/merge_lora.py $1 ~/elarablate_script_test ~/$1-Elarablated-script-test --scale $2 --layer_range 0-79
python ~/llama.cpp/convert_hf_to_gguf.py --outtype f16 ~/$1-Elarablated-script-test --outfile ~/elarablate_script_test_fp16.gguf
~/llama.cpp/build/bin/llama-quantize ~/elarablate_script_test_fp16.gguf ~/elarablate_script_test_Q6_K.gguf Q6_K
~/koboldcpp-linux-x64-cuda1210 ~/ElarablateTest.kcpps

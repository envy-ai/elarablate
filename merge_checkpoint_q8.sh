#!/bin/bash

#python ~/elarablate/merge_lora.py $1 ~/elarablate_script_test ~/$1-Elarablated-script-test --scale $2 --layer_range 0-79
TMPDIR=/var/tmp python ~/llama.cpp/convert_hf_to_gguf.py --outtype q8_0 ~/$1-Elarablated-script-test --outfile /home/envy/tmp/elarablate_script_test_q8_0.gguf
~/koboldcpp-linux-x64-cuda1210 ~/ElarablateTest_q8.kcpps

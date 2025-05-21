# Elarablation - Overview
First off, things are super preliminary here.  It's late, and I wanted to post my proof of concept.  There's a lot of work to be done on this repo, so excuse how barebones it is.

"Elarablation" is my method for surgically excising undesirable slop tokens.  The way that it works is that you train a QLoRA by caching a context where the slop that you want to get rid of starts with the next likely token.  For instance, rather than training this entire passage:

*The elf girl nods. "My name is [some non-slop name]"*

I load everything up to the name into the context window without training on it:

*The elf girl nods. "My name is*

Then we examine the top X tokens (50 by default in this script).  We then identify which tokens are "good" or "bad".  In this case, identification is easy because we want tokens that start with a space followed by a capital letter.  Any "bad" tokens among the top 50 are punished, as are any tokens above a set maximum probability threshold.  "Good" tokens below a minimum probability threshold are rewarded.  This can all happen very rapidly because we can train rewards and punishments for a large number of tokens with a single forward pass, accumulating the loss for each token and applying it all in one go.

Before the training, the tokens may look like this:
  
    ' El': 0.154
    ' Ar': 0.113
    'ara': 0.011
    ' Cy': 0.009
    ' En': 0.006
    [... and so on ...]

Each epoch, we determine which tokens are above the max threshold (assuing 0.02 here) and punish them along with any bad tokens, then reward all the good tokens below the minimum threshold (say 0.15).  Over a number of epochs, you end up with something that looks like this:

    ' En': 0.019
    ' Ali': 0.018
    ' Cal': 0.018
    ' Ar': 0.018
    ' El': 0.018
    [... a bunch more entries here ...]
    ' An': 0.015
    ' R': 0.014

Since this is trained in the specific context of introducing a name, it doesn't meaningfully affect the probability of these tokens elsewhere in text.  Furthermore, it doesn't seem to have any serious negative effect on remembering characters who are already introduced.  If there is a character already named Elara, it will continue to use that name consistently.

# Installation

    pip install -r requirements.txt

# Example command line

    python elarablate.py --model_name_or_path Steelskull/L3.3-Electra-R1-70b --contexts_folder slopcontext --output_dir electra_elarablate_lora --lr 7e-5 --epochs 100 --threshold 0.02 --low_threshold 0.015 --top_k 50

# Preliminary tests

In my preliminary tests, this method reduced the incidence of the name "Elara" in a specific context from 40% of the time (much of the remaining time the names were either Aria, Lyra, or Althaea) to about 4% of the time, with a much larger and richer variety of names. Coherence was not noticeably affected (at least in runs of 500-1000 tokens) except in the case of the token " Am", which seemed not to have any reasonably follow-up and would consistently induce repetition.  I imagine this could easily be fixed by training a "tree" one level deeper to make sure the subsequent token makes sense as well and eliminate that kind of repetition (as well as adding further variety to generated names).

I did my test training on Steelskull's Electra 70B model:

https://huggingface.co/Steelskull/L3.3-Electra-R1-70b

[update with my own huggingface repo after uploading test data]

Training was done on an A100 and I believe it requires about 40G of VRAM for a 70B model, give or take.  Due to the nature of the training process (and the fact that it took two orders of magnitude longer to convert and then quantize the model than it did to train), it may actually be practical to train large models on a CPU, although I haven't tested this.

I merged the lora into the model with a slightly modified version of this script with the scale factor doubled:

https://github.com/tdrussell/qlora-pipe/blob/main/tools/merge_lora.py

I'll clean up my version and add it to the repo later.

# Plans

This initial proof of concept only works on names. However, I believe that this method can also be used to target specific single overused tokens (despite the name of the process, it doesn't speficially target "Elara") in order to vastly reduce dialogue related to shivering spines, voices barely above a whisper, etc.  It may even be possible (with a combination of math and review by an LLM) to detect overfit cliches and train them out automatically, although no promises there.  Most likely, it'll be easier just to identify cliches manually and synthesize context/training data to train them out.


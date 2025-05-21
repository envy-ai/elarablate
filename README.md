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


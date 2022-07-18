---
layout: post
title: "[Thoughts]-On the interpretability conundrum"  
author: krunal kshirsagar
published: true
---

Of all the unknowable information, you can only get as much unknowable information out of 
a  system  as  axioms  you  put  in.  Better  understanding depends on  the  amount  of  input 
axioms. Better prediction requires better understanding. And better understanding makes a 
system  more  interpretable.  Higher  the  interpretability  of a system  the  higher  we  trust the 
system. Interpretability focuses on understanding the cause of the decision while inference 
focuses on the conclusion reached on the basis of evidence & reasoning(Causal & Bayesian).


However there’s a general notion of trade-off between the accuracy and interpretability of the 
system let alone the privacy aspect. The most affected domain by this trade-off is the medical domain. The 
problem is  that the  systems  are heavily biased; be it racial, gender or any other biases that 
you can think of. The ML models can predict self-reported race even from corrupted, cropped, and 
noised medical images as opposed to medical experts. 
These  ML  models  are  making accurate decisions  about  racial  classification using features that humans can't even notice carefully, let alone analyse. {% cite readingracebanerjee2021 %}
<br>
<img src="{{ site.baseurl }}/images/2022-01-01-trilemma-or-trilogy.png"
style="float: right; max-width: 50%; margin: 0 0 1em 2em;"> 
In order to trust the system 
we need to break things down like in the first principles approach and then make a ground-up 
approach  to  interpretability/reasoning.  Hence,  I  believe,  mathematical  methods  like  causal 
inference, differential & algebraic geometry, topology, stability theory, probabilistic methods, 
PDEs, information geometry &  algorithmic information theory can help  achieve better interpretability. I 
believe with mathematical proofs and truths we can achieve inductive/abductive reasoning and 
inference  while  discarding  the  role  of  medical  domain  experts  because  humans  have  a 
tendency to lie, mathematics doesn’t lie. Math was already there,  humans just discovered it 
they  didn’t  invent  it.  Math  should  be  the  base  of  the  prediction,  inference  and  reasoning 
instead of a ‘Domain level expert human being’. Thus, in my opinion the model should learn 
from the mathematical proofs instead (learn from nature, don’t learn from humans).

<!---

False positives, connectomes

Roses aren't red,<br>
The sky isn't blue,<br>
It's your perception you idiot,<br>
that's messing up with you.<br>

**Even Shane Warne knows why interpretability is important :p**

<blockquote class="twitter-tweet" tw-align-center><p lang="en" dir="ltr">This is simply - not out !!!!! We often discuss technology &amp; its use / accuracy. The main problem@is the interpretation of the technology. Here’s a perfect example of the ball clearly hitting the edge of the bat first. <a href="https://t.co/OATRzIHcfg">https://t.co/OATRzIHcfg</a></p>&mdash; Shane Warne (@ShaneWarne) <a href="https://twitter.com/ShaneWarne/status/1466958968735952897?ref_src=twsrc%5Etfw">December 4, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



## Trilemma or Trilogy?
<img src="{{ site.baseurl }}/images/2022-01-01-trilemma-or-trilogy.png"
style="float: right; max-width: 50%; margin: 0 0 1em 2em;"> 


Limitation of knowledge-say I don't know if you really don't understand it instead of assuming and confidently lying (false positives)

## Reasoning

Connecting the dots thats reasoning instead of processing the whole information all at once and arriving to a conclusion.

- Rigorous proof:

    Set theory, logic and geometry.

### Inductive vs Deductive approach:

 

## Science v/s Math

'Mathematics is a language plus reasoning; it is like a language plus logic. Mathematics is tool for reasoning'. ~Richard Feynman

### Trusting the human v/s trusting the nature (math)

**_Don't listen to the 'expert'!_**
The press secretary ~Robin Hanson (Economist and Advisor at the Future of humanity institute of the oxford university)-humans lie to support their narrative by sacrificing the truth. (Malcolm Gladwell fallacy)

Mathematical proofs and truths over governance and law?!

different experts will have different interpretation.
Humans aren't absolute, Math is absolute.

Example:
<blockquote class="twitter-tweet" tw-align-center><p lang="en" dir="ltr">It’s a clear indication of find the pictures to suit your narrative is all that is. In the side view the bat has not reached the ball by the time the ball reaches the pad so there for its safe to say hitting the pad first as its directly in the same line did happen first. <a href="https://twitter.com/hashtag/simple?src=hash&amp;ref_src=twsrc%5Etfw">#simple</a></p>&mdash; Simon Doull (@Sdoull) <a href="https://twitter.com/Sdoull/status/1467007579263934468?ref_src=twsrc%5Etfw">December 4, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

--->

**`'If I bet on humanity, I'd never cash a ticket. It didn't pay to trust another human being. They never had it from the beginning, whatever it took'`.** {% cite
women1978 %}

<!---
## References:

[^1]: Banerjee, I., Bhimireddy, A. R., Burns, J. L., Celi, L. A., Chen, L.-C., Correa, R., Dullerud, N., Ghassemi, M., Huang, S.-C., Kuo, P.-C., Lungren, M. P., Palmer, L. J., Price, B. J., Purkayastha, S., Pyrros, A., Oakden-Rayner, L., Okechukwu, C., Seyyed-Kalantari, L., Trivedi, H., Wang, R., Zaiman, Z., Zhang, H., Gichoya, J. W. (2021). Reading Race: AI Recognises Patient’s Racial Identity In Medical Images. CoRR, abs/2107.10356. https://arxiv.org/abs/2107.10356

[^2]: Bukowski, C. (1978). In *Women*. HarperCollins.

--->

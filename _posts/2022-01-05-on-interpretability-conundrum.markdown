---
layout: post
title: "[Thoughts]-On the interpretability conundrum"  
author: krunal kshirsagar
largeimage: /images/2022-01-01-trilemma-or-trilogy.png
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


**`'If I bet on humanity, I'd never cash a ticket. It didn't pay to trust another human being. They never had it from the beginning, whatever it took'`.** {% cite
women1978 %}


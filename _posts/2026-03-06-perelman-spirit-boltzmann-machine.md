---
layout: post
title: "Perelman's spirit inside the Boltzmann machine"
author: krunal kshirsagar
math: true
published: true
---
<!--
tags:
- boltzmann-machines
- ricci-flow
- entropy
- information-geometry 
-->
<p>Lately I've been circling around an idea that I'm not sure I fully
understand yet. It started while I was reading about Boltzmann
Machines, which is one of those models that everyone learns about
historically but almost nobody really studies anymore. They show up in classical papers, people mention them when talking about
energy-based models, and then the conversation quickly moves on to
modern architectures. But I keep coming back to them because they feel conceptually deeper than the models we actually use. A Boltzmann machine isn't really an algorithm in the traditional sense. It's more like a physical system disguised as a neural network.</p>

You define an energy function over configurations of neurons, and the
probability of a configuration follows the Boltzmann distribution from
statistical mechanics.

$$
P(s) = \frac{1}{Z} e^{-E(s)/T}
$$

where $Z$ is the partition function.

Training means modifying the energy landscape so that configurations
corresponding to real data become more probable.

So learning becomes something like: shape the energy landscape, let
stochastic dynamics explore it, and gradually the distribution aligns
with the data.

<p>It's a very strange way to think about deep learning, but also kind
of elegant. The system is basically trying to find an equilibrium
distribution. And this thermodynamic flavor of the model kept reminding me of
something completely unrelated: the work of Grigori Perelman.{% cite
perelman_2002_entropy %}</p>

Perelman's work is about topology and geometry.{% cite
perelman_2003_ricci_surgery %} Specifically, his proof
of the **Poincaré Conjecture**, which sits firmly in the world of
geometric analysis. Neural networks and geometric topology seem like they belong to different category.

But the more I read about his work, the more I started noticing these weird conceptual dynamics at play.


## Geometry That Flows

The central object in Perelman's proof is **Ricci Flow**, introduced earlier by Richard Hamilton. Ricci flow describes how geometry evolves over time. If a manifold has a
metric $g_{ij}$, the flow evolves according to

$$
\partial_t g_{ij} = -2R_{ij}
$$

where $R_{ij}$ is the Ricci curvature tensor.

<p>Intuitively, curvature diffuses across the manifold. High curvature
regions flatten out, irregularities smooth over time, and the geometry
gradually becomes more regular.

People often compare it to heat diffusion, except instead of
temperature spreading across a surface, it's curvature spreading across
a manifold.
</p>
That idea alone is already pretty fascinating: geometry not as something
static but as something that flows.

But what Perelman did was introduce something even more interesting, an entropy-like quantity that governs this flow. Because once entropy enters the picture, the whole story starts sounding suspiciously like statistical physics.


## Entropy Everywhere

In Boltzmann machines, entropy is everywhere. The model implicitly optimizes a **free energy functional**

$$
F = E - TS
$$

<p>where energy pulls the system toward low-energy states while entropy encourages exploration.

Training the model effectively reshapes the energy function so that the
Boltzmann distribution approximates the empirical data distribution.

You can think of it as sculpting a probability landscape.
</p>

Perelman introduced an entropy functional often referred to as
**W-entropy** that behaves monotonically along Ricci flow{% cite
kleiner_lott_2008 %}.

<p>As the manifold evolves, this entropy changes in a controlled way,
constraining how the geometry can evolve.

So suddenly you have this geometric system whose dynamics are governed
by something that behaves like thermodynamic entropy.

And at that point the analogy becomes hard to ignore.</p>


## Landscapes

- In both systems you have a landscape.

- For Boltzmann machines it's an **energy landscape** over neural
configurations.

- For Ricci flow it's a **curvature landscape** over a manifold.

- Both landscapes evolve over time.

- Both evolutions are guided by entropy-like quantities.

- Both processes gradually transform a complicated structure into
something more general.

And I keep wondering if these similarities are accidental or if they're pointing toward something deeper.


## Learning as Geometry

There's a whole area called **information geometry** where probability
distributions form a manifold{% cite
amari_2016 %}. Under this view, training a probabilistic model is literally a geometric
process. The parameter space has curvature, the Fisher information defines a
metric, and optimization becomes a kind of geometric flow. Which starts sounding similar to Ricci flow again.

Except now the object evolving isn't a geometric manifold in the
traditional sense - it's a manifold of probability distributions.
So, learning is actually a kind of geometric evolution process,
just happening in probability space rather than physical space.


## Singularities

Perelman had to deal with singularities forming during Ricci flow -
places where curvature becomes extremely concentrated. In energy-based models, we see something vaguely analogous in the form
of phase transitions and sharp minima in energy landscapes.

Sometimes the probability mass collapses into narrow regions. Sometimes the dynamics become unstable near critical points.
Both systems seem to struggle with situations where structure becomes too concentrated.

So there's parallel in between these models.


## A Suspicion

None of this is meant to suggest that Perelman influenced Boltzmann
machines historically.

But conceptually, both seem to revolve around the same general idea:

> **Complex systems evolve across high‑dimensional landscapes under the guidance of entropy-like quantities.**

In geometry the landscape is curvature.

In deep learning the landscape is energy.

But structurally, they feel similar.

Optimization would become a special case of something more generic - the evolution of structures in high-dimensional spaces under entropy
constraints. And if that happens, it wouldn't surprise me if some of the mathematics that appeared in Perelman's work quietly shows up again. Not because Perelman was thinking about deep learning, but because both problems are ultimately asking the same kind of question:

**How do complicated spaces reorganize themselves into simpler ones?**

---


If you found this article useful in your research, blog, or discussion, please consider citing it.


```bibtex
@online{kshirsagar2026perelman,
  author  = {Krunal Kshirsagar},
  title   = {Perelman's spirit inside the Boltzmann machine},
  year    = {2026},
  url     = {https://ksheersaagr.github.io/blog/2026/03/perelman-spirit-boltzmann-machine/},
  note    = {Blog post}
}
```
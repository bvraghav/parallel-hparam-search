# Parallel Hyperparameter Tuning #

Parallel hyperparameter tuning using PyTorch
Multiprocessing Module `torch.multiprocessing` (or
equivalently Python standard library
`multiprocessing`).

The core idea is analogous to divide and conquer
paradigm, just even more banal, and is called scatter
and gather.  In context, it means to scatter the large
job into smaller, and typically independent, pieces
that can run in parallel; and gather them once they’re
all finished.

The key observation here is that once the data is
preprocessed, each training task may be run
independently; and the results are collated, once all
are finished.  Formally put, say $i$-th task of $N$
training tasks may be expressed as

$$\begin{align*}
\rho_i &\gets \mathrm{train}(i,\mathrm{data}) \\
\boldsymbol{\rho} &\equiv \{\rho_i\}
\end{align*}$$

In short, the steps are as follows:

1. Create a search space for hyperparameters;
2. Preprocess: load, sanitise, split, and resave data.
3. Fit the model, evaluate it, and save the pre-trained
   model. Do so for each set of hyperparameters;
4. Collate results and deduce the best.



## Implementation ##

We use [the good old GNU
Make](https://www.gnu.org/software/make/) for this
purpose.

Make is an order of processes defined in `Makefile`, so
that a dependency graph is inferred and independent
processes may be run in parallel.  Once the `Makefile`,
and thereby the dependency graph, is defined by a user,
invoking `make` with the necessary switch automatically
runs the independent processes in parallel.

The illustrated example is an essential
[MWE](https://en.wiktionary.org/wiki/MWE "Minimum
Working Example"), and also offers a quick refresher of
GNU Make.  The actual implementation is slightly more
involved, and in the spirit of
[DRY/DIE](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself
"Don’t Repeat Yourself! aka. Duplication is Evil").

### Targets and Recipes ###

Consider the following `Makefile`,

``` makefile
# Makefile
all : dist/.collated
dist/.collated : dist/.trained 
	python -m collate
    touch dist/.collated
dist/.trained : dist/trained/A/.trained dist/trained/B/.trained
	touch dist/.trained
dist/trained/A/.trained : dist/preprocessed
	python -m train
	touch dist/trained/A/.trained
dist/trained/B/.trained : dist/preprocessed
	python -m train
	touch dist/trained/B/.trained
# ...and so forth
```

It consists of a set of relationships of the form:

``` makefile
target : [ dependencies ]
       [ recipe ]
```

Each `target` is a filename (or sometimes not).
`recipe` is a set of shell commands that are
responsible to create the `target`.  `dependencies`
are prerequisites, such that only after ensuring that
they are up-to-date, the recipe for a target is
invoked.

In the illustrated Makefile, the first rule says,
target `all` is satisfied if `dist/.collated` is.

The second says, `dist/.collated` is satisfied if
`dist/.trained` is and the subsequent recipe runs
without error.  It’s recipe may be understood as
invoking python CLI module `collate` and thus updating
(or creating) the target file explicitly.

Similarly the third fourth and fifth targets define how
the target `dist/.trained` are defined.

### Invocation ###

A target may be invoked from command-line using

``` shell
make [OPTIONS] [TARGET] [VAR=VAL]
```

If unspecified the first target defined in the
`Makefile` is the default.

Commonly used `[OPTIONS]` include

+ `-n` to dry run;
+ `-B` to always make;
+ `-k` to keep going as far as possible (even after error);
+ `-f` to specify Makefile; 
+ `-C` to change directory;
+ `-j` for number of parallel jobs;
+ `-i` to ignore errors.

Further Reading: [GNU Make
Manual](https://www.gnu.org/software/make/manual/)


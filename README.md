# favannat (FAbricate and eVAluate Neural Networks of Arbitrary Topology)

Crate is functional but still in early development.

## Introduction

This crates aims to provide some semantics and data structures that allow to turn a somewhat generic description of a neural net into some executable function.

Therefore it provides the "network" termes like "node" and "edge" and a roughly sketched interface to execute nets;
namely the "Fabricator" trait and the "Evaluator" trait.

Further it provides one implementation of those traits.


## Limitations

Only DAGs (directed, acyclic graphs) can be evaluated, which is by design. It is planned to implement logic to unroll recurrent networks into DAGs.

## Contribution

Any thoughts on style and correctness/usefulness are very welcome.
Different implementations of the "Fabricate/Evaluate" traits are appreciated.

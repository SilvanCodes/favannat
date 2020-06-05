# favln

This crates aims to provide some semantics and data structures that allow to turn a somewhat generic description of a neural net into some executable function.

Therefore it provides the "network" termes like "node" and "edge" and a roughly sketched interface to execute nets;
namely the "Fabricator" trait and the "Evaluator" trait.

Further it provides one implementation of those traits.

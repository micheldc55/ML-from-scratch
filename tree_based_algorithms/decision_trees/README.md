# Decision Trees

This section attempts to implement Decision Trees using as low-level libraries as possible. You will see NumPy and maybe Pandas being used, but I'll try to keep it at that for most of the implementations. The idea is to get a grasp for how the models work and try different alternatives out. And probably learn a lot in the process.

## Where to read more

If you are very curious about this implementation and what to look into it, the best place to start is with this 1986 paper by J.R. Quinlan titled **"Induction of Decision Trees"** which can be found [here](https://link.springer.com/article/10.1007/BF00116251). It presents an algorithm called ID3 which can be implemented to build a deicison tree-like system.

## Loss Functions

The first thing needed is the cost or loss function. Literature on Loss Functions for decisions trees is extensive, but most implementations I've seen either use Gini's coefficient or Cross Entropy for selecting the best next feature, so we will implement those. If I come across some other interesting metric, I'll implement it in the loss_functions.py module.
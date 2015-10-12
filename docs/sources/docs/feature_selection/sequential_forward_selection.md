mlxtend  
Sebastian Raschka, last updated: 10/11/2015


<hr>

# Sequential Forward Selection

> from mlxtend.feature_selection import SFS

Sequential Forward Selection (SFS) is a classic feature selection algorithm -- a greedy search algorithm -- that has been developed as a suboptimal solution to the computationally often not feasible exhaustive search. In a nutshell, SFS adds one feature from the original feature set at the time, based on the classifier performance, until a feature subset of the desired size *k* is reached.


*Related topics:*

- [Sequential Backward Selection](./sequential_backward_selection.md)
- [Sequential Floating Forward Selection](./sequential_floating_forward_selection.md)
- [Sequential Floating Backward Selection](./sequential_floating_backward_selection.md)


---

### The SFS Algorithm



**Input:** $Y = \{y_1, y_2, ..., y_d\}$  

- The ***SFS*** algorithm takes the whole $d$-dimensional feature set as input.


**Output:** $X_k = \{x_j \; | \;j = 1, 2, ..., k; \; x_j \in Y\}$, where $k = (0, 1, 2, ..., d)$

- SFS returns a subset of features; the number of selected features $k$, where $k < d$, has to be specified *a priori*.

**Initialization:** $X_0 = \emptyset$, $k = 0$

- We initialize the algorithm with an empty set $\emptyset$ ("null set") so that $k = 0$ (where $k$ is the size of the subset).

**Step 1 (Inclusion):**  

  $x^+ = \text{ arg max } J(x_k + x), \text{ where }  x \in Y - X_k$  
  $X_k+1 = X_k + x^+$  
  $k = k + 1$    
*Go to Step 1*

- in this step, we add an additional feature, $x^+$, to our feature subset $X_k$.
- $x^+$ is the feature that maximizes our criterion function, that is, the feature that is associated with the best classifier performance if it is added to $X_k$.
- we repeat this procedure until the termination criterion is satisfied.


**Termination:** $k = p$

- We add features from the feature subset $X_k$ until the feature subset of size $k$ contains the number of desired features $p$ that we specified *a priori*.

---

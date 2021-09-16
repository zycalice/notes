# Artificial Intelligence
Follow textbook Artificial Intelligence: A_Modern_Approach 4th edition
## Searching
### Search Data Structure
Search algorithms require a data structure to keep track of the search tree. A node in the tree is represented by a data structure with four components:

* node.STATE: the state to which the node corresponds; 
* node.PARENT: the node in the tree that generated this node; 
* node.ACTION: the action that was applied to the parent’s state to generate this node; 
* node.PATH-COST: the total cost of the path from the initial state to this node. In mathematical formulas, we use g(node) as a synonym for PATH-COST.

We need a data structure to store the frontier. The appropriate choice is a queue of some kind, because the operations on a frontier are:

* IS-EMPTY(frontier) returns true only if there are no nodes in the frontier. 
* POP(frontier) removes the top node from the frontier and returns it.
* TOP(frontier) returns (but does not remove) the top node of the frontier. 
* ADD(node, frontier) inserts node into its proper place in the queue.


Three kinds of queues are used in search algorithms:
* A priority queue first pops the node with the minimum cost according to some evaluation function, f. It is used in best-first search.
* A FIFO queue or first-in-first-out queue first pops the node that was added to the queue first; we shall see it is used in breadth-first search.
* A LIFO queue or last-in-first-out queue (also known as a stack) pops first the most recently added node; we shall see it is used in depth-first search

The reached states can be stored as a lookup table (e.g. a hash table) where each key is a state and each value is the node for that state.

### Redundant paths
As the saying goes, algorithms that cannot remember the past are doomed to repeat it. There are three approaches to this issue.
First, we can remember all previously reached states (as best-first search does), allowing us to detect all redundant paths, and keep only the best path to each state. This is appropriate for state spaces where there are many redundant paths, and is the preferred choice when the table of reached states will fit in memory.

Second, we can not worry about repeating the past. There are some problem formulations where it is rare or impossible for two paths to reach the same state.
We call a search algorithm a graph search if it checks for redundant paths and a tree-like search if it does not check.
We say “tree-like search” because the state space is still the same graph no matter how we search it; we are just choosing to treat it as if it were a tree, with only one path from each node back to the root.

Third, we can compromise and check for cycles, but not for redundant paths in general. Since each node has a chain of parent pointers, we can check for cycles with no need for additional memory by following up the chain of parents to see if the state at the end of the path has appeared earlier in the path. Some implementations follow this chain all the way up, and thus eliminate all cycles; other implementations follow only a few links (e.g., to the parent, grandparent, and great-grandparent), and thus take only a constant amount of time, while eliminating all short cycles (and relying on other mechanisms to deal with long cycles).

### Performance measures:
* COMPLETENESS: Is the algorithm guaranteed to find a solution when there is one, and to correctly report failure when there is not?
  * Need to be systematic
* COST OPTIMALITY: Does it find a solution with the lowest path cost of all solutions?
* TIME COMPLEXITY: How long does it take to find a solution? This can be measured in seconds, or more abstractly by the number of states and actions considered.
* SPACE COMPLEXITY: How much memory is needed to perform the search?
  * In theoretical computer science, the typical measure is the size of the state-space graph, |V | + |E|, where |V | is the number of vertices (state nodes) of the graph and |E| is the number of edges (distinct state/action pairs). This is appropriate when the graph is an explicit data structure, such as the map of Romania.
  * But in many AI problems, the graph is represented only implicitly by the initial state, actions, and transition model. For an implicit state space, complexity can be measured in terms of
    * d: the depth or number of actions in an optimal solution
    * m: the maximum number of actions in any path
    * b: the branching factor or number of successors of a node that need to be considered.
    

### Uninformed search
#### Breadth-first search
When all actions have the same cost, an appropriate strategy is breadth-first search, in which the root node is expanded first, then all the successors of the root node are expanded next, then their successors, and so on. This is a systematic search strategy that is therefore complete even on infinite state spaces.

We could implement best first search where the evaluation function f(n) is the depth of the node—that is, the number of actions it takes to reach the node.

Additional efficiencies:
* A first-in-first-out queue will be faster than a priority queue, and will give us the correct order of nodes: new nodes (which are always deeper than their parents) go to the back of the queue, and old nodes, which are shallower than the new nodes, get expanded first.
* Reached can be a set of states rather than a mapping from states to nodes, because once we’ve reached a state, we can never find a better path to the state.
* That also means we can do an early goal test, checking whether a node is a solution as soon as it is generated, rather than the late goal test that best-first search uses, waiting until a node is popped off the queue.

#### Uniform Cost Search
When actions have different costs, an obvious choice is to use best-first search where the evaluation function is the cost of the path from the root to the current node. This is called Dijkstra’s algorithm by the theoretical computer science community, and uniform-cost search by the AI community. The idea is that while breadth-first search spreads out in waves of uniform depth—first depth 1, then depth 2, and so on—uniform-cost search spreads out in waves of uniform path-cost. The algorithm can be implemented as a call to BEST-FIRSTSEARCH with PATH-COST as the evaluation function

#### Depth-first Search
Depth-first search always expands the deepest node in the frontier first. It could be implemented as a call to BEST-FIRST-SEARCH where the evaluation function f is the negative of the depth. However, it is usually implemented not as a graph search but as a tree-like search that does not keep a table of reached states. The progress of the search is illustrated in Figure 3.11  ; search proceeds immediately to the deepest level of the search tree, where the nodes have no successors. The search then “backs up” to the next deepest node that still has unexpanded successors. Depth-first search is not cost-optimal; it returns the first solution it finds, even if it is not cheapest.


#### Depth-limited and iterative deepening search
To keep depth-first search from wandering down an infinite path, we can use depth-limited search, a version of depth-first search in which we supply a depth limit, ℓ, and treat all nodes at depth ℓ as if they had no successors
But if we studied the map carefully, we would discover that any city can be reached from any other city in at most 9 actions. This number, known as the diameter of the state-space graph, gives us a better depth limit, which leads to a more efficient depth-limited search. However, for most problems we will not know a good depth limit until we have solved the problem.

#### Iterative deepening search
Iterative deepening search solves the problem of picking a good value for ℓ by trying all values: first 0, then 1, then 2, and so on—until either a solution is found, or the depthlimited search returns the failure value rather than the cutoff value. The algorithm is shown in Figure 3.12  . Iterative deepening combines many of the benefits of depth-first and breadth-first search. Like depth-first search, its memory requirements are modest: O(bd) when there is a solution, or O(bm) on finite state spaces with no solution. Like breadth-first search, iterative deepening is optimal for problems where all actions have the same cost, and is complete on finite acyclic state spaces, or on any finite state space when we check nodes for cycles all the way up the path.


#### Bidirectional Search
Bidirectional search simultaneously searches forward from the initial state and backwards from the goal state(s), hoping that the two searches will meet.

### Informed search
Heuristic function: h(n) = estimated cost of the cheapest path from the state at node n to a goal state.

#### Greedy best-first search
Greedy best-first search is a form of best-first search that expands first the node with the lowest h(n) value—the node that appears to be closest to the goal—on the grounds that this is likely to lead to a solution quickly. So the evaluation function f(n) = h(n).

#### A* search
The most common informed search algorithm is A* search (pronounced “A-star search”), a best-first search that uses the evaluation function
f(n) = g(n) + h(n)
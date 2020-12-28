# Data Structures Essentials (zybook)

## Authors and contributors: 
Authors:
  - Roman Lysecky / Professor of Electrical and Computer Engineering / Univ. of Arizona
  - Frank Vahid / Professor of Computer Science and Engineering / Univ. of California, Riverside
  - Evan Olds / Content Developer / zyBooks\
  
Senior Contributors:
  - Tony Givargis / Professor of Computer Science / Univ. of California, Irvine
  - Susan Lysecky / Senior Content Developer / zyBooks\
  
Additional Contributors
  - Joe Hummel / University of Illinois at Chicago (Reviewer)
  
## Notes
### Basic data structures:
- Record: A record is the data structure that stores subitems, often called fields, with a name associated with each subitem.
- Array: An array is a data structure that stores an ordered list of items, where each item is directly accessible by a positional index.
- Linked list: A linked list is a data structure that stores an ordered list of items in nodes, where each node stores data and has a pointer to the next node.
- Binary tree: A binary tree is a data structure in which each node stores data and has up to two children, known as a left child and a right child.
- Hash table: A hash table is a data structure that stores unordered items by mapping (or hashing) each item to a location in an array.
- Heap: A max-heap is a tree that maintains the simple property that a node's key is greater than or equal to the node's childrens' keys. A min-heap is a tree that maintains the simple property that a node's key is less than or equal to the node's childrens' keys.
- Graph: A graph is a data structure for representing connections among items, and consists of vertices connected by edges. A vertex represents an item in a graph. An edge represents a connection between two vertices in a graph.

### Algorithms:
Algorithm efficiency is most commonly measured by the algorithm runtime, and an **efficient algorithm is one whose runtime increases no more than polynomially with respect to the input size**. However, some problems exist for which an efficient algorithm is unknown.
- NP-complete: problems are a set of problems for which no known efficient algorithm exists. NP-complete problems have the following characteristics:
  - No efficient algorithm has been found to solve an NP-complete problem.
  - No one has proven that an efficient algorithm to solve an NP-complete problem is impossible.
  - If an efficient algorithm exists for one NP-complete problem, then all NP-complete problem can be solved efficiently.
  
### Algorithms for data structures:
Data structures not only define how data is organized and stored, but also the operations performed on the data structure. While common operations include inserting, removing, and searching for data, the algorithms to implement those operations are typically specific to each data structure. Ex: Appending an item to a linked list requires a different algorithm than appending an item to an array.

### Abstract data types:
An abstract data type (ADT) is a data type described by predefined user operations, such as "insert data at rear," without indicating how each operation is implemented. An ADT can be implemented using different underlying data structures. However, a programmer need not have knowledge of the underlying implementation to use an ADT.\
Ex: A list is a common ADT for holding ordered data, having operations like append a data item, remove a data item, search whether a data item exists, and print the list. A list ADT is commonly implemented using arrays or linked list data structures.

- List: A list is an ADT for holding ordered data. (Array, linked list)
- Dynamic array: A dynamic array is an ADT for holding ordered data and allowing indexed access. (Array)
- Stack: A stack is an ADT in which items are only inserted on or removed from the top of a stack. (Linked list)
- Queue: A queue is an ADT in which items are inserted at the end of the queue and removed from the front of the queue. (Linked list)
- Deque: A deque (pronounced "deck" and short for double-ended queue) is an ADT in which items can be inserted and removed at both the front and back. (Linked list)
- Bag: A bag is an ADT for storing items in which the order does not matter and duplicate items are allowed. (Array, linked list)
- Set: A set is an ADT for a collection of distinct items. (Binary search tree, hash table)
- Priority queue: A priority queue is a queue where each item has a priority, and items with higher priority are closer to the front of the queue than items with lower priority. (Heap)
- Dictionary (Map): A dictionary is an ADT that associates (or maps) keys with values. (Hash table, binary search tree)

### Abstraction and optimization
Abstraction means to have a user interact with an item at a high-level, with lower-level internal details hidden from the user. ADTs support abstraction by hiding the underlying implementation details and providing a well-defined set of operations for using the ADT.

Using abstract data types enables programmers or algorithm designers to focus on higher-level operations and algorithms, thus improving programmer efficiency. However, knowledge of the underlying implementation is needed to analyze or improve the runtime efficiency.


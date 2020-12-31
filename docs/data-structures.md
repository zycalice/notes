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

### Runtime complexity, best case, and worst case
An algorithm's runtime complexity is a function, T(N), that represents the number of constant time operations performed by the algorithm on an input of size N. Runtime complexity is discussed in more detail elsewhere.

Because an algorithm's runtime may vary significantly based on the input data, a common approach is to identify best and worst case scenarios. An algorithm's best case is the scenario where the algorithm does the minimum possible number of operations. An algorithm's worst case is the scenario where the algorithm does the maximum possible number of operations.

### Space complexity

An algorithm's space complexity is a function, S(N), that represents the number of fixed-size memory units used by the algorithm for an input of size N. Ex: The space complexity of an algorithm that duplicates a list of numbers is S(N) = 2N + k, where k is a constant representing memory used for things like the loop counter and list pointers.

Space complexity includes the input data and additional memory allocated by the algorithm. An algorithm's auxiliary space complexity is the space complexity not including the input data. Ex: An algorithm to find the maximum number in a list will have a space complexity of S(N) = N + k, but an auxiliary space complexity of S(N) = k, where k is a constant.

### Linear Search and Binary Search
Linear search performe search one by one. Bineary search starts with the middle of a list, sorted from 0 to N. If the middle is 4.5, take 4. Puesdo code below:
```
BinarySearch(numbers, numbersSize, key) {
   mid = 0
   low = 0
   high = numbersSize - 1
   
   while (high >= low) {
      mid = (high + low) / 2
      if (numbers[mid] < key) {
         low = mid + 1
      }
      else if (numbers[mid] > key) {
         high = mid - 1
      }
      else {
         return mid
      }
   }
   
   return -1 // not found
}
```
### Binary search efficiency
Binary search is incredibly efficient in finding an element within a sorted list. During each iteration or step of the algorithm, binary search reduces the search space (i.e., the remaining elements to search within) by half. The search terminates when the element is found or the search space is empty (element not found). For a 32 element list, if the search key is not found, the search space is halved to have 16 elements, then 8, 4, 2, 1, and finally none, requiring only 6 steps. For an N element list, the maximum number of steps required to reduce the search space to an empty sublist is log_2(N)+1. (+1 is because you need to take the last step to reduce the last element to empty.) 

Example: reduce 32 elements to an empty list - max steps to min (only 5 steps):
* step 1: reduce from [0,31] to [0,14] by checking 15
* step 2: reduce from [0,14] to [0,6] by checking 7
* step 3: reduce from [0,6] to [0,2] by checking 3
* step 4: reduce from [0,2] to [0] by checking 1
* step 5: reduce from [0] to [] by checking 0

Example: reduce 32 elements to an empty list - max steps to max (6 steps):
* step 1: reduce from [0,31] to [16,31] by checking 15
* step 2: reduce from [16,31] to [22,31] by checking 23
* step 3: reduce from [22,31] to [28,31] by checking 27
* step 4: reduce from [28,31] to [30,31] by checking 29
* step 5: reduce from [30,31] to [31] by checking 30
* step 6: reduce from [31] to [] by checking 31

Because when it is 4.5 we take 4, it takes one step less to reduce to empty list when always searching the smaller sublist.

### Constant Time Algorithm
In practice, designing an efficient algorithm aims to lower the amount of time that an algorithm runs. However, a single algorithm can always execute more quickly on a faster processor. Therefore, the theoretical analysis of an algorithm describes runtime in terms of number of constant time operations, not nanoseconds. A constant time operation is an operation that, for a given processor, always operates in the same amount of time, regardless of input values.

### Identifying constant time operations
The programming language being used, as well as the hardware running the code, both affect what is and what is not a constant time operation. Ex: Most modern processors perform arithmetic operations on integers and floating point values at a fixed rate that is unaffected by operand values. Part of the reason for this is that the floating point and integer values have a fixed size. Below summarizes operations that are generally considered constant time operations.
* Addition, subtraction, multiplication, and division of fixed size integer or floating point values.	
* Assignment of a reference, pointer, or other fixed size data value.	
* Comparison of two fixed size data values.	
* Read or write an array element at a particular index.	




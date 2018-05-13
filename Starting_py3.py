
import numpy as np
import heapq


"""
 Data structures useful for implementing SearchAgents
"""

class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0

    def size(self):
        return len(self.list)

    def top(self):
        return self.list[len(self.list) - 1]

  

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

    def top(self):
        return self.list[len(self.list) - 1]
    
    
class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

#################################################

def display_board(state):
    print( "-------------")
    print( "| %i | %i | %i |" % (state[0], state[3], state[6]))
    print( "-------------")
    print( "| %i | %i | %i |" % (state[1], state[4], state[7]))
    print( "-------------")
    print( "| %i | %i | %i |" % (state[2], state[5], state[8]))
    print( "-------------")


def move_up(state):
    """Moves the blank tile up on the board. Returns a new state as a list."""
    # Perform an object copy
    new_state = state[:]
    index = new_state.index(0)
    # Sanity check
    if index not in [0, 3, 6]:
        # Swap the values.
        temp = new_state[index - 1]
        new_state[index - 1] = new_state[index]
        new_state[index] = temp
        return new_state
    else:
        # Can't move, return None (Pythons NULL)
        return None


def move_down(state):
    """Moves the blank tile down on the board. Returns a new state as a list."""
    # Perform object copy
    new_state = state[:]
    index = new_state.index(0)
    # Sanity check
    if index not in [2, 5, 8]:
        # Swap the values.
        temp = new_state[index + 1]
        new_state[index + 1] = new_state[index]
        new_state[index] = temp
        return new_state
    else:
        # Can't move, return None.
        return None


def move_left(state):
    """Moves the blank tile left on the board. Returns a new state as a list."""
    new_state = state[:]
    index = new_state.index(0)
    # Sanity check
    if index not in [0, 1, 2]:
        # Swap the values.
        temp = new_state[index - 3]
        new_state[index - 3] = new_state[index]
        new_state[index] = temp
        return new_state
    else:
        # Can't move it, return None
        return None


def move_right(state):
    """Moves the blank tile right on the board. Returns a new state as a list."""
    # Performs an object copy. Python passes by reference.
    new_state = state[:]
    index = new_state.index(0)
    # Sanity check
    if index not in [6, 7, 8]:
        # Swap the values.
        temp = new_state[index + 3]
        new_state[index + 3] = new_state[index]
        new_state[index] = temp
        return new_state
    else:
        # Can't move, return None
        return None


def create_node(state, parent, operator, depth, cost):
    return Node(state, parent, operator, depth, cost)


def expand_node(node):
    """Returns a list of expanded nodes"""
    expanded_nodes = []
    expanded_nodes.append(create_node(move_up(node.state), node, "u", node.depth + 1, 0))
    expanded_nodes.append(create_node(move_down(node.state), node, "d", node.depth + 1, 0))
    expanded_nodes.append(create_node(move_left(node.state), node, "l", node.depth + 1, 0))
    expanded_nodes.append(create_node(move_right(node.state), node, "r", node.depth + 1, 0))
    # Filter the list and remove the nodes that are impossible (move function returned None)
    expanded_nodes = [node for node in expanded_nodes if node.state != None]  # list comprehension!
    return expanded_nodes

################################################
########### SEARCH ALGORITHMS ##################
################################################

def dfs(start, goal, depth=10):
    """Performs a depth first search from the start state to the goal. Depth param is optional."""
    # NOTE: This is a limited search or else it keeps repeating moves. This is an infinite search space.
    # I'm not sure if I implemented this right, but I implemented an iterative depth search below
    # too that uses this function and it works fine. Using this function itself will repeat moves until
    # the depth_limit is reached. Iterative depth search solves this problem, though.
    #
    # An attempt of cutting down on repeat moves was made in the expand_node() function.
    
	#Declarations
    S = Stack()
    visited = [[]]
    node = {}
    
    start = create_node(start, None, None, 0, 0)
    
    node["parent"] = None
    node["action"] = None
    node["state"] = start
    
    
    S.push(node)           #enqueue
    flag = 0
    
    while(1):
        
        node = S.top()
        state = node["state"]
        
        
        if(state.state == goal):      #if current node is goal state
            break
        
        if (state.state in visited):		#if the current node was visitted before
			
            print("#####")
            S.pop()				#pop it from the stack and continue
            continue	
        
        for i in range(len(visited)):
            if(state.state == visited[i]):
                S.pop()
                flag = 1
                break
            
        if(flag):
            continue
            
            
        visited.append(state.state)
        
        
        children = expand_node(state)
        
        
        if(children):
            S.pop()					#enqueue
            
            for i in range(len(children)):
                
                if(children[i].state not in visited):
                    
        
                    sub_node = {}
                    sub_node["parent"] = node
                    sub_node["action"] = children[i].operator
                    sub_node["state"] = children[i]
                    S.push(sub_node)
        else:
            S.pop()
        
			
    path = []
    while(node["action"] != None):
        path.insert(0, node["action"])
        node = node["parent"]

    return path


def bfs(start, goal):
    """Performs a breadth first search from the start state to the goal"""
    # A list (can act as a queue) for the nodes.
    
    Q = Queue()
    visited = [[]]
    node = {}
    
    start = create_node(start, None, None, 0, 0)
    
    node["parent"] = None
    node["action"] = None
    node["state"] = start
    

    Q.push(node)           #enqueue
    
    while(1):
        
        node = Q.top()
        state = node["state"]

        if(state.state == goal):      #if current node is goal state
            break
        
        if (state.state in visited):		#if the current node was visitted before
			
            Q.pop()				#pop it from the stack and continue
            continue	
        
        visited.append(state.state)
        
        children = expand_node(state)
        
        
        if(children):
            Q.pop()					#enqueue
            
            for i in range(len(children)):
                
                if(children[i].state not in visited):
                    
        
                    sub_node = {}
                    sub_node["parent"] = node
                    sub_node["action"] = children[i].operator
                    sub_node["state"] = children[i]
                    Q.push(sub_node)
        else:
            Q.pop()
        
			
    path = []
    while(node["action"] != None):
        path.insert(0, node["action"])
        node = node["parent"]

    return path
        

def ucs(start, goal):
    """Performs a breadth first search from the start state to the goal"""
    # A list (can act as a queue) for the nodes.
    
    priority_q = PriorityQueue()
    visited = [[]]
    node = {}

    start = create_node(start, None, None, 0, 0)
	
    node["parent"] = None
    node["action"] = None
    node["goal"] = 0
    node["state"] = start

    priority_q.push(node, node["goal"])		#push root

    while(not priority_q.isEmpty()):	

        node = priority_q.pop()
        state = node["state"]
        
    
        if(state.state == goal):      #if current node is goal state
            break
        
        if (state.state in visited):		#if the current node was visitted before
			
            priority_q.pop()				#pop it from the stack and continue
            continue	
        
        visited.append(state.state)
        
        children = expand_node(state)
		
        if(children):
            for i in range(len(children)):

                if(children[i].state not in visited):
                    sub_node = {}
                    sub_node["parent"] = node
                    sub_node["action"] = children[i].operator
                    sub_node["state"] = children[i]
                    sub_node["goal"] = children[i].cost + node["goal"]
                    priority_q.push(sub_node, sub_node["goal"])
		

    path = []
    while(node["action"] != None):
        path.insert(0, node["action"])
        node = node["parent"]

    return path
        
    

def Heuristic(position, goal):

    "The Manhattan distance heuristic for a PositionSearchProblem"

    state1 = np.array(position.state)
    state2 = np.array(goal)
    
    diff = state1 - state2
     
    return np.count_nonzero(diff)


def greedy(start, goal):
    """Heuristic for the A* search. Returns an integer based on out of place tiles"""
  
    
    priority_q = PriorityQueue()
    visited = [[]]
    node = {}

    start = create_node(start, None, None, 0, 0)
	
    node["parent"] = None
    node["action"] = None
    node["heuristic"] =  Heuristic(start, goal)
    node["state"] = start

    priority_q.push(node, node["heuristic"])		#push root

    while(not priority_q.isEmpty()):	

        node = priority_q.pop()
        state = node["state"]
        
    
        if(state.state == goal):      #if current node is goal state
            break
        
        if (state.state in visited):		#if the current node was visitted before
			
            priority_q.pop()				#pop it from the stack and continue
            continue	
        
        visited.append(state.state)
        
        children = expand_node(state)
		
        if(children):
            for i in range(len(children)):

                if(children[i].state not in visited):
                    sub_node = {}
                    sub_node["parent"] = node
                    sub_node["action"] = children[i].operator
                    sub_node["state"] = children[i]
                    sub_node["heuristic"] = Heuristic(sub_node["state"], goal)
                    priority_q.push(sub_node, sub_node["heuristic"])
		

    path = []
    while(node["action"] != None):
        path.insert(0, node["action"])
        node = node["parent"]


    return path
    
    


def a_star(start, goal):
    """Perfoms an A* heuristic search"""
    # ATTEMPTED: does not work :(
    priority_q = PriorityQueue()
    visited = [[]]
    node = {}

    start = create_node(start, None, None, 0, 0)
	
    node["parent"] = None
    node["action"] = None
    node["goal"] = 0
    node["heuristic"] = Heuristic(start, goal)
    node["state"] = start

    priority_q.push(node, node["goal"] + node["heuristic"])		#push root

    while(not priority_q.isEmpty()):	

        node = priority_q.pop()
        state = node["state"]
        
    
        if(state.state == goal):      #if current node is goal state
            break
        
        if (state.state in visited):		#if the current node was visitted before
            print("####")
            priority_q.pop()				#pop it from the stack and continue
            continue	
        
        visited.append(state.state)
        
        children = expand_node(state)
		
        if(children):
            for i in range(len(children)):

                if(children[i].state not in visited):
                    sub_node = {}
                    sub_node["parent"] = node
                    sub_node["action"] = children[i].operator
                    sub_node["state"] = children[i]
                    sub_node["goal"] = children[i].cost + node["goal"]
                    sub_node["heuristic"] = Heuristic(sub_node["state"], goal)
                    priority_q.push(sub_node, sub_node["goal"] + sub_node["heuristic"])
		

    path = []
    while(node["action"] != None):
        path.insert(0, node["action"])
        node = node["parent"]


    return path




# Node data structure
class Node:
    def __init__(self, state, parent, operator, depth, cost):
        # Contains the state of the node
        self.state = state
        # Contains the node that generated this node
        self.parent = parent
        # Contains the operation that generated this node from the parent
        self.operator = operator
        # Contains the depth of this node (parent.depth +1)
        self.depth = depth
        # Contains the path cost of this node from depth 0. Not used for depth/breadth first.
        self.cost = cost


def readfile(filename):
    f = open(filename)
    data = f.read()
    # Get rid of the newlines
    data = data.strip("\n")
    # Break the string into a list using a space as a seperator.
    data = data.split(" ")
    state = []
    for element in data:
        state.append(int(element))
    return state


# Main method
def main():
    starting_state = readfile("state.txt")
    goal_state = [1, 8, 7, 2, 0, 6, 3, 4, 5]
    ### CHANGE THIS FUNCTION TO USE bfs, dfs, ids or a_star
    result = bfs(starting_state, goal_state)
    
 
    
    if result == None:
        print( "No solution found")
    elif result == [None]:
        print( "Start node was the goal!")
    else:
        print( result)
        print( len(result), " moves")


# A python-isim. Basically if the file is being run execute the main() function.
if __name__ == "__main__":
    main()
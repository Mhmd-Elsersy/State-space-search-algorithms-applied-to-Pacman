# State-space-search-algorithms-applied-to-Pacman 

![maze](https://user-images.githubusercontent.com/19307995/36748073-7278939a-1bff-11e8-9f8b-0db8928a0c1b.png)

## Description

In this project I implemented the state space search algorithms such as DFS, BFS, UCS, A* , and greedy search.
Pacman agent will find paths through his maze world, both to reach a particular location and to collect food efficiently.
Pacman lives in a shiny blue world of twisting corridors and tasty round treats.
Navigating this world efficiently will be Pacman's first step in mastering his domain.

The simplest agent in **searchAgents.py** is called the GoWestAgent,
which always goes West (a trivial reflex agent). This agent can occasionally win, but things get ugly for this agent when turning is required.
To see it in action, check the full list of commands in 
**commands.txt**.

## Table of contents

There are 5 main files:

+ A file where all of the search algorithms reside, **search.py**.

+ A file where all of the search-based agents reside, **searchAgents.py**.

+ A file that runs Pacman games. This file describes a Pacman GameState type, **Pacman.py**,

+ A file that contains logic behind how the Pacman world works.
This file describes several supporting types like AgentState, Agent, Direction, and Grid, **game.py**.

+ A file that contains useful data structures for implementing search algorithms, **util.py**.

## Installation & Dependencies

+ All the code is written in python 2.7, so you might consider creating a separate environment if you'r using Python 3 on your system.
To be able to create an environment head to [Anaconda](https://anaconda.org/).


nsga2.py is the original example for the Kurbenchmark function

nsga2jpq.py is the modified example to demonstrate the implementation
            for Unconstrained Multi-Objective Optimization Problems and 
            has been validated against the DEB by Deb (2001) function 
            and the POL by Poloni & all (2000) function.
            Modification have been made to the original code to implement
            the boundaries of the parameters
	     
constnsga2jpq is new and is an example to demonstrate the implementation
              for Constrained Multi-Objective Optimization Problems and 
              has been validated against the DEB CONSTR by Deb (2001)
              function, the SRN by Srinivas & Deb (1994) and the 
              OSY by Osyczka and Kundu (1995).
              New class has been defined and modification have been made
              to the original code to implement the constrained functions.


the modifications or new fuctions added are encapsulated in the code by

""" added by JPQ """"

  ....
  ....

# ---
 

The following code files have been modified:

pybrain/rl/environments/functions/multiobjective.py
pybrain/rl/environments/functions/transformations.py
pybrain/tools/nondominated.py
pybrain/optimization/optimizer.py
pybrain/optimization/populationbased/ga.py
pybrain/optimization/populationbased/multiobjective/__init__.py
pybrain/optimization/populationbased/multiobjective/nsga2.py
pybrain/optimization/populationbased/multiobjective/constnsga2.py

It is clear at the end that nsga2.py and constnsga2.py should be merged.
In the transformation file only the oppositeFunction has been modified, may 
be the other functions should also be modified but this was not required to 
make the code running.

Hope this will help to make you understanding what i have been doing.


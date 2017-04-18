Welcome to Pyiiout!
===================

Pyiiout is a demonstrative project of using Python and index notation for efficient and consistent implementation of numerical algorithms based on a finite-element model.

The primary intention is to demonstrate how to transform the mathematical formulation of the finite-element model directly into the numerical code by exploiting the indicial summation rule, that is supported by the NumPy package. In particular, we use the pull-out boundary value problem as an example. A mathematical description of the pull-out problem using index notation is provided in the included PDF file. The mathematical formulations in index notation can be easily and consistently transformed to Python code using the NumPy einsum method. See the PDF file for more details.

Second issue demonstrated by the package is the stacked representation of the system matrix consisting of three dimensional array with the first index denoting the element number and the second two indices representing the degrees of freedom of an element. Simple, conjugate gradient solver of an equation system has been implemented utilizing the efficient numpy array based operators. 

The array-based implementation is fully vectorized and is well suited for parallelization.

==========================================================================================================================================
Installation
==========================================================================================================================================
Dependency:

Python 2.7

Numpy 1.10.4 or higher

Matplotlib (for visualization of the result)

1.There are two ways to install the dependencies:

  a. install Python (https://www.python.org/downloads/) and then install the required packages.

  b. use third party python distribuition such as the Canopy (https://www.enthought.com/products/canopy/) or Anaconda (https://www.continuum.io/downloads), all the required packages come with those distribuition.

2. we recommand to use Canopy. To install Canopy, just download the right version of Canopy from (https://store.enthought.com/downloads/#default). 

3. To test the code, first download the code as a zip file from (https://github.com/simvisage/Pyiiout/archive/master.zip), then unzip the file, and run the pyiiout.py script in the pyiiout folder.

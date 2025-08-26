# AutoLyap.jl

`AutoLyap.jl` is a native Julia implementation of the *AutoLyap* methodology and the associated Python package  [AutoLyap](https://github.com/AutoLyap/AutoLyap), which were developed in the papers:

>
>
>1. AutoLyap: A Python package for computer-assisted Lyapunov analyses for first-order methods by [Manu Upadhyaya](https://manuupadhyaya.github.io/), [Adrien B. Taylor](https://adrientaylor.github.io/), [Sebastian Banert](https://github.com/sbanert), and [Pontus Giselsson](https://www.control.lth.se/personnel/personnel/pontus-giselsson/), 2025. 
>2. Automated tight Lyapunov analysis for first-order methods by [Manu Upadhyaya](https://manuupadhyaya.github.io/), [Adrien B. Taylor](https://adrientaylor.github.io/), [Sebastian Banert](https://github.com/sbanert), and [Pontus Giselsson](https://www.control.lth.se/personnel/personnel/pontus-giselsson/), 2025. 
>
>

The package is functionally equivalent to the Python package [AutoLyap](https://github.com/AutoLyap/AutoLyap), however the certain design patterns are different in the Julia package, e.g., the Julia package uses the notion of `struct+method` along with `multiple dispatch` in Julia over the notion of `class` in Python. 

The `runtestsl.jl` contains Julia test code for all analogous Python test code mentioned in the paper. 



### How to install the package

* Install Julia from [https://julialang.org/install/](https://julialang.org/install/)

* Download the repository as a zipped file into your computer, and unzip it into a folder of your choice. Say the path is: `C:\Users\YourUser\Documents\AutoLyap\`.

* Start the Julia REPL, from terminal, and in Julia first install the dependencies: 
  ```julia
  ] add Clarabel, Combinatorics, JuMP, LinearAlgebra, Mosek, MosekTools
  ```

  which will install all the dependencies. The package does not need `Mosek` (it uses clarabel by default) so no worries, if you get some licensing error associated with `Mosek`.

* Now type: 

  ```julia
  ] dev "C:\\Users\\YourUser\\Documents\\AutoLyap"
  ```

  which will track the package from that local path by createing a link to them.

  To test the package, run the following

  ```julia
  cd("C:\\Users\\YourUser\\Documents\\AutoLyap")
  ```

  ```julia
  activate .
  ```

  ```julia
  ] st
  ```

  ```
  ] test
  ```

  which will run all the test files (all the examples in the paper). 

The package supports the solvers `Clarabel` (open-source), `Mosek` (commercial, free for academic use), `SCS` (open-source), `COSMO` (open-source) as the solvers, however pretty much all the SDP solvers can be added. Based on our tests, we recommend the second-order SDP solvers  `Clarabel` and `Mosek` for reliable and high-precision solutions. 

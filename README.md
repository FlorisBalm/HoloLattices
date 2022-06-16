# HoloLattices
A generic code design for  to solve the problems typically arising in applications of holography to condensed matter systems. It consists of
a set of initialization blocks, which include importing the form of equations from Mathematica (or elsewhere), setting up the grid and choosing the initial solution
and the parameters of the problem; and the internal loop, where the solution is found after several iterations. For performance purposes it is important to optimize
the internal loop, while the initialization blocks only need to preserve the precision of data. See also Manual/package_description.pdf

# Workload
- Grids, Differentiation matrices and operators (including pseudospectral): Floris 
- Importing coefficients, observables, and  evaluating them: Josko
- Constructing the BVP operator (while taking care of all the boundary conditions), solving and updating: Aurelio


# License
    Copyright (C) 2018 Floris Balm

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

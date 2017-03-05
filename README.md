#TRANSITION RATES IN THE “ELECTRICAL” ISING MODEL

### by Daniel Sigg
#### adapted by Vincent Voelz

The “electrical” Ising model is entirely isomorphic to the standard Ising model, so any previous literature dealing with the standard model applies here also. I use a finite model with a 20 x 20 grid, yielding 400 cells.

## Energetics

The system energy for any of the $2^{400}$ configurations is:

$ E = \sum_{i,j} (e_i + e_j - 2e_i e_j )\delta\epsilon - \sum_i \delta q e_i V $,

where index $i$ ranges over all 400 cells and cell values are $e_i = 0$ (resting) or $1$ (activated). Index $j$ covers all nearest-neighbor cells (north, south, east, and west) to cell $i$.

The first sum (interaction term) runs over all nearest-neighbor pairs, with care being taken not to count the same interaction twice. I use periodic boundary conditions. The interaction term is designed so that adjacent cells oriented in the same direction have zero interaction. If they are oppositely aligned, there is an energy penalty of $\delta \epsilon$.

The second sum is the field term, driven by the voltage $V$. As cells flip from resting to activated they move a microscopic gating charge $\delta q$.  Thus the system charge $q = \sum_i \delta q e_i$ ranges from $0$ to $400 q$. I’ve chosen the total gating charge to be 1 eu, so $q = 0.0025$ eu.

The energy $\epsilon_{en}$ of an individual cell can take on 10 values, depending on its own state ($e = 0, 1$) and the number of activated nearest neighbors ($n = 0, 1, 2, 3, 4$). These can be conveniently calculated ahead of time.

The possible values are: $\epsilon_{en} = (2(2-n)\delta\epsilon - \delta q V) e + \delta\epsilon n$

We assume the whole system is in contact with a heat bath at temperature T. I prefer energy units of meV. The corresponding value of Boltzmann’s constant is 0.086174 meV/K, yielding kT = 25 meV at 290 K.

## Kinetics

There are 5 activating and 5 deactivating rate constants $a_i$ that apply to each cell $i$, consistent with the 10 local configurations. The rate constants are a function of the energy needed to activate the cell. This activation energy is equal to:

$\Delta\epsilon_n \equiv \epsilon_{1n} - \epsilon_{0n} - \delta q V = 2(2-n)\delta\epsilon - \delta q V$

Note for $n = 2$, there is no penalty for activation/deactivation provided $V = 0$. The formula describing the value of the 5 activating or "forward" rate constants $\alpha_n$, which applies to a cell originally in the resting state, and surrounded by $n = 0...4$ activated neighbors, is given by:

$\alpha_n = \nu \exp( -x\Delta\epsilon_n/kT)$

where $\nu$ is the pre-exponential factor, and $x$ (the Brønsted slope) is any number between 0 and 1 (usually assigned the value 0.5). I don’t ascribe any temperature-dependence to $\nu$ (implying the microscopic transition is a “bottleneck” or purely entropy-driven event) so that any temperature sensitivity of the macroscopic rate can be attributed to cell-cell interactions.

The corresponding 5 deactivating or “backward” rate constants $\beta_n$ are:

$\beta_n = \nu \exp( (1-x)\Delta\epsilon_n/kT)$

These formulas satisfy detailed balance, as evidenced by:

$\frac{\alpha_n}{\beta_n} = \exp(-\Delta\epsilon_n/kT)$.




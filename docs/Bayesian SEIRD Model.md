% Bayesian SEIRD Model
% Dan Sheldon and Casey Gibson
% April 27, 2020

\newcommand{\Unif}{\text{Unif}}
\renewcommand{\Gamma}{\text{Gamma}}
\newcommand{\Beta}{\text{Beta}}
\newcommand{\Normal}{\text{Normal}}
\newcommand{\N}{\mathcal{N}}
\newcommand{\Binomial}{\text{Binomial}}
\newcommand{\E}{\mathbb{E}}


# Underlying Disease Model 
We use an SEIR model augmented with two additional compartments: "D" for death, and "H" for "hospitalized-and-will-die". Note that H does not model all hospitalizations, only those that eventually lead to death.

$$
\begin{aligned}
\frac{dS}{dt} &= - \beta(t) \frac{SI}{N}  \\
\frac{dE}{dt} &= \beta(t) \cdot \frac{SI}{N} - \sigma E \\
\frac{dI}{dt} &= \sigma E - \gamma I\\
\frac{dR}{dt} &= (1-\rho)\gamma I  \\
\frac{dH}{dt} &= \rho \gamma I - \lambda H\\
\frac{dD}{dt} &= \lambda H 
\end{aligned}
$$

The parameters are:

* $\beta(t)$: (time-varying) contact rate
* $\sigma$: rate of transition from E to I
* $\gamma$: rate of transition from I to (R/H)
* $\rho$: fatality rate (i.e., probability of transitioning from I to H instead of I to R)
* $\lambda$: rate of transition from H to D (inverse of expected number of days in H compartment before death)

One additional variable $C(t)$ is added to track cumulative number of infections for the purposes of the observation model:
$$
\frac{dC}{dt} = \sigma E
$$
The nature of $\beta(t)$ will be discussed below --- it will change stochastically on a daily basis, but is constant for the duration of each day.


# Stochastic Model
## Parameters and Intial State Variables

$$
\begin{aligned}
I_0 &\sim \Unif(0, 0.02 N) \\
E_0 &\sim \Unif(0, 0.02 N) \\
H_0 &\sim \Unif(0, 0.02 N) \\
D_0 &\sim \Unif(0, 0.02 N) \\
\sigma &\sim \Gamma(5, 5 \hat{d}_E) \\
\gamma &\sim \Gamma(7, 7 \hat{d}_I) \\
\beta_0 &\sim \Gamma(1, \hat{d}_I/\hat{R}) \\
\lambda &\sim \Gamma(10, 100) \\
\rho &\sim \Beta(10, 90)\\
p &\sim \Beta(15, 35) \\
p_d &\sim \Beta(90, 10) 
\end{aligned}
$$

Justification and values for user-selected parameters: 

* $I_0$, $E_0$, $H_0$, and $D_0$ are the initial numbers of infectious, exposed, hospitalized-and-will-die, and dead. The priors are self-explanatory.
* $\sigma$ is the rate for leaving the exposed compartment; i.e., $1/\sigma$ is the expected duration in the exposed compartment. The prior satisfies $\E[\sigma] = 1/\hat{d}_E$, where $\hat{d}_E$ is an initial guess of the duration in the exposed compartment. Currently $\hat{d}_E = 4.0$ based on published estimates (shortened slightly to account for possible infectiousness prior to developing symptoms)
* $\gamma$ is the rate for leaving the infectious compartment; i.e., $1/\gamma$ is the expected duration in the infectious compartment. The prior satisfies $\E[\gamma] = 1/\hat{d}_I$, where $\hat{d}_I$ is an initial guess for the duration in the infectious compartment. The current setting is $\hat{d}_I = 2.0$ to model the likely isolation of individuals after symptom onset. 
* $\beta_0$ is the initial contact rate. In the SEIR model, it is known that $R_0 = \beta/\gamma$, so we set our prior to have mean $\E[\beta_0] = \hat{R}/\hat{d}_I$ where $\hat{R} = 3.0$ is an initial guess for $R_0$ and $\hat{d}_I = 2.0$, as described above.
* $\lambda$ is the inverse of the expected number of days in the H compartment; it satisfies $\E[\lambda] = 0.1$ with shape 10 (i.e. roughly 10 days in the H compartment)
* $\rho$ is the fatality rate: it satisfies $\E[\rho] = 0.1$ with concentration of $100$
* $p$ is the detection probability: it satisfies $\E[p] = 0.3$ with concentration 50
* $p_d$ is the detection probability for deaths: it satisfies $\E[p_d] = 0.9$ with concentration 100

Informative priors are useful on at least some of the internal parameters, since observed cases alone essentially place only one constraint on $(\sigma, \gamma, \beta)$--- we put more informative priors on $(\sigma, \gamma)$ and allow the model more freedom to estimate $\beta$. We have made some but not extensive efforts to keep priors on other parameters less informative. Certain parameters such as detection probability are poorly determined from observed data. Relaxing priors can lead to numerical issues during model estimation for states with fewer cases or noisier data.

## Process model
The process model proceeds in discrete time steps $k=1, 2, 3, \ldots$.

The state variables are initialized as:
$$
X(0) = \big(S(0), E(0), I(0), R(0), H(0), D(0), C(0)\big) = \big(N-E_0-I_0-H_0-D_0,E_0,I_0, 0, H_0, D_0, I_0\big)
$$
The updates are

$$
\begin{aligned}
\beta_{k+1} &= \beta_{k} \times \exp(\epsilon_k), \qquad \epsilon_k \sim \N(0, 0.2) \\
X(k+1) &= \mathtt{odesolve}\big(X(k),\, dX/dt,\, \beta_k\big)
\end{aligned}
$$
The contact rate $\beta_k$ undergoes an exponentiated Gaussian random walk starting from $\beta_0$ (defined above) with scale $\tau$. The function $\mathtt{odesolve}$ finds the state vector at time $k+1$ by simulating the ODE for one time step with (constant) contact rate $\beta_k$.

## Observation model

Observations are made on confirmed cases and deaths. Let $y(t)$ be the cumulative number of confirmed cases at time $t$ and $z(t)$ be the cumulative number of reported deaths. The observation model is:
$$
\begin{aligned}
y(t) &\sim \N(\mu_y, 0.15 \mu_y), \quad \mu_y = p C(t)  \\
z(t) &\sim \N(\mu_y, 0.15 \mu_y), \quad \mu_z = p_d D(t)
\end{aligned}
$$
Our models weight the log-probabilities of the two different types of observations differently to encourage them to focus slightly more on deaths than confirmed cases. Specifically $\log p(y(t) | \cdots)$ is scaled by 0.5 and $\log p(z(t) | \cdot)$ is scaled by 2.0.

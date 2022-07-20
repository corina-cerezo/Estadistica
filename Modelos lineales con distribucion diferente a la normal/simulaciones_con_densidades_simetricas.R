
# Simulaciones solo tomando el caso de una dimensión


# Errores con distribución Laplace (Exponencial potencia con alpha = 1/2) ------------------------------------
library(ExtDist) 

# Parámetros de los errores ----------------------------------------------------
mu <- 0 # mu debe ser cero por supuesto del modelo
sigma2 <- 17
b <- 2*sqrt(sigma2)

# Función g y contante de integración
f_g <- function(u){
  exp(-abs(u)/2)
}
ci <- 4 # integrate exp(-abs(x)/2) from 0 to inf 

# generamos los errores con esta distribución ----------------------------------
n = 300
p = 6
set.seed(2022)
e <- rLaplace(n = n,mu = mu,b = b)
hist(e)
qqnorm(e)

# generamos el vector de Y -----------------------------------------------------

x1 <- rnorm(n,mean = 0,sd = 3)
x2 <- rnorm(n,mean = 10,sd = 2)^2
x3 <- sin(rnorm(n,mean = -30,sd = 5))
x4 <- abs(rnorm(n,mean = 100,sd = 10))
x5 <- acos(runif(n,min = -1,max = 1))*180/(2*pi)
y = 15 + 5*x1 + 8*x2 -7*x5 + e
modelo <- lm(y ~ x1 + x2 + x3 + x4 + x5)
modelo
summary(modelo)
sum(modelo$residuals^2)
modelo2 <- lm(y  ~ x1 + x2 + x4 + x5)
modelo2
summary(modelo2)
modelo3 <- lm(y  ~ x1 + x2 + x5)
modelo3
summary(modelo3)

# Matriz X ---------------------------------------------------------------------
X <- matrix(c(rep(1,300),x1,x2,x3,x4,x5),ncol = 6, byrow = FALSE)
head(X)

#Estimaciones ------------------------------------------------------------------
#Para los coeficientes, pues es el mismo que por mínimos cuadrados
beta_gorro <- solve(t(X)%*%X) %*% t(X) %*% y
beta_gorro

# Para sigma2

f_l <- function(z){
  z^{1/2}*f_g(z) # aquí no se pudo porque no tenemos g
}
curve(f_l,from = -3,10)
z_h <- optimize(f_l, c(0,3), tol = 0.0001,maximum = TRUE)$maximum

f_phi <- expression((1+4*x)^(-1)) #creo que esta sería phi
f_phi_1 = D(f_phi, 'x')
x <- 0
(c <- -2*eval(f_phi_1)) # wolfram: derivate (1+4x)^(-1)

sigma2_gorro_E <- 1/(z_h) * t(y- X %*% beta_gorro) %*% (y- X %*% beta_gorro)
sigma2_gorro_E

sigma2_gorro_EU <- z_h/((c)*(n-p))*sigma2_gorro_E
sigma2_gorro_EU #OMGGGGGG

# suma de cuadrados del error
t(y- X %*% beta_gorro) %*% (y- X %*% beta_gorro)

# Pero el LRT para cada beta es ------------------------------------------------
# Beta0
A=matrix(c(1,0,0,0,0,0),ncol=6)
c = 0
H = X %*% solve(t(X)%*%X) %*% t(X)
numerador <- t(y) %*% (diag(n) - H) %*% y
denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
LRT <- (numerador/denominador)^(n/2)
LRT

# Beta1
A=matrix(c(0,1,0,0,0,0),ncol=6)
c = 0
H = X %*% solve(t(X)%*%X) %*% t(X)
numerador <- t(y) %*% (diag(n) - H) %*% y
denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
LRT <- (numerador/denominador)^(n/2)
LRT

# Beta2
A=matrix(c(0,0,1,0,0,0),ncol=6)
c = 0
H = X %*% solve(t(X)%*%X) %*% t(X)
numerador <- t(y) %*% (diag(n) - H) %*% y
denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
LRT <- (numerador/denominador)^(n/2)
LRT

# Beta3
A=matrix(c(0,0,0,1,0,0),ncol=6)
c = 0
H = X %*% solve(t(X)%*%X) %*% t(X)
numerador <- t(y) %*% (diag(n) - H) %*% y
denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
LRT <- (numerador/denominador)^(n/2)
LRT

# Beta4
A=matrix(c(0,0,0,0,1,0),ncol=6)
c = 0
H = X %*% solve(t(X)%*%X) %*% t(X)
numerador <- t(y) %*% (diag(n) - H) %*% y
denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
LRT <- (numerador/denominador)^(n/2)
LRT

# Beta5
A=matrix(c(0,0,0,0,0,1),ncol=6)
c = 0
H = X %*% solve(t(X)%*%X) %*% t(X)
numerador <- t(y) %*% (diag(n) - H) %*% y
denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
LRT <- (numerador/denominador)^(n/2)
LRT

################################################################################
################################################################################

# Errores con distribución t-Student ------ ------------------------------------
library(mvtnorm) 
comb = function(n, x) {
  factorial(n) / factorial(n-x) / factorial(x)
}

# Parámetros de los errores ----------------------------------------------------
mu <- 0 # mu debe ser cero por supuesto del modelo
sigma2 <- 7
q <- 1 # porque los errores son vectores de dimensión q
gl <- 3 # más de 2 según la librería

# Función g y constante de integración
f_g <- function(u){
  (1+u/gl)^(-(gl+q)/2)
}
ci <- gamma((gl+q)/2) /((pi*gl)^(q/2)*gamma(gl/2)) # integrate (1+x^2/gl)^(-(gl+1)/2) from 0 to inf 
ci
# 1/(2*(sqrt(3)*pi/4))
# aquí la constante me queda igual con la definición de la t que con la formula de la página 60 de mi tesis
# sí me integra 1: integrate ci(sqrt(sigma2))*g(x^2) dx from -inf to inf

# generamos los errores con esta distribución ----------------------------------
n = 300
p = 6
set.seed(2022)
e <- rmvt(n = n,sigma = sigma2*diag(q), df = gl)
hist(e)
qqnorm(e)

# generamos el vector de Y -----------------------------------------------------

x1 <- rnorm(n,mean = 0,sd = 3)
x2 <- rnorm(n,mean = 10,sd = 2)^2
x3 <- sin(rnorm(n,mean = -30,sd = 5))
x4 <- abs(rnorm(n,mean = 100,sd = 10))
x5 <- acos(runif(n,min = -1,max = 1))*180/(2*pi)
y = 15 + 5*x1 + 8*x2 -7*x5 + e
modelo <- lm(y ~ x1 + x2 + x3 + x4 + x5)
modelo
summary(modelo)

# Matriz X ---------------------------------------------------------------------
X <- matrix(c(rep(1,300),x1,x2,x3,x4,x5),ncol = p, byrow = FALSE)
head(X)

#Estimaciones ------------------------------------------------------------------
#Para los coeficientes, pues es el mismo que por mínimos cuadrados
beta_gorro <- solve(t(X)%*%X) %*% t(X) %*% y
beta_gorro

# Para sigma2
f_l <- function(z){
  z^{1/2}*f_g(z) 
}
curve(f_l,from = -3,10)
z_h <- optimize(f_l, c(0,3), tol = 0.0001,maximum = TRUE)$maximum
z_h #según Wolfram es juto 1: f(x)=x^(1/2)*(1+x/3)^(-(3+1)/2)

f_phi <- expression(sqrt(pi)*gamma((gl+1)/2)*exp(-sqrt(gl)*sqrt(x))/(2^(gl-1)*gamma(gl/2))*(2*(2*sqrt(gl)*sqrt(x))^(1-1)/factorial(1-1) + 1*(2*sqrt(gl)*sqrt(x))^(2-1)/factorial(2-1))) #creo que esta sería phi
#                   sqrt(pi)*gamma((gl+1)/2)*exp(-sqrt(gl)*sqrt(x))/(2^(gl-1)*gamma(gl/2))*(comb(2*2-1-1,2-1)*(2*sqrt(gl)*sqrt(x))^(1-1)/factorial(1-1) + comb(2*2-2-1,2-2)*(2*sqrt(gl)*sqrt(x))^(2-1)/factorial(2-1))

f_phi_1 = D(f_phi, 'x')
x <- 0
(c <- -2*eval(f_phi_1)) # en wolfram da 3

sigma2_gorro_E <- 1/(z_h) * t(y- X %*% beta_gorro) %*% (y- X %*% beta_gorro)
sigma2_gorro_E

sigma2_gorro_EU <- z_h/((3)*(n-p))*sigma2_gorro_E
sigma2_gorro_EU 

sigma2_gorro_E == sigma2_gorro_EU








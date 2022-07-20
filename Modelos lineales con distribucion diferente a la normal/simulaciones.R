library(nimble)

# suponiendo que los errores siguen distribución  Laplace
# entonces los parámetros son
mu <- 0
b <- 5
# y la función phi es
f_phi <- expression((1+b^2*x)^-1) # phi va en función de t^2
f_phi_1 = D(f_phi, 'x') 
x <- 0
var_e <- -2*eval(f_phi_1) # varianza real de e
# y la función generadora de densidad es
f_g <- function(u){
  -2/b * exp(-abs(u)/b)
}

# install.packages("PearsonDS")
library(PearsonDS)
# Parámetros





# Modelo con errores Laplace
x1 <- rnorm(300,mean = 50,sd = 3)
x2 <- rnorm(300,mean = 10,sd = 2)
x3 <- rnorm(300,mean = -30,sd = 5)
e <- rdexp(n = 300,location = mu, scale = b)
hist(e)
qqnorm(e)
y = 5*x1 + 8*x2^2 + e
x22 <- x2^2
modelo <- lm(y ~ x1 + x22 + x3)
modelo
summary(modelo)

# Matriz X
X <- matrix(c(rep(1,300),x1,x2^2,x3),ncol = 4, byrow = FALSE)
X
beta_gorro <- solve(t(X)%*%X) %*% t(X) %*% y
beta_gorro



####################################################################

# Errores con distribución logística
# Parámetros
mu <- 0 # localización
s <- 17 # de escala, en esta distribución es sigma2
# Definimos la función phi
f_phi <- expression(pi*s*sqrt(x)/sinh(pi*s*sqrt(x)))
f_phi_1 = D(f_phi, 'x') 
x <- 0
(var_e <- -2*eval(f_phi_1)) # esta parte no funcionó porque en R
                            # se indetermina, pero en wolfram alpha sí
                            # sale con el sig comando
                            # eval derivate -2*pi*7*sqrt(x)/sinh(pi*7*sqrt(x)) at x=0
# Definimos la función generadora de densidad g
c <- 0.673718 #integrate 2*exp(-x^2)/(1+exp(-x^2))^2 dx from 0 to inf
f_g <- function(u){
  exp(-u^2/7)/(c*sqrt(7)*(1+exp(-u^2/7))^2) #aquí me está causando duda la constante
}# integrate exp(-x^2/7)/(1+exp(-x^2/7))^2/0.673718/sqrt(7) from -inf to inf
curve(f_g,from = -5,to = 5)
# generamos los errores con esta distribución
n = 300
set.seed(201)
e <- rlogis(n = n,location = mu,scale = s)
hist(e)
qqnorm(e)


##############################################################################
# Errores con distribución tipo Kotz --------------------------------------
########################################################################

# Parámetros ----------------------------------------------------------
p <- 1 # es la dimensión del vector aleatorio, pero como es v.a. entonces es 1
r <- 1/3
s <- 1 # La función phi dada a continuación es cuando s=1 (Nadarajah 2003)
N <- 2
mu <- 0
sigma2 <- 7

# Definimos la función phi --------------------------------------------
# exp(-x/(4*r))*(combn(1,0)*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(0)
#                + combn(1,1)*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(1))
f_phi <- expression(exp(-x/(4*r))*(1*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(0)
                                   + 1*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(1)))
f_phi_1 = D(f_phi, 'x')
x <- 0
(var_e <- -2*eval(f_phi_1)) # esta parte no funcionó porque en R
                            # se indetermina, pero en wolfram alpha sí
                            # sale con el sig comando
                            # eval derivate exp(-x/(4*(1/3)))*(1*(gamma(1/2))/(gamma(1/2)+0)*(-x/(4*(1/3)))^(0) + 1*(gamma(1/2))/(gamma(1/2)+0)*(-x/(4*(1/3)))^(1)) at x=0
                            # y da -3/2
                            # entonces la varianza de los errores es -2*(-3/2)*7

# Definimos la función g ----------------------------------------------
f_g <- function(u){
  u^(N-1)*exp(-r*u^(s))
}


# Se define la constante de integración -------------------------------
c_0 <- s*gamma(p/2)/(pi^(p/2)*gamma((2*N+p-2)/(2*s)))*r^((2*N+p-2)/(2*s)) #Segun la def de la dist
c_1 <- (2*3*sqrt(3*pi)/4)^(-1) # Según la constante de Gupta página 23
                                 # integrate r^(1-1)g(r^2) dr from 0 to inf
                                 # integrate x^(2*(2-1))*exp(-(1/3)*x^(2)) from 0 to inf
# OMG aquí las constantes sí dan igual

# Entonces la función de densidad es ----------------------------------
f <- function(x){
  c_1*(sigma2)^(-1/2)*((x-mu)^2/sigma2)^(N-1)*exp(-r*((x-mu)^2/sigma2)^(s)) #aquí me está causando duda la constante
}# integrate 0.2171567*(7)^(-1/2)*((x-5)^(2)/7)*(2-1)*exp(-(1/3)*((x-5)^2/7)^(1)) from -inf to inf
# aww es hermosa porque sí integra 1


library(fMultivar)
## delliptical2d -
# Kotz' Elliptical Density:
x <- (-40:40)/10
X <- grid2d(x)
z <- delliptical2d(X$x, X$y, rho = 0.5, type = "kotz")
Z <- list(x = x, y = x, z = matrix(z, ncol = length(x)))
## Perspective Plot:
persp(Z, theta = -40, phi = 30, col = "steelblue")
## Image Plot with Contours:
image(Z, main = "Bivariate Kotz")
contour(Z, add=TRUE)









# generamos los errores con esta distribución --------------------------
n = 300
set.seed(201)
e <- rlogis(n = n,location = mu,scale = s)
hist(e)
qqnorm(e)






# generamos el vector de Y

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
modelo2 <- lm(y ~ -1 + x1 + x2 + x3 )
modelo2
summary(modelo2)


# Matriz X
X <- matrix(c(rep(1,300),x1,x2,x3,x4,x5),ncol = 6, byrow = FALSE)
head(X)

#Estimaciones
#Para los coeficientes, pues es el mismo que por mínimos cuadrados
beta_gorro <- solve(t(X)%*%X) %*% t(X) %*% y
beta_gorro

# Para sigma2
f_l <- function(z){
  z^{1/2}*f_g(z)
}
z_h <- optimize(f_l, c(0,3), tol = 0.0001,maximum = TRUE)$maximum
sigma2_gorro <- 1/(z_h) * t(y- X %*% beta_gorro) %*% (y- X %*% beta_gorro)
sigma2_gorro

# suma de cuadrados del error
t(y- X %*% beta_gorro) %*% (y- X %*% beta_gorro)

# Pero el LRT para cada beta es
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

g_num <- t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
g_denom <- t(y) %*% (diag(n) - H) %*% y
g_num / g_denom

# install.packages("gwer")
library(gwer)

elliptical.fit <- elliptical(y ~ x1+x2+x3+x4+x5, family = LogisI())
t(y- X %*% elliptical.fit$coefficients) %*% (y- X %*% elliptical.fit$coefficients)
sum(elliptical.fit$residuals^2)



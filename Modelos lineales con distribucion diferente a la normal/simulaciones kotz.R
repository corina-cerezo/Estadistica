################################################################################
########################## Simulaciones para la tesis ##########################
################################################################################


# Errores con distribución tipo Kotz -------------------------------------------
library(ExtDist) 
require(ggplot2)
library(latex2exp) # que para el Latex

# Parametros definidos
p <- 1 # es la dimensión del vector aleatorio, pero como es v.a. entonces es 1
r <- 1/3
s <- 1 # La función phi dada a continuación es cuando s=1 (Nadarajah 2003)
N <- 2
mu <- 0
sigma2 <- 1/7

# Valores necesarios para la estimación
# Función g y contante de integración
f_g <- function(u){
  u^(N-1)*exp(-r*u^(s))
}
ci <- s*gamma(p/2)/(pi^(p/2)*gamma((2*N+p-2)/(2*s)))*r^((2*N+p-2)/(2*s)) #Segun la def de la dist

# Función de densidad
f_f <- function(x){
  ci*(sigma2)^(-1/2)*f_g(x^2/sigma2)
}
# Sí integra 1
# integrate 2*x^2/(3*7sqrt(3*pi*7))*exp(-x^2/(3*7)) dx from -inf to inf

# Gráfica de la función de densidad
ggplot(data.frame(x = c(-4, 4)), aes(x = x)) +
  stat_function(fun = dnorm,  aes(colour = "Normal estándar")) +
  stat_function(fun = f_f,  aes(colour = "Tipo Kotz")) +
  ggtitle("Función de densidad") +
  scale_colour_manual(" ", values = c("deeppink", "dodgerblue3")) +
  scale_x_continuous(name = "x") +
  ylab(TeX(r'($f(x)$)')) +
  theme(legend.position="bottom",plot.title = element_text(hjust = 0.5))

# Función l(z)
f_l <- function(z){
  z^{1/2}*f_g(z) 
}
curve(f_l,from = -3,10)
(z_h <- optimize(f_l, c(0,5), tol = 0.000001,maximum = TRUE)$maximum)

# Definimos la función phi --------------------------------------------
# exp(-x/(4*r))*(combn(1,0)*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(0)
#                + combn(1,1)*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(1))
f_phi <- expression(exp(-x/(4*r))*(1*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(0)
                                   + 1*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(1)))
f_phi_1 = D(f_phi, 'x')
x <- 0
(c <- -2*eval(f_phi_1)) # esta parte no funcionó porque en R
# se indetermina, pero en wolfram alpha sí
# sale con el sig comando
# eval derivate exp(-x/(4*(1/3)))*(1*(gamma(1/2))/(gamma(1/2)+0)*(-x/(4*(1/3)))^(0) + 1*(gamma(1/2))/(gamma(1/2)+0)*(-x/(4*(1/3)))^(1)) at x=0
# y da -3/2
# entonces la varianza de los errores es -2*(-3/2)*7































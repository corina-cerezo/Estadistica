################################################################################
########################## Simulaciones para la tesis ##########################
################################################################################


# Errores con distribución t-Student -------------------------------------------
library(PearsonDS) 
require(ggplot2)
library(latex2exp) # que para el Latex


# Parametros definidos
m <- 3 # grados de libertad
mu <- 0
sigma2 <- 19

# Valores necesarios para la estimación
# Función g y contante de integración
f_g <- function(u){
  (1+u/m)^(-(m+1)/2)
}
ci <- gamma((m+1)/2)/((pi*m)^0.5*gamma(m/2))

# Función de densidad
f_f <- function(x){
  ci*(sigma2)^(-1/2)*f_g(x^2/sigma2)
}
# Sí integra 1
# integrate 2*x^2/(3*7sqrt(3*pi*7))*exp(-x^2/(3*7)) dx from -inf to inf

# Gráfica de la función de densidad
ggplot(data.frame(x = c(-4, 4)), aes(x = x)) +
  stat_function(fun = dnorm,  aes(colour = "Normal estándar")) +
  stat_function(fun = f_f,  aes(colour = "t-Student")) +
  ggtitle("Función de densidad") +
  scale_colour_manual(" ", values = c("deeppink", "dodgerblue3")) +
  scale_x_continuous(name = "x") +
  ylab(TeX(r'($f(x)$)')) +
  theme(legend.position="bottom",plot.title = element_text(hjust = 0.5))

# Función l(z) ---------------------------------------------------------
f_l <- function(z){
  z^{1/2}*f_g(z) 
}
curve(f_l,from = -3,10)
(z_h <- optimize(f_l, c(0,5), tol = 0.000001,maximum = TRUE)$maximum)

# Definimos la función phi --------------------------------------------
# exp(-x/(4*r))*(combn(1,0)*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(0)
#                + combn(1,1)*(gamma(p/2))/(gamma(p/2)+0)*(-x/(4*r))^(1))
f_phi <- expression((1+sqrt(m*x))/(exp(sqrt(m*x))))
f_phi_1 = D(f_phi, 'x')
x <- 0
(c <- -2*eval(f_phi_1)) # esta parte no funcionó porque en R
# se indetermina, pero en wolfram alpha sí
# sale con el sig comando
# evaluate derivate (1+sqrt(3x))/(exp(sqrt(3x))) at x=0
# y da -3/2
# entcones c = -2*(-3/2)
# entonces la varianza de los errores es -2*(-3/2)*7

# Luego, vamos a modelar a y como: y = beta0 + beta1*x1 + error
# pero pondremos una variable de "confusión" en la regresión
# entonces
p <- 3 #número de betas
# las estadísticas se reportan serán:
# a) Media y desviación estándar de la estimación de sigma^2_gorro
# con el estimador elíptico vs asumiendo normalidad
# b) % en que se rechaza cada una de las betas
# Con la prueba de hipótesis elíptica vs asumiendo normalidad

# Función que repite m veces las estimaciones
simula_tStudent <- function(betas, df, location, scale, semilla, m, n){ # m es el número de veces que se repite el experimento
  aux1 <- c()                      # n es el número de observaciones por experimento
  aux2 <- c()
  aux3 <- c()
  aux4 <- c()
  aux5 <- c()
  aux6 <- c()
  aux7 <- c()
  aux8 <- c()
  aux9 <- c()
  aux10 <- c()
  aux11 <- c()
  aux12 <- c()
  aux13 <- c()
  aux14 <- c()
  p <- length(betas)+1
  set.seed(semilla)
  x1 <- abs(rnorm(n,mean = 100,sd = 10))
  x2 <- acos(runif(n,min = -1,max = 1))*180/(2*pi)
  for(k in 1:m){ # i=2; j=1; l=1; n=50
    # Se fija la semilla para que se pueda replicar el experimento
    set.seed(semilla*k)
    # Se simulan los datos aleatoriamente
    e <- rpearsonVII(n = n, df = df, location = location, scale = scale)
    # Se define el modelo
    y <- betas[1] + betas[2]*x1 + e
    # Se define la matriz X
    X <- matrix(c(rep(1,n),x1,x2),ncol = p, byrow = FALSE)
    # Estimadores
    beta_gorro <- solve(t(X)%*%X) %*% t(X) %*% y
    sigma2_gorro_E <- 1/(z_h) * t(y- X %*% beta_gorro) %*% (y- X %*% beta_gorro)
    sigma2_gorro_EU <- z_h/((3)*(n-p))*sigma2_gorro_E
    
    # definimos los valores que se van a regresar
    aux1[k] <- sigma2_gorro_EU
    aux2[k] <- 1/(n-p) * t(y- X %*% beta_gorro) %*% (y- X %*% beta_gorro)
    # Pruebas de hipótesis
    c = 0
    H = X %*% solve(t(X)%*%X) %*% t(X)
    numerador <- t(y) %*% (diag(n) - H) %*% y
    # Beta0
    A=matrix(c(1,0,0),ncol=p)
    denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
    LRT <- (numerador/denominador)^(n/2)
    F_T <- (n-p)*(LRT^(-2/n)-1) > qf(0.05,1,n-p) # si es mayor, se rechaza H0
    aux3[k] <- ifelse(F_T, 1, 0) # cuenta las veces que se rechaza H0, decisión buena
    aux9[k] <- LRT
    
    # Beta1
    A=matrix(c(0,1,0),ncol=p)
    denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
    LRT <- (numerador/denominador)^(n/2)
    F_T <- (n-p)*(LRT^(-2/n)-1) > qf(0.05,1,n-p) # si es mayor, se rechaza H0
    aux4[k] <- ifelse(F_T, 1, 0) # cuenta las veces que se rechaza H0, decisión buena
    aux10[k] <- LRT
    
    # Beta2
    A=matrix(c(0,0,1),ncol=p)
    denominador <- numerador + t(A %*% beta_gorro - c) %*% solve(A %*% solve(t(X)%*%X) %*% t(A)) %*% (A %*% beta_gorro - c)
    LRT <- (numerador/denominador)^(n/2)
    F_T <- (n-p)*(LRT^(-2/n)-1) < qf(0.05,1,n-p) # si es mayor, se rechaza H0
    aux5[k] <- ifelse(F_T, 0, 1) # cuenta las veces que NO se rechaza H0, decisión buena
    aux11[k] <- LRT
    
    # ahora las pruebas de hipótesis asumiendo normalidad
    modelo <- lm(y ~ x1 + x2)
    a <- summary(modelo)
    aux6[k] <- ifelse(a$coefficients[1,4]<0.05, 1, 0) # cuenta las veces que se rechaza H0 
    aux7[k] <- ifelse(a$coefficients[2,4]<0.05, 1, 0)# cuenta las veces que se rechaza H0 
    aux8[k] <- ifelse(a$coefficients[3,4]<0.05, 0, 1)# cuenta las veces que NO se rechaza H0 
    aux12[k] <- a$coefficients[1,4] 
    aux13[k] <- a$coefficients[2,4]
    aux14[k] <- a$coefficients[3,4]
  }
  return(list(beta_gorro = beta_gorro,
              sigma2_gorroEU = aux1,
              sigma2_gorroNorm = aux2,
              bien_H0_E_beta0 = aux3,
              bien_H0_E_beta1 = aux4,
              bien_H0_E_beta2 = aux5,
              bien_H0_norm_beta0 = aux6,
              bien_H0_norm_beta1 = aux7,
              bien_H0_norm_beta2 = aux8,
              LR_H0_E_beta0 = aux9,
              LR_H0_E_beta1 = aux10,
              LR_H0_E_beta2 = aux11,
              pvalue_H0_norm_beta0 = aux12,
              pvalue_norm_beta1 = aux13,
              pvalue_norm_beta2 = aux14))
}

# Aquí pruebo la función simula_laplace

a <- simula_tStudent(n = 300, betas = c(-5, 9),
                     df = m,location = mu,scale = sqrt(3),
                     semilla = 133, m = 800)
a$beta_gorro
mean(a$sigma2_gorroEU)
mean(a$sigma2_gorroNorm)
# número de veces que se rechaza H0
# como beta0 != 0, lo que queremos que siempre se rechace
# así que aquí queremos ver un 1000 :D
sum(a$bien_H0_E_beta0) 
sum(a$bien_H0_norm_beta0) 
# como beta1 != 0, queremos que siempre se rechace
# así que aquí queremos ver un 1000 :D
sum(a$bien_H0_E_beta1)
sum(a$bien_H0_norm_beta1)
# como beta2 = 0, nunca queremos rechazar
# así que aquí queremos ver un 0
sum(a$bien_H0_E_beta2)
sum(a$bien_H0_norm_beta0)

# Ahora sí
# voy a hacer vm = c(100, 300, 1000) repeticiones, esto es, primero m=100,
# luego m = 300 y por último m = 1000, para diferentes tamaños de muestra
# sería vn = c(30, 100, 500, 1000) a ver qué pasa

vn = c(30, 100, 500, 1000)
vm = c(100, 300, 1000)
df <- data.frame(n = c(vn, vn),
                 Estimador = factor(c(rep("Elíptico",length(vn)), rep("Normal",length(vn)))),
                 sigma2_gorro = c(rep(0,2*length(vn))),
                 L1 = c(rep(0,2*length(vn))),
                 U1 = c(rep(0,2*length(vn))))
# para m = 100
for(i in 1:length(vn)){
  a <- simula_tStudent(betas = c(-5,11),df = m, location = mu,
                       scale = sqrt(sigma2), semilla = 123,
                       m = vm[1],n = vn[i])
  df[i,3] <- mean(a$sigma2_gorroEU)
  df[i,4] <- mean(a$sigma2_gorroEU)-sd(a$sigma2_gorroEU)
  df[i,5] <- mean(a$sigma2_gorroEU)+sd(a$sigma2_gorroEU)
  df[length(vn)+i,3] <- mean(a$sigma2_gorroNorm)
  df[length(vn)+i,4] <- mean(a$sigma2_gorroNorm)-sd(a$sigma2_gorroNorm)
  df[length(vn)+i,5] <- mean(a$sigma2_gorroNorm)+sd(a$sigma2_gorroNorm)
}

ggplot(df, aes(x=n, y=sigma2_gorro, colour=Estimador), ) + 
  geom_errorbar(aes(ymin=L1, ymax=U1)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept=sigma2, linetype="dashed", color = "steelblue") +
  ylab(TeX(r'(Estimación de $\sigma^2$)')) +
  ggtitle(label = TeX(r'(Media de la estimación de $\sigma^2$ con)'), subtitle = paste("m =",vm[1]))

# para m = 300
for(i in 1:length(vn)){
  a <- simula_tStudent(betas = c(-5,11),df = m, location = mu,
                       scale = sqrt(sigma2), semilla = 123,
                       m = vm[1],n = vn[i])
  df[i,3] <- mean(a$sigma2_gorroEU)
  df[i,4] <- mean(a$sigma2_gorroEU)-sd(a$sigma2_gorroEU)
  df[i,5] <- mean(a$sigma2_gorroEU)+sd(a$sigma2_gorroEU)
  df[length(vn)+i,3] <- mean(a$sigma2_gorroNorm)
  df[length(vn)+i,4] <- mean(a$sigma2_gorroNorm)-sd(a$sigma2_gorroNorm)
  df[length(vn)+i,5] <- mean(a$sigma2_gorroNorm)+sd(a$sigma2_gorroNorm)
}

ggplot(df, aes(x=n, y=sigma2_gorro, colour=Estimador), ) + 
  geom_errorbar(aes(ymin=L1, ymax=U1)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept=sigma2, linetype="dashed", color = "steelblue") +
  ylab(TeX(r'(Estimación de $\sigma^2$)')) +
  ggtitle(label = TeX(r'(Media de la estimación de $\sigma^2$ con)'), subtitle = paste("m =",vm[2]))

# para m = 1000
for(i in 1:length(vn)){
  a <- simula_tStudent(betas = c(-5,11),df = m, location = mu,
                       scale = sqrt(sigma2), semilla = 123,
                       m = vm[1],n = vn[i])
  df[i,3] <- mean(a$sigma2_gorroEU)
  df[i,4] <- mean(a$sigma2_gorroEU)-sd(a$sigma2_gorroEU)
  df[i,5] <- mean(a$sigma2_gorroEU)+sd(a$sigma2_gorroEU)
  df[length(vn)+i,3] <- mean(a$sigma2_gorroNorm)
  df[length(vn)+i,4] <- mean(a$sigma2_gorroNorm)-sd(a$sigma2_gorroNorm)
  df[length(vn)+i,5] <- mean(a$sigma2_gorroNorm)+sd(a$sigma2_gorroNorm)
}

ggplot(df, aes(x=n, y=sigma2_gorro, colour=Estimador), ) + 
  geom_errorbar(aes(ymin=L1, ymax=U1)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept=sigma2, linetype="dashed", color = "steelblue") +
  ylab(TeX(r'(Estimación de $\sigma^2$)')) +
  ggtitle(label = TeX(r'(Media de la estimación de $\sigma^2$ con)'), subtitle = paste("m =",vm[3]))

# Ahora, una gráfica de las pruebas de hipótesis

df2 <- data.frame(n = c(vn,vn,vn),
                  Beta = factor(c(rep("Beta0",length(vn)),rep("Beta1",length(vn)),rep("Beta2",length(vn)))),
                  Cociente = rep(0,3*length(vn)))
# para m = 100
for(i in 1:length(vn)){# decisiones eplit/norm
  a <- simula_tStudent(betas = c(-5,11),df = m, location = mu,
                       scale = sqrt(sigma2), semilla = 123,
                       m = vm[1],n = vn[i])
  df2[i,3] <- sum(a$bien_H0_E_beta0)/sum(a$bien_H0_norm_beta0)
  df2[i+length(vn),3] <- sum(a$bien_H0_E_beta1)/sum(a$bien_H0_norm_beta1)
  df2[i+2*length(vn),3] <- sum(a$bien_H0_E_beta2)/sum(a$bien_H0_norm_beta2)
}

ggplot(df2, aes(x=n, y=Cociente, colour=Beta), ) + 
  geom_point() +
  ylab("Cociente") +
  ggtitle(label = "Cociente de decisiónes correctas elíticas/normales", subtitle = paste("m =",vm[1])) +
  geom_hline(yintercept=1, linetype="dashed", color = "steelblue") 

# para m = 300
for(i in 1:length(vn)){# decisiones eplit/norm
  a <- simula_tStudent(betas = c(-5,11),df = m, location = mu,
                       scale = sqrt(sigma2), semilla = 123,
                       m = vm[1],n = vn[i])
  df2[i,3] <- sum(a$bien_H0_E_beta0)/sum(a$bien_H0_norm_beta0)
  df2[i+length(vn),3] <- sum(a$bien_H0_E_beta1)/sum(a$bien_H0_norm_beta1)
  df2[i+2*length(vn),3] <- sum(a$bien_H0_E_beta2)/sum(a$bien_H0_norm_beta2)
}

ggplot(df2, aes(x=n, y=Cociente, colour=Beta)) + 
  geom_point() +
  ylab("Cociente") +
  ggtitle(label = "Cociente de decisiónes correctas", subtitle = paste("m =",vm[2])) +
  geom_hline(yintercept=1, linetype="dashed", color = "steelblue") 



# para m = 1000
for(i in 1:length(vn)){# decisiones eplit/norm
  a <- simula_tStudent(betas = c(-5,11),df = m, location = mu,
                       scale = sqrt(sigma2), semilla = 123,
                       m = vm[1],n = vn[i])
  df2[i,3] <- sum(a$bien_H0_E_beta0)/sum(a$bien_H0_norm_beta0)
  df2[i+length(vn),3] <- sum(a$bien_H0_E_beta1)/sum(a$bien_H0_norm_beta1)
  df2[i+2*length(vn),3] <- sum(a$bien_H0_E_beta2)/sum(a$bien_H0_norm_beta2)
}

ggplot(df2, aes(x=n, y=Cociente, colour=Beta), ) + 
  geom_point() +
  ylab("Cociente") +
  ggtitle(label = "Cociente de decisiónes correctas elíticas/normales", subtitle = paste("m =",vm[3])) +
  geom_hline(yintercept=1, linetype="dashed", color = "steelblue") 







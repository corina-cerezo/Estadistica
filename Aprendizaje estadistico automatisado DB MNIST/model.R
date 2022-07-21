library(glmnet)
library(caret) # Para la matriz de confusión
library(doParallel)# para trabajar los núcleos de la compu en paralelo
registerDoParallel(4)

# Ejercicio 2 -------------------------------------------------------------

# Se leen los datos
MNISTtrain <- read.csv("MNISTtrain_40000.csv")
MNISTtrain[,785] <- as.factor(MNISTtrain[,785])

MNISTtest <- read.csv("MNISTtest_9000.csv")
MNISTtest[,785] <- as.factor(MNISTtest[,785])

MNISTvalidate <- read.csv("MNISTvalidate_11000.csv")

# Modelo no regularizado

# Función que devuelve las estadísticas para un vector dado de alphas
# Selecciona al mejor modelo y de él predice los datos de test y train

ajusta.aplhas <- function(datostrain, datostest, datosvalida, nfolds = 10, alpha){
  estadisticas <- data.frame(alfa = NA,
                             train_accuracy = NA,
                             test_accuracy = NA,
                             test_ac0 = NA,
                             test_ac1 = NA,
                             test_ac2 = NA,
                             test_ac3 = NA,
                             test_ac4 = NA,
                             test_ac5 = NA,
                             test_ac6 = NA,
                             test_ac7 = NA,
                             test_ac8 = NA,
                             test_ac9 = NA,
                             lambda_min = NA,
                             lambda_1se = NA)
  cont <- 1
  for(a in alpha){
    print(c("entra con alpha=",a))
    Sys.time()
    # a=0.2; datostrain = datostrain = MNISTtrain[1:1000,];
    # datostest = MNISTtrain[2000:3000,]; nfolds=10; cont = 2
    fit=cv.glmnet(x = as.matrix(datostrain[,1:784]),
                  y = datostrain[,785],
                  family = "multinomial",
                  alpha = a,
                  nfolds = nfolds,
                  parallel=TRUE)
    
    estadisticas[cont,"alfa"] <- a
    
    # error de entrenamiento
    entrena=predict(fit,as.matrix(datostrain[,1:784]),s=fit$lambda.min,type="class")
    aux <- confusionMatrix(data=as.factor(entrena), reference = as.factor(datostrain[,785]))
    estadisticas$train_accuracy[cont] <- aux$overall[1]
    
    # error de predicción
    pred=predict(fit,as.matrix(datostest[,1:784]),s=fit$lambda.min,type="class")
    aux2 <- confusionMatrix(data=as.factor(pred), reference = as.factor(datostest[,785]))
    estadisticas$test_accuracy[cont] <- aux2$overall[1]
    estadisticas$test_ac0[cont] <- aux2$byClass[1,1]
    estadisticas$test_ac1[cont] <- aux2$byClass[2,1]
    estadisticas$test_ac2[cont] <- aux2$byClass[3,1]
    estadisticas$test_ac3[cont] <- aux2$byClass[4,1]
    estadisticas$test_ac4[cont] <- aux2$byClass[5,1]
    estadisticas$test_ac5[cont] <- aux2$byClass[6,1]
    estadisticas$test_ac6[cont] <- aux2$byClass[7,1]
    estadisticas$test_ac7[cont] <- aux2$byClass[8,1]
    estadisticas$test_ac8[cont] <- aux2$byClass[9,1]
    estadisticas$test_ac9[cont] <- aux2$byClass[10,1]
    
    estadisticas$lambda_min[cont] <- fit$lambda.min
    estadisticas$lambda_1se[cont] <- fit$lambda.1se
    
    print(c("sale con alpha=",a))
    beepr::beep(5) 
    Sys.time()
    
    esta_prec = aux2$overall[1]
    
    if(cont == 1){
      predtest = pred
      predvalidados = predict(fit,as.matrix(datosvalida[,1:784]),s=fit$lambda.min,type="class")
      mejormodelo = fit
      mejoralfa = a
    }
    else if(esta_prec == max(estadisticas$test_accuracy)){
      predtest = pred
      predvalidados = predict(fit,as.matrix(datosvalida[,1:784]),s=fit$lambda.min,type="class")
      mejormodelo = fit
      mejoralfa = a
    }
    cont <- cont + 1
  }
  lista <- list()
  lista$mejoralfa = mejoralfa
  lista$mejormodelo = mejormodelo
  lista$estadisticas = estadisticas
  lista$predtest = predtest
  lista$predval = predvalidados
  return(lista)
}

# a ver cuanto tarda una vez con todos los datos
Sys.time()
corrida2google <- ajusta.aplhas(datostrain = MNISTtrain,
                      datostest = MNISTtest,
                      datosvalida = MNISTvalidate,
                      nfolds = 10,
                      alpha = rep((0:10)/10,5))
Sys.time()
beepr::beep(8) 

rm(MNISTtrain)
rm(MNISTtest)
rm(MNISTvalidate)

print(c("La mejor alpfa es:",corrida2google$mejoralfa))

save.image("corrida2google.RData")
write.table(corrida2google$estadisticas,"C2GoogleEstaditicas.txt")
write.table(corrida2google$predtest,"C2GooglePredTest.txt")
write.table(corrida2google$predval,"C2GooglePredVal.txt")



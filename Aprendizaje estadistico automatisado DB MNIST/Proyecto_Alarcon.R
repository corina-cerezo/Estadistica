#            Aprendizaje Estad\'istico Automatizado
#            Proyecto final: Redes Neuronales (NN).
#            Profesora: Guillermina Eslava G\'omez
#               Alumnos:  Alarc\'on-Gonz\'alez, 
#                         Cerezo-Silva,
#                         Zarco-Romero.
#
library(h2o)
library(randomForest)
library(caret) # para validación cruzada
library(xtable)
h2o.init()
#h2o.shutdown(prompt = FALSE) #por si se acaba la memoria de java.
library(glmnet)
library(caret) # Para la matriz de confusión
library(doParallel)# para trabajar los núcleos de la compu en paralelo
library(xlsx)
library(latex2exp)
library(knitr)
library(kableExtra)
library(dplyr)
library(tidyr)
library(graphics)
options(width=160,digits=4)

# h2o
library(h2o)
h2o.init(nthreads = -1)  #Conexión a internet
h2o.getVersion() #"3.32.1.3"


registerDoParallel(4)

#setwd("C:/Users/migue/Desktop/proyectobrenda")
setwd("C:/Users/Brendis/Google Drive/Maestría en Ciencias Matemáticas/Aprendizaje estadístico automatizado/Proyecto")

# Funciones auxiliares ----------------------------------------------------

# LaTex en plots
LaTeX <- function(...){
  latex2exp::TeX(paste0(...)) 
}

# ¿Qué número quieres pintar?
pinta.num <- function(registro,datos,respuesta=FALSE,predict=NULL,valida = FALSE,ajustes) {
  
  # registro  := Número de registro que se quiere visualizar
  # datos     := Los pixeles de los números a visualizar en formato data.frame
  # respuesta := ¿Los datos tienen la respuesta? (Columna C785)
  # predict   := Vector columna de predicciones de TODO "datos".
  # valida    := indicar si se van a pintar datos de la base de validación y comparar ajustes
  # ajustes   := vector de las tres predicciones de la base de validación
  
  plot(c(0, 28), c(0, 28), type = "n", xlab = "", ylab = "",yaxt='n',xaxt='n')
  for (i in 1:28) {
    for (j in 1:28) { # i=1; j=1
      img.rgb = as.raster((255 - datos[registro, j + 28 * (i - 1)]) / 255)
      rect(j - 1, 28 - i - 1, j, 28 - i, col = img.rgb, border = NA)
    }
  }
  
  if(respuesta & !is.null(predict)){
    legend("topleft", 
           legend=c(LaTeX("Y = ",datos[registro,"C785"]),
                    LaTeX("$\\hat{Y}$ = ",predict[registro])), 
           fill=c("blue","red"),
           title="Respuesta", 
           text.font=4, bg='lightblue')
  }else if(respuesta){
    legend("topleft", 
           legend=LaTeX("Y = ",datos[registro,"C785"]), 
           fill="blue", title="Respuesta", 
           text.font=4, bg='lightblue')
  }else if(!is.null(predict)){
    legend("topleft", 
           legend=LaTeX("$\\hat{Y}$ = ",predict[registro]), 
           fill="red", title="Respuesta", 
           text.font=4, bg='lightblue')
  }
  
  if(valida){
    legend("topleft", 
           legend=c(LaTeX("$\\hat{Y}_{RF}$ = ",ajustes[1]),
                    LaTeX("$\\hat{Y}_{LogReg}$ = ",ajustes[2]),
                    LaTeX("$\\hat{Y}_{DNN}$ = ",ajustes[3])),
           #fill=c("blue","red"),
           #title="Ajuste", 
           text.font=4, bg='lightblue')
  }
  
}

# Función para exportar las tablas a latex
kabla <- function(data,title=NULL,hold=TRUE,row.names=FALSE){
  library(knitr)
  library(kableExtra)
  if(hold) {
    kable(data, "latex",
          caption = title,
          row.names = row.names,
          align = "c",
          booktabs = T) %>%
      kable_styling(latex_options = c("striped", "hold_position"))
  } else{
    kable(data, "latex",
          caption = title,
          row.names = row.names,
          align = "c",
          booktabs = T) %>%
      kable_styling(latex_options = c("striped"))
  }
}

# Para tablas en PDF-Markdown.
kabla <- function(data,title=NULL,ref=NULL,hold=TRUE,row.names=FALSE){
  library(knitr)
  library(kableExtra)
  if(hold) {
    kable(data, "latex",
          caption = paste0(title," \\label{",ref,"}"),
          row.names = row.names,
          align = "c",
          booktabs = T) %>%
      kable_styling(latex_options = c("striped", "hold_position"))
  } else{
    kable(data, "latex",
          caption = paste0(title," \\label{",ref,"}"),
          row.names = row.names,
          align = "c",
          booktabs = T) %>%
      kable_styling(latex_options = c("striped"))
  }
}

substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}


substrLeft <- function(x, n){
  substr(x, 1, n)
}

copy2clipboard <- function(x){
  clipr::write_clip(x)
}


# Redes Neuronales Profundas (DNN) ----------------------------------------

# # Precarga
# load(file = "DNN_Resultados.RData")
# # Lo pasamos a formato H2OFrame
# train <- as.h2o(MNISTtrain)
# test <- as.h2o(MNISTtest)

# Ajuste de datos ---------------------------------------------------------

# Guardando variables y covariables
y <- "C785"  # Variale respuesta
x <- setdiff(x = names(MNISTtrain),y = y)  # Variables explicativas

# Hacemos la respuesta una variable categórica pues estamos en un problema
# de clasificación.
MNISTtrain[, y] <- as.factor(MNISTtrain[, y])
MNISTtest[, y]  <- as.factor(MNISTtest[, y])

# Lo pasamos a formato H2OFrame
train <- as.h2o(MNISTtrain)
test <- as.h2o(MNISTtest)

# Simples -----------------------------------------------------------------


# _ Fit 1 -------------------------------------------------------------------

# Creando el modelo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# set.seed(2012)
# system.time({
#   DNN_fit1 <- h2o.deeplearning(x = x, y = y,
#                                training_frame = train,
#                                model_id = "DNN_fit1")
#   h2o.saveModel(object = DNN_fit1,path = paste0(getwd(),"/Modelos"),force=TRUE) %>% try()
# }) -> DNN_fit1.time

# user  system elapsed 
# 0.39    0.00   67.93 

# DNN_fit1 <- h2o.loadModel(paste0(getwd(),"/Modelos/DNN_fit1"))

# Clase y probabilidades
DNN_Prd1Trn=h2o.predict(DNN_fit1, newdata=train) 
DNN_Prd1Tst=h2o.predict(DNN_fit1, newdata=test)

# Valuando el modelo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# General ******************************************************************

# Épocas
par(mfrow=c(1,1),mai=c(1,1,1,1))
plot(DNN_fit1) #Training classification error
# Log-pérdida
h2o.logloss(DNN_fit1)

# Training ******************************************************************

# Rendimiento (Performance)

## Esto genera muchas medidas a analizar que podemos visualizar del modelo.
DNN_prf1Trn = h2o.performance(model = DNN_fit1, newdata = train)

# Log-pérdida
h2o.logloss(DNN_prf1Trn) #on training set (depende de la corrida)

# Matriz de confusión

# Opción 1
DNN_CMfit1 <- h2o.confusionMatrix(DNN_fit1, train)
DNN_CMfit1                   # Matriz en general
DNN_CMfit1$Error             # Columna de errores
DNN_CMfit1["Totals","Error"] # Tasa de error global

# Opción 2
DNN_prf1Trn@metrics$cm$table
DNN_prf1Trn@metrics$cm$table["Totals","Error"]
DNN_prf1Trn@metrics$cm$table["Totals","Rate"]

# Test **********************************************************************

# Rendimiento (Performance)

## Esto genera muchas medidas a analizar que podemos visualizar del modelo.
DNN_prf1Tst = h2o.performance(model = DNN_fit1, newdata = test)

# Log-pérdida
h2o.logloss(DNN_prf1Tst) #on training set (depende de la corrida)

# Matriz de confusión

# Opción 1
DNN_CMfit1Tst <- h2o.confusionMatrix(DNN_fit1, test)
DNN_CMfit1Tst                   # Matriz en general
DNN_CMfit1Tst$Error             # Columna de errores
DNN_CMfit1Tst["Totals","Error"] # Tasa de error global

# Opción 2
DNN_prf1Tst@metrics$cm$table
DNN_prf1Tst@metrics$cm$table["Totals","Error"]
DNN_prf1Tst@metrics$cm$table["Totals","Rate"]

# Visualización de las clasificaciones mal realizadas ~~~~~~~~~~~~~~~~~~~~~~~

# Training ******************************************************************

# Seleccionamos los mal clasificados
DNN_Mal1Trn <- MNISTtrain[,y] %>% as.character() != DNN_Prd1Trn[,"predict"] %>% as.vector() %>% as.character()
sum(DNN_Mal1Trn) # ¿Cuántas están mal clasificadas?

# ¿Cuál es el peor clasificado?
MNISTtrain[which(DNN_Mal1Trn),y] %>% table()

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.2,.2))
for (i in 1:9) {
  pinta.num(i,MNISTtrain[which(DNN_Mal1Trn),],TRUE,
            DNN_Prd1Trn[which(DNN_Mal1Trn),"predict"] %>% as.vector())
}
par(mfrow = c(1,1))

# Test **********************************************************************

# Seleccionamos los mal clasificados
DNN_Mal1Tst <- MNISTtest[,y] %>% as.character() != DNN_Prd1Tst[,"predict"] %>% as.vector() %>% as.character()
sum(DNN_Mal1Tst) # ¿Cuántas están mal clasificadas?

# ¿Cuál es el peor clasificado?
MNISTtest[which(DNN_Mal1Tst),y] %>% table()

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.2,.2))
for (i in 1:9) {
  pinta.num(i,MNISTtest[which(DNN_Mal1Tst),],TRUE,
            DNN_Prd1Tst[which(DNN_Mal1Tst),"predict"] %>% as.vector())
}
par(mfrow = c(1,1))

# _ Fit 2 -------------------------------------------------------------------

# Creando el modelo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# set.seed(2012)
# system.time({
#   DNN_fit2 <- h2o.deeplearning(x = x, y = y,
#                                training_frame = train,
#                                model_id = "DNN_fit2",
#                                #Parámetros
#                                hidden = c(500,500,500,500,500),
#                                epochs = 25,
#                                l1 = 0)
#   h2o.saveModel(object = DNN_fit2,path = paste0(getwd(),"/Modelos"),force = TRUE) %>% try()
# }) -> DNN_fit2.time

# user  system elapsed 
# 0.55    0.04  592.20

# DNN_fit2 <- h2o.loadModel(paste0(getwd(),"/Modelos/DNN_fit2"))

# Clase y probabilidades
DNN_Prd2Trn=h2o.predict(DNN_fit2, newdata=train) 
DNN_Prd2Tst=h2o.predict(DNN_fit2, newdata=test)

# Valuando el modelo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# General ******************************************************************

# Épocas
par(mfrow=c(1,1),mai=c(1,1,1,1))
plot(DNN_fit2) #Training classification error
# Log-pérdida
h2o.logloss(DNN_fit2)

# Training ******************************************************************

# Rendimiento (Performance)

## Esto genera muchas medidas a analizar que podemos visualizar del modelo.
DNN_prf2Trn = h2o.performance(model = DNN_fit2, newdata = train)

# Log-pérdida
h2o.logloss(DNN_prf2Trn) #on training set (depende de la corrida)

# Matriz de confusión

# Opción 1
DNN_CMfit2 <- h2o.confusionMatrix(DNN_fit2, train)
DNN_CMfit2                   # Matriz en general
DNN_CMfit2$Error             # Columna de errores
DNN_CMfit2["Totals","Error"] # Tasa de error global

# Opción 2
DNN_prf2Trn@metrics$cm$table
DNN_prf2Trn@metrics$cm$table["Totals","Error"]
DNN_prf2Trn@metrics$cm$table["Totals","Rate"]

# Test **********************************************************************

# Rendimiento (Performance)

## Esto genera muchas medidas a analizar que podemos visualizar del modelo.
DNN_prf2Tst = h2o.performance(model = DNN_fit2, newdata = test)

# Log-pérdida
h2o.logloss(DNN_prf2Tst) #on training set (depende de la corrida)

# Matriz de confusión

# Opción 1
DNN_CMfit2Tst <- h2o.confusionMatrix(DNN_fit2, test)
DNN_CMfit2Tst                   # Matriz en general
DNN_CMfit2Tst$Error             # Columna de errores
DNN_CMfit2Tst["Totals","Error"] # Tasa de error global

# Opción 2
DNN_prf2Tst@metrics$cm$table
DNN_prf2Tst@metrics$cm$table["Totals","Error"]
DNN_prf2Tst@metrics$cm$table["Totals","Rate"]

# Visualización de las clasificaciones mal realizadas ~~~~~~~~~~~~~~~~~~~~~~~

# Training ******************************************************************

# Seleccionamos los mal clasificados
DNN_Mal2Trn <- MNISTtrain[,y] %>% as.character() != DNN_Prd2Trn[,"predict"] %>% as.vector() %>% as.character()
sum(DNN_Mal2Trn) # ¿Cuántas están mal clasificadas?

# ¿Cuál es el peor clasificado?
MNISTtrain[which(DNN_Mal2Trn),y] %>% table()

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.2,.2))
for (i in 1:9) {
  pinta.num(i,MNISTtrain[which(DNN_Mal2Trn),],TRUE,
            DNN_Prd2Trn[which(DNN_Mal2Trn),"predict"] %>% as.vector())
}
par(mfrow = c(1,1))

# Test **********************************************************************

# Seleccionamos los mal clasificados
DNN_Mal2Tst <- MNISTtest[,y] %>% as.character() != DNN_Prd2Tst[,"predict"] %>% as.vector() %>% as.character()
sum(DNN_Mal2Tst) # ¿Cuántas están mal clasificadas?

# ¿Cuál es el peor clasificado?
MNISTtest[which(DNN_Mal2Tst),y] %>% table()

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.5,.2))
faux <- function(x){which(MNISTtest[which(DNN_Mal2Tst),]$C785==x)[1]}
NUM <- sapply(1:9,faux)
count=0
for (i in NUM) {
  pinta.num(i,MNISTtest[which(DNN_Mal2Tst),],TRUE,
            DNN_Prd2Tst[which(DNN_Mal2Tst),"predict"] %>% as.vector())
  count=count+1
  if(count==3){par(mai = c(.2,.2,.2,.2)) }
}
mtext("Mal Asignados", side = 3, line = -2.75, outer = TRUE,cex=2)
par(mfrow = c(1,1))

# Mostramos 9 números.
# Gráficos
faux <- function(x){which(MNISTtest$C785==x)[1]}
NUM <- sapply(1:9,faux)
par(mfrow = c(3,3),mai = c(.2,.2,.5,.2))
count=0
for (i in NUM) {
  pinta.num(i,MNISTtest,TRUE,
            DNN_Prd2Tst[,"predict"] %>% as.vector())
  count=count+1
  if(count==3){par(mai = c(.2,.2,.2,.2)) }
}
mtext("Bien Asignados", side = 3, line = -2.75, outer = TRUE,cex=2)
par(mfrow = c(1,1))


# _ Fit 3 -------------------------------------------------------------------

# Creando el modelo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# set.seed(2012)
# system.time({
#   DNN_fit3 <- h2o.deeplearning(x = x, y = y,
#                                training_frame = train,
#                                model_id = "DNN_fit3",
#                                #Parámetros
#                                hidden = c(2500,2500),
#                                epochs = 40,
#                                l1 = 0)
#   h2o.saveModel(object = DNN_fit3,path = paste0(getwd(),"/Modelos"),force=TRUE) %>% try()
# }) -> DNN_fit3.time

# user  system elapsed 
# 3.18    0.29 3612.86

# DNN_fit3 <- h2o.loadModel(paste0(getwd(),"/Modelos/DNN_fit3"))

# Clase y probabilidades
DNN_Prd3Trn=h2o.predict(DNN_fit3, newdata=train) 
DNN_Prd3Tst=h2o.predict(DNN_fit3, newdata=test)

# Valuando el modelo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# General ******************************************************************

# Épocas
par(mfrow=c(1,1),mai=c(1,1,1,1))
plot(DNN_fit3) #Training classification error
# Log-pérdida
h2o.logloss(DNN_fit3)

# Training ******************************************************************

# Rendimiento (Performance)

## Esto genera muchas medidas a analizar que podemos visualizar del modelo.
DNN_prf3Trn = h2o.performance(model = DNN_fit3, newdata = train)

# Log-pérdida
h2o.logloss(DNN_prf3Trn) #on training set (depende de la corrida)

# Matriz de confusión

# Opción 1
DNN_CMfit3 <- h2o.confusionMatrix(DNN_fit3, train)
DNN_CMfit3                   # Matriz en general
DNN_CMfit3$Error             # Columna de errores
DNN_CMfit3["Totals","Error"] # Tasa de error global

# Opción 2
DNN_prf3Trn@metrics$cm$table
DNN_prf3Trn@metrics$cm$table["Totals","Error"]
DNN_prf3Trn@metrics$cm$table["Totals","Rate"]

# Test **********************************************************************

# Rendimiento (Performance)

## Esto genera muchas medidas a analizar que podemos visualizar del modelo.
DNN_prf3Tst = h2o.performance(model = DNN_fit3, newdata = test)

# Log-pérdida
h2o.logloss(DNN_prf3Tst) #on training set (depende de la corrida)

# Matriz de confusión

# Opción 1
DNN_CMfit3Tst <- h2o.confusionMatrix(DNN_fit3, test)
DNN_CMfit3Tst                   # Matriz en general
DNN_CMfit3Tst$Error             # Columna de errores
DNN_CMfit3Tst["Totals","Error"] # Tasa de error global

# Opción 2
DNN_prf3Tst@metrics$cm$table
DNN_prf3Tst@metrics$cm$table["Totals","Error"]
DNN_prf3Tst@metrics$cm$table["Totals","Rate"]

# Visualización de las clasificaciones mal realizadas ~~~~~~~~~~~~~~~~~~~~~~~

# Training ******************************************************************

# Seleccionamos los mal clasificados
DNN_Mal3Trn <- MNISTtrain[,y] %>% as.character() != DNN_Prd3Trn[,"predict"] %>% as.vector() %>% as.character()
sum(DNN_Mal3Trn) # ¿Cuántas están mal clasificadas?

# ¿Cuál es el peor clasificado?
MNISTtrain[which(DNN_Mal3Trn),y] %>% table()

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.2,.2))
for (i in 1:9) {
  pinta.num(i,MNISTtrain[which(DNN_Mal3Trn),],TRUE,
            DNN_Prd3Trn[which(DNN_Mal3Trn),"predict"] %>% as.vector())
}
par(mfrow = c(1,1))

# Test **********************************************************************

# Seleccionamos los mal clasificados
DNN_Mal3Tst <- MNISTtest[,y] %>% as.character() != DNN_Prd3Tst[,"predict"] %>% as.vector() %>% as.character()
sum(DNN_Mal3Tst) # ¿Cuántas están mal clasificadas?

# ¿Cuál es el peor clasificado?
MNISTtest[which(DNN_Mal3Tst),y] %>% table()

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.2,.2))
for (i in 1:9) {
  pinta.num(i,MNISTtest[which(DNN_Mal3Tst),],TRUE,
            DNN_Prd3Tst[which(DNN_Mal3Tst),"predict"] %>% as.vector())
}
par(mfrow = c(1,1))


# _ Resultados Simple -----------------------------------------------------

# Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# fit1
DNN_CMfit1["Totals","Error"] # Tasa de error global
DNN_CMfit1["Totals","Rate"] # Tasa de error global
# fit2
DNN_CMfit2["Totals","Error"] # Tasa de error global
DNN_CMfit2["Totals","Rate"] # Tasa de error global
# fit3
DNN_CMfit3["Totals","Error"] # Tasa de error global
DNN_CMfit3["Totals","Rate"] # Tasa de error global

# Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# fit1
DNN_CMfit1Tst["Totals","Error"] # Tasa de error global
DNN_CMfit1Tst["Totals","Rate"] # Tasa de error global
# fit2
DNN_CMfit2Tst["Totals","Error"] # Tasa de error global
DNN_CMfit2Tst["Totals","Rate"] # Tasa de error global
# fit3
DNN_CMfit3Tst["Totals","Error"] # Tasa de error global
DNN_CMfit3Tst["Totals","Rate"] # Tasa de error global

# _ LaTeX Simple ----------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Matrix of model parameters and metrics
Resultados_Simple=matrix(NA,nrow=3,ncol=12)
colnames(Resultados_Simple)=c("Model","RunTimeMins","Layers","Epochs","l1","InDropoutRat",
                              "Loglss_trn","Loglss_tst","Loglss_cv",
                              "Err_trn","Err_tst","Err_cv")

mfit=list(DNN_fit1,DNN_fit2,DNN_fit3)
mprfTrn=list(DNN_prf1Trn,DNN_prf2Trn,DNN_prf3Trn)
mprfTst=list(DNN_prf1Tst,DNN_prf2Tst,DNN_prf3Tst)

for (i in 1:3){
  Resultados_Simple[i,1]=mfit[[i]]@model_id
  Resultados_Simple[i,2]=round(mfit[[i]]@model$run_time/60000, digits=2) #valor incorrecto para dl_fit3!
  Resultados_Simple[i,3]=paste(mfit[[i]]@allparameters$hidden, collapse="-")
  Resultados_Simple[i,4]=mfit[[i]]@allparameters$epochs
  Resultados_Simple[i,5]=mfit[[i]]@allparameters$l1
  Resultados_Simple[i,6]=mfit[[i]]@allparameters$input_dropout_ratio 
  Resultados_Simple[i,7]=round(h2o.logloss(mprfTrn[[i]]), digits=4)
  Resultados_Simple[i,8]=round(h2o.logloss(mprfTst[[i]]), digits=4)
  # Resultados_Simple[i,9]=mfit[[i]]@model$cross_validation_metrics_summary$mean[4]
  Resultados_Simple[i,10]=round(mprfTrn[[i]]@metrics$cm$table$Error[11], digits=4)
  Resultados_Simple[i,11]=round(mprfTst[[i]]@metrics$cm$table$Error[11], digits=4)
  # Resultados_Simple[i,12]=mfit[[i]]@model$cross_validation_metrics_summary$mean[2]
}

#Resultados_Simple

# En este caso no hicimos CV
# Resultados_Simple[3,9] =mfit[[3]]@model$cross_validation_metrics_summary$mean[6]  #logloss
# Resultados_Simple[3,12]=mfit[[3]]@model$cross_validation_metrics_summary$mean[4]  #overall error

# Ajuste visual de la tabla
Resultados_Simple <- Resultados_Simple %>% subset(select=-c(l1,InDropoutRat,Loglss_cv,Err_cv)) %>%  
  as.data.frame() %>% arrange(Err_tst)
# Errores
Resultados_Simple[,c("Err_trn","Err_tst")] <- 100 * (Resultados_Simple[,c("Err_trn","Err_tst")] %>% apply(2,as.numeric))
colnames(Resultados_Simple)[c(7,8)] <- c("%Err_trn","%Err_tst")
# Loglss
Resultados_Simple[,c("Loglss_trn","Loglss_tst")] <- (Resultados_Simple[,c("Loglss_trn","Loglss_tst")] %>% apply(2,function(x){round(100*as.numeric(x),2)}))
colnames(Resultados_Simple)[c(5,6)] <- c("%Loglss_trn","%Loglss_tst")
# Modelos
Resultados_Simple$Model <- Resultados_Simple$Model %>% extract_numeric()

# LaTeX
Resultados_Simple %>% kabla(title = "Estadísticas y parámetros de los modelos simples ajustados. \\textit{Nota: Todos los parámetros que no estén aquí mencionados fueron tomados por default.}",
                            ref = "Resultados_Simple")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

aux <- data.frame(DNN_CMfit2$Error,
                  DNN_CMfit3$Error,
                  DNN_CMfit1$Error) %>% t()
aux <- cbind(c(2,3,1),aux)
rownames(aux) <- NULL
colnames(aux) <- c("Model",0:9,"Total")
aux[,-1] <- (100*aux[,-1]) %>% round(2)
aux %>% kabla(title = "\\textbf{Porcentaje de error} de los modelos simples por clase y total en el conjunto de \\textbf{entrenamiento}.",
              ref = "NumErr_Simple") %>% copy2clipboard()

aux2 <- t(aux)[-c(1,ncol(aux)),]
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(x = 0:9,y = aux2,
        pch=19,cex=1.0, col=c("orange", "red","blue"),
        type="b",ylab="Errors", xlab="Clase",
        main="%Errores de entrenamiento por clase",xaxt="n",yaxt="n")
axis(1, at=0:9, labels=0:9,
     tck = 1,lty=2,col="gray")
axis(1, at=0:9, labels=0:9,
     tick = 1)
axis(2, at=0:8/10, labels=0:8/10,tck = 1,lty=2,col="gray")
axis(2, at=0:8/10, labels=0:8/10,tick = 1)
box()
legend("topright", legend=paste("Modelo",c(2,3,1)) %>% sapply(LaTeX), pch=19,
       col=c("orange","red","blue"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

aux <- data.frame(DNN_CMfit2Tst$Error,
                  DNN_CMfit3Tst$Error,
                  DNN_CMfit1Tst$Error) %>% t()
aux <- cbind(c(2,3,1),aux)
rownames(aux) <- NULL
colnames(aux) <- c("Model",0:9,"Total")
aux[,-1] <- (100*aux[,-1]) %>% round(2)
aux %>% kabla(title = "\\textbf{Porcentaje de error} de los modelos simples por clase y total en el conjunto de \\textbf{prueba}.",
              ref = "NumErrTst_Simple") %>% copy2clipboard()

aux2 <- t(aux)[-c(1,ncol(aux)),]
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(x = 0:9,y = aux2,
        pch=19,cex=1.0, col=c("orange", "red","blue"),
        type="b",ylab="Errors", xlab="Clase",
        main="%Errores de prueba por clase",xaxt="n")#,yaxt="n")
axis(1, at=0:9, labels=0:9,
     tck = 1,lty=2,col="gray")
axis(1, at=0:9, labels=0:9,
     tick = 1)
axis(2, at=1:8, labels=1:8,tck = 1,lty=2,col="gray")
axis(2, at=1:8, labels=1:8,tick = 1)
box()
legend("topright", legend=paste("Modelo",c(2,3,1)) %>% sapply(LaTeX), pch=19,
       col=c("orange","red","blue"))


# Mallas ------------------------------------------------------------------


# _ Malla 1 ---------------------------------------------------------------

# Vamos a construír la malla basándonos en los resultados y parámetros anteriores.  
# La idea es seleccionar modelos aleatorios con una estructura alrededor de la de los
# anteriores.

# Hiper-parámetros de los modelos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(312102319)
# Capas y nodos del modelo
hidden_options <- replicate(n = 100,simplify = FALSE,
                            expr = {
                              # Cantidad de capas
                              capas <- sample(x = 2:6,size = 1,replace = TRUE,prob = c(0.1,0.25,0.25,0.3,0.1))
                              # Número de nodos en cada capa
                              sample(c(200,300,500,1000,2500),size = capas,replace = TRUE,prob = c(0.25,0.25,0.3,0.1,0.1))
                            })
# Número de épocas
(epoch_options = c(10,25,40,50)) # params de épocas
# Parámetro de regularización.
(l1_options    = c(0,1e-5)) #2 params de regularización norma l1
# Guardamos los hypera parámetros
(hyper_params <- list(hidden = hidden_options, l1 = l1_options, epochs=epoch_options))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

search_criteria = list(
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Esto así porque son modelos MUY pesados.
  strategy ="RandomDiscrete", # Escoge de los modelos aleatoriamente
  max_models = 20, # A lo más correrá 20 modelos
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  seed=27)

# system.time({
#   DNN_grid <- h2o.grid(
#                         algorithm = "deeplearning",
#                         grid_id = "DNN_grid",
#                         hyper_params = hyper_params,
#                         search_criteria = search_criteria,
#                         x = x,
#                         y = y,
#                         activation = "RectifierWithDropout",
#                         distribution = "multinomial",
#                         training_frame = as.h2o(MNISTtrain),
#                         validation_frame = as.h2o(MNISTtest),
#                         input_dropout_ratio = 0.2,
#                         stopping_rounds = 3,
#                         stopping_tolerance = 0.05,
#                         stopping_metric = "misclassification",
#                         nfolds = 4
#              )
#   h2o.saveGrid(grid_id = "DNN_grid", grid_directory = paste0(getwd(),"/Modelos")) %>% try()
# }) -> DNN_grid.time

# user   system  elapsed 
# 25.19     2.85 18707.28 

# DNN_grid <- h2o.loadGrid(grid_path = paste0(getwd(),"/Modelos/DNN_grid"))

summary(DNN_grid, show_stack_traces = TRUE)


# __ Valuando la malla ----------------------------------------------------

# Extraemos los modelos
Modelos_Mallas <- lapply(DNN_grid@model_ids, function(id){h2o.getModel(id)})

# Vamos a ver su rendimiento en el training y test.
Modelos_Mallas.Trn <- list()
Modelos_Mallas.Tst <- list()
system.time({
  # Performance global
  Modelos_Mallas.Perf = h2o.getGrid(grid_id = "DNN_grid", sort_by = "err", decreasing = F)
  # Performance por modelo
  for(i in 1:length(Modelos_Mallas)){
    Modelos_Mallas.Trn[[i]] <- h2o.performance(Modelos_Mallas[[i]], newdata = as.h2o(MNISTtrain))
    Modelos_Mallas.Tst[[i]] <- h2o.performance(Modelos_Mallas[[i]], newdata = as.h2o(MNISTtest))
  }
}) -> Modelos_Mallas.Perf.Time
# Vemos los resultados globales.
Modelos_Mallas.Perf

# Guardamos los resultados individuales de las estadísticas de interés
Resultados_Mallas=data.frame()
for(i in 1:length(Modelos_Mallas)){
  Resultados_Mallas[i,1]=Modelos_Mallas[[i]]@model_id
  Resultados_Mallas[i,2]=round(Modelos_Mallas[[i]]@model$run_time/60000, digits=2)
  Resultados_Mallas[i,3]=paste(Modelos_Mallas[[i]]@allparameters$hidden, collapse="-")
  Resultados_Mallas[i,4]=round(Modelos_Mallas[[i]]@allparameters$epochs, digits=2)
  Resultados_Mallas[i,5]=Modelos_Mallas[[i]]@allparameters$l1
  Resultados_Mallas[i,6]=Modelos_Mallas[[i]]@allparameters$input_dropout_ratio 
  Resultados_Mallas[i,7]=round(h2o.logloss(Modelos_Mallas.Trn[[i]]), digits=4)
  Resultados_Mallas[i,8]=round(h2o.logloss(Modelos_Mallas.Tst[[i]]), digits=4)
  Resultados_Mallas[i,9]=round(Modelos_Mallas.Trn[[i]]@metrics$cm$table$Error[11], digits=4)
  Resultados_Mallas[i,10]=round(Modelos_Mallas.Tst[[i]]@metrics$cm$table$Error[11], digits=4)
}
colnames(Resultados_Mallas)=c("model","RunTimeMins","layers","epochs","l1","InDropoutRat",
                              "loglss_trn","loglss_tst","Err_trn","Err_tst")
Resultados_Mallas <- Resultados_Mallas %>% arrange(Err_tst)
Resultados_Mallas

# Graficamos los 9 modelos con test error más bajo
dev.off()
par(mfrow = c(3,3))#, mai = c(.5,.5,.5,.5))
# Plot the scoring history over time for the 5 models
for(i in 1:length(Modelos_Mallas)){
  if(Modelos_Mallas[[i]]@model_id%in%Resultados_Mallas$model[1:9]){
    plot(Modelos_Mallas[[i]],metric = "classification_error",cex=.7)
    grid()
    abline(h = 0.02,col="red",lty=2)
    text(5,0.025,0.02,col="red")
  }
}

# Gráfico de las métricas de los modelos
dev.off()
#par(mfrow = c(1,1), mai = c(.8,.8,.8,.8))
matplot(1:19, 
        cbind(Resultados_Mallas[-20,7], 
              Resultados_Mallas[-20,8],
              Resultados_Mallas[-20,9], 
              Resultados_Mallas[-20,10]
        )*100,
        pch=19, 
        col=c("lightblue", "orange","blue","red"),
        type="b",ylab="%(Errors|logloss)", xlab="models",
        main="Resultados de la Malla aleatoria",xaxt="n",yaxt="n")
axis(1, at=1:19, labels=readr::parse_number(Resultados_Mallas$model[-20]),
     tck = 1,lty=2,col="gray")
axis(1, at=1:19, labels=readr::parse_number(Resultados_Mallas$model[-20]),
     tick = 1)
axis(2, at=0:15, labels=0:15,tck = 1,lty=2,col="gray")
axis(2, at=0:15, labels=0:15,tick = 1)
box()
legend("topleft", legend=c("$logloss_{train}$", "$logloss_{test}$",
                           "$Error_{train}$", "$Error_{test}$") %>% sapply(LaTeX), pch=19,
       col=c("lightblue", "orange","blue","red"))

# Nos quedamos con el modelo final
for(i in 1:length(Modelos_Mallas)){
  if(Modelos_Mallas[[i]]@model_id==Resultados_Mallas$model[1]){
    Modelo_Mallas.Final <- Modelos_Mallas[[i]]
  }
}

# __ Valuando el modelo ---------------------------------------------------

# Clase y probabilidades
DNN_Prd.Final.Trn=h2o.predict(Modelo_Mallas.Final, newdata=as.h2o(MNISTtrain)) 
DNN_Prd.Final.Tst=h2o.predict(Modelo_Mallas.Final, newdata=as.h2o(MNISTtest))

# General ******************************************************************

# Épocas
par(mfrow=c(1,1),mai=c(1,1,1,1))
plot(Modelo_Mallas.Final) #Training classification error
# Log-pérdida
h2o.logloss(Modelo_Mallas.Final)

# Training ******************************************************************

# Rendimiento (Performance)

## Esto genera muchas medidas a analizar que podemos visualizar del modelo.
Modelo_Mallas.Final.prf.Trn = h2o.performance(model = Modelo_Mallas.Final, 
                                              newdata = train)

# Log-pérdida
h2o.logloss(Modelo_Mallas.Final.prf.Trn) #on training set (depende de la corrida)

# Matriz de confusión

# Opción 1
Modelo_Mallas.Final.CMfit <- h2o.confusionMatrix(Modelo_Mallas.Final, train)
Modelo_Mallas.Final.CMfit                   # Matriz en general
Modelo_Mallas.Final.CMfit$Error             # Columna de errores
Modelo_Mallas.Final.CMfit["Totals","Error"] # Tasa de error global

# Opción 2
Modelo_Mallas.Final.prf.Trn@metrics$cm$table
Modelo_Mallas.Final.prf.Trn@metrics$cm$table["Totals","Error"]
Modelo_Mallas.Final.prf.Trn@metrics$cm$table["Totals","Rate"]

# Test **********************************************************************

# Rendimiento (Performance)

## Esto genera muchas medidas a analizar que podemos visualizar del modelo.
Modelo_Mallas.Final.prf.Tst = h2o.performance(model = Modelo_Mallas.Final, newdata = test)

# Log-pérdida
h2o.logloss(Modelo_Mallas.Final.prf.Tst) #on training set (depende de la corrida)

# Matriz de confusión

# Opción 1
Modelo_Mallas.Final.CMfit.Tst <- h2o.confusionMatrix(Modelo_Mallas.Final, test)
Modelo_Mallas.Final.CMfit.Tst        # Matriz en general
Modelo_Mallas.Final.CMfit.Tst$Error  # Columna de errores
Modelo_Mallas.Final.CMfit.Tst["Totals","Error"] # Tasa de error global

# Opción 2
Modelo_Mallas.Final.prf.Tst@metrics$cm$table
Modelo_Mallas.Final.prf.Tst@metrics$cm$table["Totals","Error"]
Modelo_Mallas.Final.prf.Tst@metrics$cm$table["Totals","Rate"]

# Visualización de las clasificaciones mal realizadas ~~~~~~~~~~~~~~~~~~~~~~~

# Training ******************************************************************

# Seleccionamos los mal clasificados
DNN_Mal.Final.Trn <- MNISTtrain[,y] %>% as.character() != DNN_Prd.Final.Trn[,"predict"] %>% as.vector() %>% as.character()
sum(DNN_Mal.Final.Trn) # ¿Cuántas están mal clasificadas?

# ¿Cuál es el peor clasificado?
MNISTtrain[which(DNN_Mal.Final.Trn),y] %>% table()

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.2,.2))
for (i in 1:9) {
  pinta.num(i,MNISTtrain[which(DNN_Mal.Final.Trn),],TRUE,
            DNN_Prd.Final.Trn[which(DNN_Mal.Final.Trn),"predict"] %>% as.vector())
}
par(mfrow = c(1,1))

# Test **********************************************************************

# Seleccionamos los mal clasificados
DNN_Mal.Final.Tst <- MNISTtest[,y] %>% as.character() != DNN_Prd.Final.Tst[,"predict"] %>% as.vector() %>% as.character()
sum(DNN_Mal.Final.Tst) # ¿Cuántas están mal clasificadas?

# ¿Cuál es el peor clasificado?
MNISTtest[which(DNN_Mal.Final.Tst),y] %>% table()

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.2,.2))
for (i in 1:9) {
  pinta.num(i,MNISTtest[which(DNN_Mal.Final.Tst),],TRUE,
            DNN_Prd.Final.Tst[which(DNN_Mal.Final.Tst),"predict"] %>% as.vector())
}
par(mfrow = c(1,1))

# Gráficos PRO TEST **************************************************

par(mfrow = c(3,3),mai = c(.2,.2,.5,.2))
faux <- function(x){which(MNISTtest[which(DNN_Mal.Final.Tst),]$C785==x)[3]}
NUM <- sapply(1:9,faux)
count=0
for (i in NUM) {
  pinta.num(i,MNISTtest[which(DNN_Mal.Final.Tst),],TRUE,
            DNN_Prd.Final.Tst[which(DNN_Mal.Final.Tst),"predict"] %>% as.vector())
  count=count+1
  if(count==3){par(mai = c(.2,.2,.2,.2)) }
}
mtext("Mal Asignados", side = 3, line = -2.75, outer = TRUE,cex=2)
par(mfrow = c(1,1))

# Mostramos 9 números.
# Gráficos
faux <- function(x){which(MNISTtest$C785==x)[3]}
NUM <- sapply(1:9,faux)
par(mfrow = c(3,3),mai = c(.2,.2,.5,.2))
count=0
for (i in NUM) {
  pinta.num(i,MNISTtest,TRUE,
            DNN_Prd.Final.Tst[,"predict"] %>% as.vector())
  count=count+1
  if(count==3){par(mai = c(.2,.2,.2,.2)) }
}
mtext("Bien Asignados", side = 3, line = -2.75, outer = TRUE,cex=2)
par(mfrow = c(1,1))


# Cross-Validation ********************************************************
Modelo_Mallas.Final@model$cross_validation_metrics_summary
Modelo_Mallas.Final@model$cross_validation_metrics_summary$mean[6]  #logloss
Modelo_Mallas.Final@model$cross_validation_metrics_summary$mean[4]  #overall error
(Modelo_Mallas.Final@model$cross_validation_metrics_summary$mean[3] %>% as.numeric)*100 %>% round(2)  #error cv 


# __ Guardamos predicciones validate --------------------------------------

DNN_Prd.Final.Val = h2o.predict(Modelo_Mallas.Final, newdata=as.h2o(MNISTvalidate))

# Gráficos
par(mfrow = c(3,3),mai = c(.2,.2,.2,.2))
for (i in sample(1:nrow(MNISTvalidate),size = 9,replace = FALSE)) {
  pinta.num(i,MNISTvalidate,FALSE,
            DNN_Prd.Final.Val[,"predict"] %>% as.vector())
}
par(mfrow = c(1,1))

DNN_Prd.Final.Val.csv <- DNN_Prd.Final.Val[,"predict"] %>% as.data.frame()
# write.csv(x = DNN_Prd.Final.Val.csv,file = "Proyecto_Alarcon_DNN_pred.csv",row.names = FALSE)

# _ Malla 2 ---------------------------------------------------------------

# Vamos a construír la malla basándonos en los resultados y parámetros anteriores.  
# La idea es seleccionar modelos aleatorios con una estructura alrededor de la de los
# anteriores.

# Hiper-parámetros de los modelos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(891065)
# Capas y nodos del modelo
hidden_options2 <- replicate(n = 100,simplify = FALSE,
                             expr = {
                               # Cantidad de capas
                               capas <- sample(x = 2:5,size = 1,replace = TRUE,prob = c(0.1,0.3,0.3,0.3))
                               # Número de nodos en cada capa
                               sample(c(200,300,500,1000,2500),size = capas,replace = TRUE,prob = c(0.25,0.25,0.3,0.1,0.1))
                             })
# Número de épocas
(epoch_options2 = c(10,25,40,50,75)) # params de épocas
# Parámetro de regularización.
(l1_options2    = c(0,1e-10,1e-5,1e-2)) #2 params de regularización norma l1
# Guardamos los hypera parámetros
(hyper_params2 <- list(hidden = hidden_options2, l1 = l1_options2, epochs=epoch_options2))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

search_criteria2 = list(
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Esto así porque son modelos MUY pesados.
  strategy ="RandomDiscrete", # Escoge de los modelos aleatoriamente
  max_models = 10, # A lo más correrá 30 modelos
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  seed=29)

# system.time({
#   DNN_grid2 <- h2o.grid(
#                         algorithm = "deeplearning",
#                         grid_id = "DNN_grid2",
#                         hyper_params = hyper_params2,
#                         search_criteria = search_criteria2,
#                         x = x,
#                         y = y,
#                         activation = "RectifierWithDropout",
#                         distribution = "multinomial",
#                         training_frame = as.h2o(MNISTtrain),
#                         validation_frame = as.h2o(MNISTtest),
#                         input_dropout_ratio = 0.2,
#                         stopping_rounds = 3,
#                         stopping_tolerance = 0.05,
#                         stopping_metric = "misclassification",
#                         nfolds = 4
#              )
#   h2o.saveGrid(grid_id = "DNN_grid2", grid_directory = paste0(getwd(),"/Modelos")) %>% try()
# }) -> DNN_grid2.time

# user  system elapsed 
# 12.30    1.91 8897.00 

# DNN_grid2 <- h2o.loadGrid(grid_path = paste0(getwd(),"/Modelos/DNN_grid2"))

summary(DNN_grid2, show_stack_traces = TRUE)

# __ Valuando la malla ----------------------------------------------------

# Extraemos los modelos
Modelos_Mallas2 <- lapply(DNN_grid2@model_ids, function(id){h2o.getModel(id)})

# Vamos a ver su rendimiento en el training y test.
Modelos_Mallas2.Trn <- list()
Modelos_Mallas2.Tst <- list()
system.time({
  # Performance global
  Modelos_Mallas2.Perf = h2o.getGrid(grid_id = "DNN_grid2", sort_by = "err", decreasing = F)
  # Performance por modelo
  for(i in 1:length(Modelos_Mallas2)){
    Modelos_Mallas2.Trn[[i]] <- h2o.performance(Modelos_Mallas2[[i]], newdata = as.h2o(MNISTtrain))
    Modelos_Mallas2.Tst[[i]] <- h2o.performance(Modelos_Mallas2[[i]], newdata = as.h2o(MNISTtest))
  }
}) -> Modelos_Mallas2.Perf.Time
# Vemos los resultados globales.
Modelos_Mallas2.Perf

# Guardamos los resultados individuales de las estadísticas de interés
Resultados_Mallas2=data.frame()
for(i in 1:length(Modelos_Mallas2)){
  Resultados_Mallas2[i,1]=Modelos_Mallas2[[i]]@model_id
  Resultados_Mallas2[i,2]=round(Modelos_Mallas2[[i]]@model$run_time/60000, digits=2)
  Resultados_Mallas2[i,3]=paste(Modelos_Mallas2[[i]]@allparameters$hidden, collapse="-")
  Resultados_Mallas2[i,4]=round(Modelos_Mallas2[[i]]@allparameters$epochs, digits=2)
  Resultados_Mallas2[i,5]=Modelos_Mallas2[[i]]@allparameters$l1
  Resultados_Mallas2[i,6]=Modelos_Mallas2[[i]]@allparameters$input_dropout_ratio 
  Resultados_Mallas2[i,7]=round(h2o.logloss(Modelos_Mallas2.Trn[[i]]), digits=4)
  Resultados_Mallas2[i,8]=round(h2o.logloss(Modelos_Mallas2.Tst[[i]]), digits=4)
  Resultados_Mallas2[i,9]=round(Modelos_Mallas2.Trn[[i]]@metrics$cm$table$Error[11], digits=4)
  Resultados_Mallas2[i,10]=round(Modelos_Mallas2.Tst[[i]]@metrics$cm$table$Error[11], digits=4)
}
colnames(Resultados_Mallas2)=c("model","RunTimeMins","layers","epochs","l1","InDropoutRat",
                               "loglss_trn","loglss_tst","Err_trn","Err_tst")
Resultados_Mallas2 <- Resultados_Mallas2 %>% arrange(Err_tst)
Resultados_Mallas2

# Graficamos los 9 modelos con test error más bajo
dev.off()
par(mfrow = c(3,3))#, mai = c(.5,.5,.5,.5))
# Plot the scoring history over time for the 5 models
for(i in 1:length(Modelos_Mallas2)){
  if(Modelos_Mallas2[[i]]@model_id%in%Resultados_Mallas2$model[1:9]){
    plot(Modelos_Mallas2[[i]],metric = "classification_error",cex=.7)
    grid()
    abline(h = 0.02,col="red",lty=2)
    text(5,0.025,0.02,col="red")
  }
}

# Gráfico de las métricas de los modelos
dev.off()
#par(mfrow = c(1,1), mai = c(.8,.8,.8,.8))
matplot(1:10, 
        cbind(Resultados_Mallas2[,7], 
              Resultados_Mallas2[,8],
              Resultados_Mallas2[,9], 
              Resultados_Mallas2[,10]
        ),
        pch=19, 
        col=c("lightblue", "orange","blue","red"),
        type="b",ylab="Errors/logloss", xlab="models",
        main="Malla aleatoria",xaxt="n",yaxt="n")
axis(1, at=1:10, labels=readr::parse_number(Resultados_Mallas2$model %>% substr(nchar(Resultados_Mallas2$model)-2,nchar(Resultados_Mallas2$model))),
     tck = 1,lty=2,col="gray")
axis(1, at=1:10, labels=readr::parse_number(Resultados_Mallas2$model %>% substr(nchar(Resultados_Mallas2$model)-2,nchar(Resultados_Mallas2$model))),
     tick = 1)
axis(2, at=0:25/10, labels=0:25/10,tck = 1,lty=2,col="gray")
axis(2, at=0:25/10, labels=0:25/10,tick = 1)
box()
legend("topleft", legend=c("$logloss_{train}$", "$logloss_{test}$",
                           "$Error_{train}$", "$Error_{test}$") %>% sapply(LaTeX), pch=19,
       col=c("lightblue", "orange","blue","red"))

# Nos quedamos con el modelo final
for(i in 1:length(Modelos_Mallas2)){
  if(Modelos_Mallas2[[i]]@model_id==Resultados_Mallas2$model[1]){
    Modelo_Mallas2.Final <- Modelos_Mallas2[[i]]
  }
}

# Estos resultados son peores... nos quedaremos con la malla original.

# _ Resultados Mallas -----------------------------------------------------

# Resultados globales ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Final
Modelos_Mallas.Perf
Resultados_Mallas

# Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Final
Modelo_Mallas.Final.CMfit["Totals","Error"] # Tasa de error global
Modelo_Mallas.Final.CMfit["Totals","Rate"] # Tasa de error global

# Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Final
Modelo_Mallas.Final.CMfit.Tst["Totals","Error"] # Tasa de error global
Modelo_Mallas.Final.CMfit.Tst["Totals","Rate"] # Tasa de error global

# save.image(file = "DNN_Resultados.RData")


# _ LaTeX Mallas ----------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Resultados_Mallas

# Ajuste visual de la tabla
aux <- Resultados_Mallas %>% subset(select=-c(InDropoutRat)) %>%  
  as.data.frame() %>% arrange(Err_tst)
# Errores
aux[,c("Err_trn","Err_tst")] <- 100 * (aux[,c("Err_trn","Err_tst")] %>% apply(2,as.numeric))
colnames(aux)[c(8,9)] <- c("%Err_trn","%Err_tst")
# Loglss
aux[,c("loglss_trn","loglss_tst")] <- (aux[,c("loglss_trn","loglss_tst")] %>% apply(2,function(x){round(100*as.numeric(x),2)}))
colnames(aux)[c(6,7)] <- c("%Loglss_trn","%Loglss_tst")
# Modelos
aux$model <- aux$model %>% extract_numeric()
colnames(aux)[1] <- "Model"

# Seleccionamos solo los primeros 10
aux2 <- aux[1:10,]

# LaTeX
aux2 %>% kabla(title = "Estadísticas y parámetros de los 10 mejores modelos de los 20 obtenidos en la malla ordenados por error de prueba. \\textit{Nota: Todos los parámetros que no estén aquí mencionados fueron tomados por default.}",
               ref = "Resultados_Mallas") %>% copy2clipboard()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modelos_Mallas.Trn[[1]]@metrics$model$name
Modelos_Mallas.Trn[[19]]@metrics$model$name
Modelos_Mallas.Trn[[2]]@metrics$model$name

aux <- data.frame(Modelos_Mallas.Trn[[1]]@metrics$cm$table$Error,
                  Modelos_Mallas.Trn[[19]]@metrics$cm$table$Error,
                  Modelos_Mallas.Trn[[2]]@metrics$cm$table$Error) %>% t()
aux <- cbind(c(8,7,18),aux)
rownames(aux) <- NULL
colnames(aux) <- c("Model",0:9,"Total")
aux[,-1] <- (100*aux[,-1]) %>% round(2)
aux %>% kabla(title = "\\textbf{Porcentaje de error} de los 3 mejores modelos de la malla por clase y total en el conjunto de \\textbf{entrenamiento}.",
              ref = "NumErr_Malla") %>% copy2clipboard()

aux2 <- t(aux)[-c(1,ncol(aux)),]
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(x = 0:9,y = aux2,
        pch=19,cex=1.0, col=c("orange", "red","blue"),
        type="b",ylab="Errors", xlab="Clase",
        main="%Errores de entrenamiento por clase",xaxt="n"
        ,yaxt="n"
)
axis(1, at=0:9, labels=0:9,
     tck = 1,lty=2,col="gray")
axis(1, at=0:9, labels=0:9,
     tick = 1)
axis(2, at=0:14/10, labels=0:14/10,tck = 1,lty=2,col="gray")
axis(2, at=0:14/10, labels=0:14/10,tick = 1)
box()
legend("topleft", legend=paste("Modelo",c(8,7,18)) %>% sapply(LaTeX), pch=19,
       col=c("orange","red","blue"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modelos_Mallas.Tst[[1]]@metrics$model$name
Modelos_Mallas.Tst[[19]]@metrics$model$name
Modelos_Mallas.Tst[[2]]@metrics$model$name

aux <- data.frame(Modelos_Mallas.Tst[[1]]@metrics$cm$table$Error,
                  Modelos_Mallas.Tst[[19]]@metrics$cm$table$Error,
                  Modelos_Mallas.Tst[[2]]@metrics$cm$table$Error) %>% t()
aux <- cbind(c(8,7,18),aux)
rownames(aux) <- NULL
colnames(aux) <- c("Model",0:9,"Total")
aux[,-1] <- (100*aux[,-1]) %>% round(2)
aux %>% kabla(title = "\\textbf{Porcentaje de error} de los 3 mejores modelos de la malla por clase y total en el conjunto de \\textbf{prueba}.",
              ref = "NumErr_Malla2") %>% copy2clipboard()

aux2 <- t(aux)[-c(1,ncol(aux)),]
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(x = 0:9,y = aux2,
        pch=19,cex=1.0, col=c("orange", "red","blue"),
        type="b",ylab="Errors", xlab="Clase",
        main="%Errores de prueba por clase",xaxt="n"
        ,yaxt="n"
)
axis(1, at=0:9, labels=0:9,
     tck = 1,lty=2,col="gray")
axis(1, at=0:9, labels=0:9,
     tick = 1)
axis(2, at=(0:25)*2/10, labels=(0:25)*2/10,tck = 1,lty=2,col="gray")
axis(2, at=(0:25)*2/10, labels=(0:25)*2/10,tick = 1)
box()
legend("topleft", legend=paste("Modelo",c(8,7,18)) %>% sapply(LaTeX), pch=19,
       col=c("orange","red","blue"))

#-------FIN---------------------------------------------------------------------

# Logistic Regression -------------------------------------------------------------
#save.image("ambientecasifinal.RData")
#load("ambientecasifinal.RData")

# Se leen los datos
MNISTtrain <- read.csv("MNISTtrain_40000.csv")
MNISTtrain[,785] <- as.factor(MNISTtrain[,785])
levels(MNISTtrain[,785]) <- c(levels(MNISTtrain[,785]),0,1,2,3,4,5,6,7,8,9)

MNISTtest <- read.csv("MNISTtest_9000.csv")
MNISTtest[,785] <- as.factor(MNISTtest[,785])
levels(MNISTtest[,785]) <- c(levels(MNISTtest[,785]),0,1,2,3,4,5,6,7,8,9)

MNISTvalidate <- read.csv("MNISTvalidate_11000.csv")

# Modelo no regularizado

# Función que devuelve las estadísticas para un vector dado de alphas
# Selecciona al mejor modelo y de él predice los datos de test y train

ajusta.alphas <- function(datostrain, datostest, datosvalida, nfolds = 10, alpha){
  #Primero se hace la variable respuesta de tipo factor
  # datostrain[,785] <- as.factor(c(datostrain[,785],0,1,2,3,4,5,6,7,8,9))[1:length(datostrain[,785])]
  # datostest[,785] <-  as.factor(c(datostest[,785],0,1,2,3,4,5,6,7,8,9))[1:length(datostest[,785])]
  # 
  # Se va a devolver una tabla con todas las estadísticas importantes
  estadisticas <- data.frame(alfa = NA,
                             lambda_minTF = NA,
                             lambda_min = NA,
                             lambda_1se = NA,
                             non_zero = NA,
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
                             test_ac9 = NA)
  cont <- 1
  for(a in alpha){
    print(c("entra con alpha=",a))
    print(Sys.time())
    # a=0.2; datostrain = datostrain = MNISTtrain[1:1000,];
    # datostest = MNISTtrain[2000:3000,]; nfolds=10; cont = 1
    # datostrain = MNISTtrain[1:100,]
    # Se ajusta el modelo con la base train
    fit=cv.glmnet(x = as.matrix(datostrain[,1:784]),
                  y = datostrain[,785],
                  family = "multinomial",
                  alpha = a,
                  nfolds = nfolds,
                  parallel=TRUE)
    
    estadisticas[cont,"alfa"] <- a
    
    # Escogiendo entre lambda_min y lambda_1se
    
    # Se estiman las predicciones con cada lambda y se hacen de tipo factor
    predtest_lambdamin = predict(fit,as.matrix(datostest[,1:784]),s=fit$lambda.min,type="class")
    predtest_lambdamin = as.factor(c(predtest_lambdamin,0,1,2,3,4,5,6,7,8,9))[1:length(predtest_lambdamin)]
    predtest_lambda1se = predict(fit,as.matrix(datostest[,1:784]),s=fit$lambda.1se,type="class")
    predtest_lambda1se = as.factor(c(predtest_lambda1se,0,1,2,3,4,5,6,7,8,9))[1:length(predtest_lambda1se)]
    
    # Se compara cual predicción es mejor
    bien_lambdamin = sum(predtest_lambdamin == datostest[,785])
    bien_lambda1se = sum(predtest_lambda1se == datostest[,785])
    
    lambda_min = ifelse(bien_lambdamin > bien_lambda1se, TRUE, FALSE)
    
    estadisticas[cont,"lambda_minTF"] <- lambda_min
    estadisticas[cont,"lambda_min"] <- fit$lambda.min
    estadisticas[cont,"lambda_1se"] <- fit$lambda.1se
    
    estadisticas[cont,"non_zero"] <- fit$nzero[fit$index[2-lambda_min]]
    
    # error de entrenamiento
    entrena=predict(fit,
                    as.matrix(datostrain[,1:784]),
                    s=fit$lambda[fit$index[2-lambda_min]],
                    type="class")
    entrena = as.factor(c(entrena,0,1,2,3,4,5,6,7,8,9))[1:length(entrena)]
    
    aux <- confusionMatrix(data=entrena, reference = datostrain[,785])
    estadisticas$train_accuracy[cont] <- aux$overall[1]
    
    # error de predicción
    if(lambda_min == TRUE){
      pred= predtest_lambdamin
    } else{pred=predtest_lambda1se}
    
    aux2 <- confusionMatrix(data=pred,
                            reference = datostest[,785])
    
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
    
    
    print(c("sale con alpha=",a))
    beepr::beep(5) 
    Sys.time()
    
    esta_prec = aux2$overall[1]
    
    if(cont == 1){
      predtest = pred
      predvalidados = predict(fit,as.matrix(datosvalida[,1:784]),s=fit$lambda.min,type="class")
      mejormodelo = fit
      mejoralfa = a
    } else if(esta_prec == max(estadisticas$test_accuracy)){
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

# Corriendo 3 veces cada modelo
iteraciones <- list()
informe <- data.frame()
cont <- 4
for (j in 1:3) {
  for (i in c(0,0.2,0.5,0.7)) {
    print(c("iteración:",j))
    print(c("Empieza con el ajuste con alpha = ", i))
    print(Sys.time())
    iteraciones[[cont]] <- ajusta.alphas(datostrain = MNISTtrain,
                                         datostest = MNISTtest,
                                         datosvalida = MNISTvalidate,
                                         nfolds = 10,
                                         alpha = i)
    print("Terminó con el ajuste ahora va guardar los archivos")
    informe <- rbind(informe, iteraciones[[cont]]$estadisticas)
    write.xlsx(informe,
               file = paste("i", j, "a", i, "Informe.xlsx", sep = ""),
               sheetName = "Informe",
               append = FALSE)
    write.xlsx(iteraciones[[cont]]$estadisticas,
               file = paste("i", j, "a", i, ".xlsx", sep = ""),
               sheetName = "Estadisticas",
               append = FALSE)
    write.xlsx(iteraciones[[cont]]$predtest,
               file = paste("i", j, "a", i, ".xlsx", sep = ""),
               sheetName = "PredTest",
               append = TRUE)
    write.xlsx(iteraciones[[cont]]$predval,
               file = paste("i", j, "a", i, ".xlsx", sep = ""),
               sheetName = "PredVal",
               append = TRUE)
    save.image("ambiente.RData")
    print("Terminó de guardar los archivos")
    print(Sys.time())
    cont <- cont + 1
  }
}
beepr::beep(8)

# Como se trabajó en varias computadoras se tiene que juntar los resultados
informes <- rbind(informe,informep)

informes[c(3,6,9),4]

#Tabla para informar los parámetros de los modelos y sus errores
informe_final <- data.frame(Alpha = c(0,0.2,0.5,0.7),
                            lambda = c(mean(informes[c(1,4,7),3]),
                                       mean(informes[c(2,5,8),4]),
                                       mean(informes[c(3,6,9),4]),
                                       mean(informes[c(10),3])),
                            Non_zero = c(mean(informes[c(1,4,7),5]),
                                         mean(informes[c(2,5,8),5]),
                                         mean(informes[c(3,6,9),5]),
                                         mean(informes[c(10),5])),
                            Train_error = c(1-mean(informes[c(1,4,7),6]),
                                            1-mean(informes[c(2,5,8),6]),
                                            1-mean(informes[c(3,6,9),6]),
                                            1-mean(informes[c(10),6])),
                            Test_error = c(1-mean(informes[c(1,4,7),7]),
                                           1-mean(informes[c(2,5,8),7]),
                                           1-mean(informes[c(3,6,9),7]),
                                           1-mean(informes[c(10),7])))
informe_final$lambda <- round(informe_final$lambda*10000,2) #x10^-4
informe_final$Train_error <- round(informe_final$Train_error*100,2) # %
informe_final$Test_error <- round(informe_final$Test_error*100,2) # %

kabla(informe_final,"Media de parámetros de ajuste, errores aparentes y de prueba de cuatro modelos multonmiales ajustados con la base \\ctexttt{MNISTtrain} (B=3 repeticiones) \\label{difmodeloslogreg}")

# Esta tabla es para informar los errores de prueba por clase de cada modelo
errores_test_porclase <- data.frame(Model = NA, cer = NA, uno = NA, dos = NA,
                                    tres = NA, cuatro = NA, cinco = NA, seis = NA,
                                    siete = NA, ocho = NA, nueve = NA,Total = NA)
for (i in 1:4) {#i=1
  errores_test_porclase[i, 1] <- i
  errores_test_porclase[i, 2:11] <- round(100*(1 - apply(informes[c(i, 1+3, i+6),8:17],2,mean)),2)
  errores_test_porclase[i, 12] <- informe_final$Test_error[i]
}

kabla(errores_test_porclase,"Media de los porcentajes de error de los modelos multinomiales por clase y total en el conjunto deprueba (B=3 repeticiones) \\label{errortestclassmodel}")

# Dado que se escoge como mejor modelo el estimado por alpha = 0.2
# se hacen sus gráficas descriptivas

fit = iteracionesp[[2]]$mejormodelo
fit$cvm
par(mfrow=c(1,1))
plot(fit)
par(mfrow=c(5,2))
plot(fit$glmnet.fit)
par(mfrow=c(1,1))
# Error por validación cruzada
assess.glmnet(fit,s=fit$lambda.1se,newx=as.matrix(MNISTtrain[,1:784]),newy=MNISTtrain[,785])$mse #lasso



# Para graficar 9 estimaciones bien hechas y 9 mal hechas

pred_test_LogReg <- read.csv("test_RegLog_pred.csv")
pred_test_LogReg <- pred_test_LogReg[,2]

mal = which(MNISTtest[,785] != pred_test_LogReg)
mal1a9 <- c()
for(i in 1:9){
  mal1a9[i] <- mal[which(MNISTtest[mal,785]==i)[1]]
}

bien = which(MNISTtest[,785] == pred_test_LogReg)
bien1a9 <- c()
for(i in 1:9){
  bien1a9[i] <- bien[which(MNISTtest[bien,785]==i)[1]]
}

# Gráficos de 9 que están bien
par(mfrow = c(3,3),mai = c(.2,.2,.5,.2))
count=0
for (i in bien1a9) {
  pinta.num(registro = i,
            datos = MNISTtest,
            respuesta = TRUE,
            predict = pred_test_LogReg)
  count=count+1
}
mtext("Bien Asignados (LogReg)", side = 3, line = -2.75, outer = TRUE,cex=2)
par(mfrow = c(1,1))

# Gráficos de 1 a 9 que están mal
par(mfrow = c(3,3),mai = c(.2,.2,.5,.2))
count=0
for (i in mal1a9) {
  pinta.num(registro = i,
            datos = MNISTtest,
            respuesta = TRUE,
            predict = pred_test_LogReg)
  count=count+1
}
mtext("Mal Asignados (LogReg)", side = 3, line = -2.75, outer = TRUE,cex=2)
par(mfrow = c(1,1))

#-------FIN---------------------------------------------------------------------

#Random forest-------------------------------------------------------------------------

#Preliminares: Lectura de las bases de datos.

MNISTtest=read.csv("MNISTtest_9000.csv")
MNISTtrain=read.csv("MNISTtrain_40000.csv")
MNISTvalidate=read.csv("MNISTvalidate_11000.csv")

MNISTtrain$C785 = factor(MNISTtrain$C785)

MNISTtest.h2o=as.h2o(MNISTtest)
MNISTtrain.h2o=as.h2o(MNISTtrain)
MNISTvalidate.h2o=as.h2o(MNISTvalidate)

#Guardaremos los errores.
errores=matrix(NA, ncol=8, nrow=11)
rownames(errores)=c("0","1","2","3","4",
                    "5","6","7","8","9","Global")
colnames(errores)=c("Ap mtries=26","Tst mtries=26",
                    "Ap mtries=28","Tst mtries=28",
                    "Ap mtries=30","Tst mtries=30",
                    "Ap mtries=100","Tst mtries=100")

#I. Calibramos mtries. Modelos están ordenados de acuerdo a mtries.
set.seed(1)
train.rf.2=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 26)
train.rf.0=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 28) #sqrt(714)=28
train.rf.3=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 30)
train.rf.1=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 100, max_depth = 10)
train.rf.7=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 714)
#Mejor desempenio con mtries = 28.

#Prueba
set.seed(1)
rf.pred.0=predict(train.rf.0, newdata=MNISTtest.h2o, type="response")
rf.pred.1=predict(train.rf.1, newdata=MNISTtest.h2o, type="response")
rf.pred.2=predict(train.rf.2, newdata=MNISTtest.h2o, type="response")
rf.pred.3=predict(train.rf.3, newdata=MNISTtest.h2o, type="response")

conf.rf.0=table(MNISTtest$C785, as.vector(rf.pred.0[,1]))
conf.rf.1=table(MNISTtest$C785, as.vector(rf.pred.1[,1]))
conf.rf.2=table(MNISTtest$C785, as.vector(rf.pred.2[,1]))
conf.rf.3=table(MNISTtest$C785, as.vector(rf.pred.3[,1]))

#Errores aparentes.
for(i in 1:11){
  errores[i,1]=h2o.confusionMatrix(train.rf.2)[i,11]
  errores[i,3]=h2o.confusionMatrix(train.rf.0)[i,11]
  errores[i,5]=h2o.confusionMatrix(train.rf.3)[i,11]
  errores[i,7]=h2o.confusionMatrix(train.rf.1)[i,11]
}
#Test errors.
for(i in 1:10){
  errores[i,2]=1-conf.rf.2[i,i]/sum(conf.rf.2[i,])
  errores[i,4]=1-conf.rf.0[i,i]/sum(conf.rf.0[i,])
  errores[i,6]=1-conf.rf.3[i,i]/sum(conf.rf.3[i,])
  errores[i,8]=1-conf.rf.1[i,i]/sum(conf.rf.1[i,])
}
#Global test errors.
errores[11,2]=1-sum(diag(conf.rf.2))/sum(conf.rf.2) 
errores[11,4]=1-sum(diag(conf.rf.0))/sum(conf.rf.0) 
errores[11,6]=1-sum(diag(conf.rf.3))/sum(conf.rf.3)
errores[11,8]=1-sum(diag(conf.rf.1))/sum(conf.rf.1)

kabla(round(t(errores*100), digits=2)) #Exportamos a latex.

#II. Calibramos max_depth:

err.max_d=matrix(NA, ncol=6, nrow=11)
rownames(err.max_d)=c("0","1","2","3","4",
                      "5","6","7","8","9","Global")
colnames(err.max_d)=c("Ap m_depth=10","Tst m_depth=10",
                      "Ap m_depth=20","Tst m_depth=20",   #20=default
                      "Ap m_depth=21","Tst m_depth=21")

#Entrenamiento.
train.rf.4=h2o.randomForest(y=785,training_frame = MNISTtrain.h2o, ntrees=500, mtries = 28, max_depth=10) 
train.rf.5=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 28, max_depth=21) 

#Prueba
set.seed(1)
rf.pred.4=predict(train.rf.4, newdata=MNISTtest.h2o, type="response")
rf.pred.5=predict(train.rf.5, newdata=MNISTtest.h2o, type="response")

conf.rf.4=table(MNISTtest$C785, as.vector(rf.pred.4[,1]))
conf.rf.5=table(MNISTtest$C785, as.vector(rf.pred.5[,1]))

#Errores aparentes.
for(i in 1:11){
  err.max_d[i,1]=h2o.confusionMatrix(train.rf.4)[i,11] #max_depth=10
  err.max_d[i,3]=h2o.confusionMatrix(train.rf.0)[i,11] #max_depth=20
  err.max_d[i,5]=h2o.confusionMatrix(train.rf.5)[i,11] #max_depth=21
}
#Test errors.
for(i in 1:10){
  err.max_d[i,2]=1-conf.rf.4[i,i]/sum(conf.rf.4[i,]) #max_depth=10
  err.max_d[i,4]=1-conf.rf.0[i,i]/sum(conf.rf.0[i,]) #max_depth=20
  err.max_d[i,6]=1-conf.rf.0[i,i]/sum(conf.rf.5[i,]) #max_depth=21
}
#Global test errors.
err.max_d[11,2]=1-sum(diag(conf.rf.4))/sum(conf.rf.4) 
err.max_d[11,4]=1-sum(diag(conf.rf.0))/sum(conf.rf.0) 
err.max_d[11,6]=1-sum(diag(conf.rf.5))/sum(conf.rf.5) 

df_maxdepth=kabla(data.frame(round(t(err.max_d[,c(5,3,1)]*100), digits=2))) #Aparentes max_depth
df_maxdepth=kabla(data.frame(round(t(err.max_d[,c(6,4,2)]*100), digits=2))) #Test max_depth

round(err.max_d*100, digits=3)

#III. Aumentamos el número de árboles:
#Guardaremos los errores.
err_ntrees=matrix(NA, ncol=6, nrow=11)
rownames(err_ntrees)=c("0","1","2","3","4",
                       "5","6","7","8","9","Global")
colnames(err_ntrees)=c("Ap ntrees=500","Tst ntrees=500",
                       "Ap ntrees=800","Tst ntrees=800",
                       "Ap ntrees=1000","Tst ntrees=1000")

train.rf.6=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=800, mtries = 28) 
train.rf.8=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=1000, mtries = 28) 

#Prueba
set.seed(1)
rf.pred.6=predict(train.rf.6, newdata=MNISTtest.h2o, type="response")
rf.pred.8=predict(train.rf.6, newdata=MNISTtest.h2o, type="response")

conf.rf.6=table(MNISTtest$C785, as.vector(rf.pred.6[,1]))
conf.rf.8=table(MNISTtest$C785, as.vector(rf.pred.8[,1]))

#Errores aparentes.
for(i in 1:11){
  err_ntrees[i,1]=h2o.confusionMatrix(train.rf.0)[i,11]
  err_ntrees[i,3]=h2o.confusionMatrix(train.rf.6)[i,11]
  err_ntrees[i,5]=h2o.confusionMatrix(train.rf.8)[i,11]
}
#Test errors.
for(i in 1:10){
  err_ntrees[i,2]=1-conf.rf.0[i,i]/sum(conf.rf.0[i,])
  err_ntrees[i,4]=1-conf.rf.6[i,i]/sum(conf.rf.6[i,])
  err_ntrees[i,6]=1-conf.rf.6[i,i]/sum(conf.rf.8[i,])
}
#Global test errors.
err_ntrees[11,2]=1-sum(diag(conf.rf.0))/sum(conf.rf.0) 
err_ntrees[11,4]=1-sum(diag(conf.rf.6))/sum(conf.rf.6) 
err_ntrees[11,6]=1-sum(diag(conf.rf.8))/sum(conf.rf.8) 

round(err_ntrees*100, digits=3)

#IV. Cambiamos de libreria a randomforest().

train.rf.9=randomForest(C785~.,data=MNISTtrain,ntree=500,mtry=28)
train.rf.10=randomForest(C785~.,data=MNISTtrain,ntree=1000,mtry=28, strata=MNISTtrain$C785)
train.rf.11=randomForest(C785~.,data=MNISTtrain,ntree=800,mtry=28)

round(train.rf.10$confusion[,11]*100, digits=2)
1-sum(diag(train.rf.10$confusion))/sum(train.rf.10$confusion) #0.03355815 error global.

rf.pred.9.train=predict(train.rf.9, newdata=MNISTtrain, type="response")
rf.pred.10.train=predict(train.rf.10, newdata=MNISTtrain, type="response")
rf.pred.11.train=predict(train.rf.11, newdata=MNISTtrain, type="response")

#conf.rf.9.train=table(MNISTtrain$C785, rf.pred.9.train) #ajusta todo bien.
#1-sum(diag(conf.rf.9.train))/sum(conf.rf.9.train) #da cero

kabla(round(t(train.rf.9$confusion[,11])*100, digits=2))

train.rf.9$err.rate[1000]                       #oob errors 0.033625
round(train.rf.10$err.rate[1000]*100, digits=2) #oob errors 0.03345
train.rf.11$err.rate[800] #oob errors 

rf.pred.9.test=predict(train.rf.9, newdata=MNISTtest, type="response")
rf.pred.10.test=predict(train.rf.10, newdata=MNISTtest, type="response")
rf.pred.11.test=predict(train.rf.11, newdata=MNISTtest, type="response")

conf.rf.9.test=table(MNISTtest$C785, rf.pred.9.test)
conf.rf.10.test=table(MNISTtest$C785, rf.pred.10.test)
conf.rf.11.test=table(MNISTtest$C785, rf.pred.11.test)

err.tst=matrix(NA, ncol=3, nrow=11)
rownames(err.tst)=c("0","1","2","3","4",
                    "5","6","7","8","9","Global")
colnames(err.tst)=c("Test ntree=1000", "Test ntree=1000 strata", "Test ntree=800")
err.tst
kabla(cbind(3.36,3.14,3.65))
#Test errors.
for(i in 1:10){
  err.tst[i,1]=1-conf.rf.9.test[i,i]/sum(conf.rf.9.test[i,])
  err.tst[i,2]=1-conf.rf.10.test[i,i]/sum(conf.rf.10.test[i,])
  err.tst[i,3]=1-conf.rf.11.test[i,i]/sum(conf.rf.11.test[i,])
}
#Global test errors.
err.tst[11,1]=1-sum(diag(conf.rf.9.test))/sum(conf.rf.9.test)   #0.03144444
err.tst[11,2]=1-sum(diag(conf.rf.10.test))/sum(conf.rf.10.test) #0.03233333
err.tst[11,3]=1-sum(diag(conf.rf.11.test))/sum(conf.rf.11.test) #0.03211111

kabla(round(t(err.tst)*100,digits=2))

#Validate
validate=predict(train.rf.9, newdata=MNISTvalidate, type="response")
validate=as.vector(validate)
write.csv(validate, file = "Proyecto_Alarcon_RF_pred.csv" )

#Validacion cruzada. 

set.seed(2021)
k=3
err.cv=numeric(k)
folds=createFolds(MNISTtrain$C785, k = k, list = TRUE, returnTrain = F)
for(i in 1:k){
  rf_classifier = randomForest(C785~.,data=MNISTtrain[-folds[[i]],],ntree= 1000,mtry= 28)
  rf.pred.9.cv = predict(rf_classifier,MNISTtrain[folds[[i]],])
  conf.rf.9.cv = table(MNISTtrain[folds[[i]],]$C785,rf.pred.9.cv)
  err.cv[i]=1-sum(diag(conf.rf.9.cv))/sum(conf.rf.9.cv) 
}
round(mean(err.cv)*100,digits=2) #Error CV: 3.65

#-------FIN---------------------------------------------------------------------






# Comparacion de predicciones de los datos de validación ------------------

pred_val_LogReg <- read.csv("Proyecto_Alarcon_RegLog_pred.csv")
pred_val_DNN <- read.csv("Proyecto_Alarcon_DNN_pred.csv")
pred_val_RF <- read.csv("Proyecto_Alarcon_RF_pred.csv")

pred_val_LogReg <- as.factor(pred_val_LogReg[,2])
pred_val_DNN <- as.factor(pred_val_DNN[,1])
pred_val_RF <- as.factor(pred_val_RF[,2])


validate_prediction <- data.frame(RF = pred_val_RF,
                                  LogReg = pred_val_LogReg,
                                  DNN = pred_val_DNN)
names(validate_prediction) <- c("RF","LogReg","DNN")
validate_prediction$Dif <- ifelse(validate_prediction$RF == validate_prediction$LogReg,
                                  FALSE,
                                  ifelse(validate_prediction$RF == validate_prediction$DNN,
                                         FALSE,
                                         ifelse(validate_prediction$LogReg == validate_prediction$DNN,
                                                FALSE,
                                                TRUE)))
#Hay 54 observaciones que los tres métodos los estiman diferente
sum(validate_prediction$Dif)

diferentes <- which(validate_prediction$Dif == TRUE)

par(mfrow = c(4,4),mai = c(.2,.2,.5,.2))
for (i in 1:16) {
  pinta.num(registro = diferentes[i],
            datos = MNISTvalidate,
            valida = TRUE,
            ajustes = c(validate_prediction$RF[diferentes[i]],
                        validate_prediction$LogReg[diferentes[i]],
                        validate_prediction$DNN[diferentes[i]]))
  
}
mtext("Predicciones diferentes por cada modelo", side = 3, line = -2.75, outer = TRUE,cex=2)
par(mfrow = c(1,1))

table(pred_val_DNN,pred_val_RF)
DNNvsLogReg <- confusionMatrix(data = pred_val_DNN,reference = pred_val_LogReg)
DNNvsLogReg$table

DNNvsRF <- confusionMatrix(data = pred_val_DNN,reference = pred_val_RF)
aux2$table[1,]

confusiondoble <- data.frame(DNN = 0:9)
for (i in 1:20) {
  confusiondoble[,i+1] <- rep(NA,10)
  if(i<=10){
    colnames(confusiondoble)[i+1] <- paste("LogReg",i-1,sep = "")
  } else{
    colnames(confusiondoble)[i+1] <- paste("RF",i-1,sep = "")
    }
}
for (i in 1:10) {
  confusiondoble[,i+1] <- DNNvsLogReg$table[,i]
  confusiondoble[,i+11] <- DNNvsRF$table[,i]
}
head1 <- c(1,10)
names(head1) <- c(" ","LogReg")
head2 <- rep(1,11)
names(head2) <- c("DNN",0:9)
colnames(confusiondoble) <- NULL
kabla(confusiondoble[,1:11],"Estimaciones por clase, de los datos \\ctexttt{MNISTvalidate} del modelo DNN vs los modelos LogReg y RF de \\label{difmodeloslogreg}") %>% 
  add_header_above(head2) %>% 
  add_header_above(head1) %>% clipr::write_clip()

head1 <- c(1,10)
names(head1) <- c(" ","RF")
head2 <- rep(1,11)
names(head2) <- c("DNN",0:9)
colnames(confusiondoble) <- NULL
kabla(confusiondoble[,c(1,12:21)],"Estimaciones por clase, de los datos \\ctexttt{MNISTvalidate} del modelo DNN vs los modelos LogReg y RF de \\label{difmodeloslogreg}") %>% 
  add_header_above(head2) %>% 
  add_header_above(head1) %>% clipr::write_clip()

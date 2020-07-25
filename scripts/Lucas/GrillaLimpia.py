from __future__ import print_function
import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import scipy.stats as stats
import math
import sys
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib import pyplot as plt


class Grilla:

    def __init__(self, filas, columnas, defaultValue = 0.0):
        #self.deltaFila = deltaFila
        #self.deltaColumna = deltaColumna
        self.values = np.zeros((filas,columnas))
        self.labels = np.zeros((filas, columnas), dtype=int)
        self.indexes = np.zeros((filas,columnas), dtype=int)
        self.indexToCoordinate = []
        index = 0
        for i in range(self.indexes.shape[0]):
            for j in range(self.indexes.shape[1]):
                self.indexes[i,j] = index
                self.indexToCoordinate.append((i,j))
                index = index + 1
                self.values[i,j] = defaultValue


    def getCantidadFilas(self):
        return self.values.shape[0]

    def getCantidadColumnas(self):
        return self.values.shape[1]

    def setValue(self, fila, columna, value):
        self.values[fila, columna] = value

    def getValue(self, fila,columna):
        return self.values[fila,columna]

    def setLabel(self, fila,columna,label):
        self.labels[fila,columna] = label

    def getIndex(self, fila, columna):
        return self.indexes[fila,columna]

    def getCantidadIndexes(self):
        return self.indexes.shape[0]*self.indexes.shape[1]

    def getLabel(self, fila, columna):
        return self.labels[fila, columna]

    def getKernelRow(self, fila, columna, gridKernel):
        row = np.zeros(self.getCantidadIndexes())
        totalWeight = 0.0
        for i in range(gridKernel.getCantidadWeights()):
            gridWeight = gridKernel.getWeight(i)
            filaDesp = fila + gridWeight.getDispFila()
            colDesp = columna + gridWeight.getDispColumna()
            weight = gridWeight.getWeight()
            if filaDesp >= 0 and filaDesp < self.getCantidadFilas() and colDesp >= 0 and colDesp < self.getCantidadColumnas():
                row[self.getIndex(filaDesp,colDesp)] = weight
                totalWeight = totalWeight + weight
        return row/totalWeight           

    def getKernelValue(self, fila, columna, gridKernel):
        accum = 0.0
        totalWeight = 0.0
        for i in range(gridKernel.getCantidadWeights()):
            gridWeight = gridKernel.getWeight(i)
            filaDesp = fila + gridWeight.getDispFila()
            colDesp = columna + gridWeight.getDispColumna()
            if filaDesp >= 0 and filaDesp < self.getCantidadFilas() and colDesp >= 0 and colDesp < self.getCantidadColumnas():
                weight = gridWeight.getWeight()
                totalWeight = totalWeight + weight 
                accum = accum + weight*self.getValue(filaDesp, colDesp)
        return accum/totalWeight

    def getValues(self):
        return self.values

    def getLabels(self):
        return self.labels

    def getCoordinateFromIndex(self, index):
        return self.indexToCoordinate[index]

    def getLabelFromIndex(self, index):
        coord = self.getCoordinateFromIndex(index)
        return self.getLabel(coord[0], coord[1])

    def loadX(self,x):
        for i in range(len(x)):
            coord = self.getCoordinateFromIndex(i)
            self.setValue(coord[0], coord[1], x[i])

    def loadLabelsHorizontal(self,labels):
        index = 0
        for i in range(self.values.shape[0]):
            for j in range(self.values.shape[1]):
                self.setLabel(i,j, labels[index])
                index = index + 1

    def getLabelsHorizontal(self):
        labelsRet = []
        for i in range(self.values.shape[0]):
            for j in range(self.values.shape[1]):
                labelsRet.append(self.getLabel(i,j))
        return labelsRet


    # Por ahora devuelve A e Y, despues habria que hacer que devuelva mas cosas.
    # Tipo las covarianzas y demas.
    def constructSystem(self, coord_kernels):
        columnasA = len(self.indexToCoordinate)
        filasA = len(coord_kernels)
        A = np.zeros((filasA, columnasA))
        y = np.zeros(filasA)
        for i in range(len(coord_kernels)):
            coord_kernel = coord_kernels[i]
            fila = coord_kernel["fila"]
            columna = coord_kernel["columna"]
            kernel = coord_kernel["kernel"]
            A[i,:] = self.getKernelRow(fila, columna, kernel)
            #row = self.getKernelRow(fila, columna, kernel)
            #for j in range(columnasA):
                #A[i,j] = row[j]
            y[i] = self.getKernelValue(fila, columna, kernel)
        return A,y            

    # Miramos los cuatro vecinos y promediamos diferencias con elementos del mismo tipo.
    def penalizacionDiferenciaMismoTipoLocal(self, i, j):
        #print "Analizando (i,j) ", i, j
        mismoTipo = 0
        labelCentral = self.getLabel(i,j)
        #print "label central ", labelCentral
        valueCentral = self.getValue(i,j)
        accum = 0.0
        if i-1 >= 0 and self.getLabel(i-1,j) == labelCentral:
            accum = accum + np.absolute(self.getValue(i-1,j) - valueCentral)
            mismoTipo = mismoTipo + 1
        if i+1 < self.getCantidadFilas() and self.getLabel(i+1,j) == labelCentral:
            accum = accum + np.absolute(self.getValue(i+1,j) - valueCentral)
            mismoTipo = mismoTipo + 1
        if j-1 >= 0 and self.getLabel(i,j-1) == labelCentral:
            accum = accum + np.absolute(self.getValue(i,j-1) - valueCentral)
            mismoTipo = mismoTipo + 1
        if j+1 < self.getCantidadColumnas() and self.getLabel(i,j+1) == labelCentral:
            accum = accum + np.absolute(self.getValue(i,j+1) - valueCentral)
            mismoTipo = mismoTipo + 1
        if mismoTipo == 0:
            return 0.0
        else:
            return accum / float(mismoTipo)

    # Miramos los cuatro vecinos y promediamos diferencias con elementos del mismo tipo.
    def penalizacionDiferenciaMismoTipoGlobal(self):
        accum = 0.0
        for i in range(self.getCantidadFilas()):
            for j in range(self.getCantidadColumnas()):
                accum = accum + self.penalizacionDiferenciaMismoTipoLocal(i,j)
        return accum


    # Miramos vecinos segun el kernel y promediamos diferencias con elementos del mismo tipo.
    def penalizacionDiferenciaMismoTipoLocalKernel(self, i, j, kernel):
        mismoTipo = 0
        labelCentral = self.getLabel(i,j)
        valueCentral = self.getValue(i,j)
        accum = 0.0
        totalWeight = 0.0
        for index in range(kernel.getCantidadWeights()):
            gridWeight = kernel.getWeight(index)
            filaDesp = i + gridWeight.getDispFila()
            colDesp = j + gridWeight.getDispColumna()
            if filaDesp >= 0 and filaDesp < self.getCantidadFilas() and colDesp >= 0 and colDesp < self.getCantidadColumnas():
                if self.getLabel(filaDesp, colDesp) == labelCentral:
                    weight = gridWeight.getWeight()
                    totalWeight = totalWeight + weight 
                    accum = accum + weight*np.absolute(self.getValue(filaDesp, colDesp)-valueCentral)
        if totalWeight == 0.0:
            return 0.0
        else:
            return accum/totalWeight

    def penalizacionDiferenciaMismoTipoGlobalKernel(self, kernel):
        accum = 0.0
        for i in range(self.getCantidadFilas()):
            for j in range(self.getCantidadColumnas()):
                accum = accum + self.penalizacionDiferenciaMismoTipoLocalKernel(i,j,kernel)
        return accum

    def getTychonovH(self, kernel):
        filas = self.getCantidadFilas()
        columnas = self.getCantidadColumnas()
        cantidadVars = filas*columnas
        H = np.zeros((cantidadVars, cantidadVars))
        labelsHor = self.getLabelsHorizontal()
        for i in range(filas):
            for j in range(columnas):
                totalWeight = 0.0
                indexHorCentral = self.getIndex(i,j)
                labelCentral = self.getLabelFromIndex(indexHorCentral)
                for index in range(kernel.getCantidadWeights()):
                    gridWeight = kernel.getWeight(index)
                    filaDesp = i + gridWeight.getDispFila()
                    colDesp = j + gridWeight.getDispColumna()
                    if filaDesp >= 0 and filaDesp < self.getCantidadFilas() and colDesp >= 0 and colDesp < self.getCantidadColumnas():
                        if self.getLabel(filaDesp, colDesp) == labelCentral:
                            indexHorDesp = self.getIndex(filaDesp, colDesp)
                            weight = gridWeight.getWeight()
                            H[indexHorCentral,indexHorDesp] = -weight
                            totalWeight = totalWeight + weight
                # Renormalizamos.
                if totalWeight > 0.0:
                    H[indexHorCentral, :] = H[indexHorCentral,:]/totalWeight
                    H[indexHorCentral, indexHorCentral] = 1.0
        return H


    def pcolor(self):
        values = self.getValues()
        m = np.ma.masked_where(np.isnan(values),values)
        plt.pcolor(np.arange(values.shape[0]),np.arange(values.shape[1]),np.array(m), cmap='jet', shading = 'interp')
        plt.colorbar()
        plt.xlabel("Columna")
        plt.ylabel("Fila")
        plt.show() 


    def pcolorKernelContinuo(self, kernel):
        smoothed = np.zeros(self.values.shape)
        for i in range(self.values.shape[0]):
            for j in range(self.values.shape[1]):
                smoothed[i,j] = self.getKernelValue(i,j,kernel)
        plt.pcolor(np.arange(smoothed.shape[0]),np.arange(smoothed.shape[1]),np.array(smoothed), cmap='jet', shading = 'interp')
        plt.colorbar()
        plt.xlabel("Columna")
        plt.ylabel("Fila")
        plt.show() 

    def pcolorCoordKernels(self, coordKernels):
        newValues = np.zeros(self.values.shape)
        values = self.values
        for i in range(newValues.shape[0]):
            for j in range(newValues.shape[1]):
                newValues[i,j] = np.nan
        for coordKernel in coordKernels:
            fila = coordKernel["fila"]
            col = coordKernel["columna"]
            newValues[fila,col] = self.getKernelValue(fila, col, coordKernel["kernel"])
        m = np.ma.masked_where(np.isnan(newValues),newValues)
        plt.pcolor(np.arange(values.shape[0]),np.arange(values.shape[1]),np.array(m), cmap='jet', shading = 'interp')
        plt.colorbar()
        plt.xlabel("Columna")
        plt.ylabel("Fila")
        plt.show() 




######################################### KERNELS #########################

# Para construir los Kernels.
class GridWeight:

    def __init__(self,dispFila, dispColumna, weight):
        self.dispFila = dispFila
        self.dispColumna = dispColumna
        self.weight = weight

    def getDispFila(self):
        return self.dispFila

    def getDispColumna(self):
        return self.dispColumna

    def getWeight(self):
        return self.weight

    def setWeight(self, weight):
        self.weight = weight

    def __str__(self):
        return "(" + str(self.dispFila) + "," + str(self.dispColumna) + ", " + str(self.weight) + ")"




class GridKernel:

    def __init__(self):
        self.gridWeights = []

    def addWeight(self, gridWeight):
        self.gridWeights.append(gridWeight)

    def getCantidadWeights(self):
        return len(self.gridWeights)

    def getWeight(self, i):
        return self.gridWeights[i]

    def renormalize(self):
        accum = 0.0
        for gridWeight in self.gridWeights:
            accum = accum + gridWeight.getWeight()
        for gridWeight in self.gridWeights:
            gridWeight.setWeight(gridWeight.getWeight() / accum)

    def __str__(self):
        s = ""
        for gridWeight in self.gridWeights:
            s = s + str(gridWeights) + "\n"
        return s

# Kernel de cruz que promedia los bordes de conectividad 4 y el centro, con mas peso.
class CrossKernel(GridKernel):

    def __init__(self):
        GridKernel.__init__(self)
        self.addWeight(GridWeight(0,0, 4.0))
        self.addWeight(GridWeight(0,-1,1.0))
        self.addWeight(GridWeight(0,+1,1.0))
        self.addWeight(GridWeight(-1,0,1.0))
        self.addWeight(GridWeight(+1,0,1.0))
        self.renormalize()

# Kernel de bloque 3x3 centrado en el punto.
class BlockKernel(GridKernel):

    def __init__(self):
        GridKernel.__init__(self)
        self.addWeight(GridWeight(0,0, 1.0))
        self.addWeight(GridWeight(0,-1,1.0))
        self.addWeight(GridWeight(0,+1,1.0))
        self.addWeight(GridWeight(-1,0,1.0))
        self.addWeight(GridWeight(+1,0,1.0))
        self.addWeight(GridWeight(+1,+1,1.0))
        self.addWeight(GridWeight(+1,-1,1.0))
        self.addWeight(GridWeight(-1,+1,1.0))
        self.addWeight(GridWeight(-1,-1,1.0))
        self.renormalize()


# Kernel de bloque 3x3 no uniforme.
class BlockNonUniformKernel(GridKernel):

    def __init__(self):
        GridKernel.__init__(self)
        self.addWeight(GridWeight(0,0, 8.0))
        self.addWeight(GridWeight(0,-1,2.0))
        self.addWeight(GridWeight(0,+1,2.0))
        self.addWeight(GridWeight(-1,0,2.0))
        self.addWeight(GridWeight(+1,0,2.0))
        self.addWeight(GridWeight(+1,+1,1.0))
        self.addWeight(GridWeight(+1,-1,1.0))
        self.addWeight(GridWeight(-1,+1,1.0))
        self.addWeight(GridWeight(-1,-1,1.0))
        self.renormalize()



## Arma al azar una serie de kernels coordinados en puntos.
def kernelizar(grilla, KernelClass, cantidad):
    filas = grilla.getCantidadFilas()
    columnas = grilla.getCantidadColumnas()
    coordKernels = []
    for i in range(cantidad):
        kernel = KernelClass()
        fila = random.randint(0, filas)
        columna = random.randint(0,columnas)
        coordKernels.append(constructCoord_Kernel(fila,columna, kernel))
    return coordKernels



def constructCoord_Kernel(fila, columna, kernel):
    return {"fila": fila, "columna": columna, "kernel": kernel}

####################################### AUXILIARES ########################

# Devuelve la Median Absolute Deviation ajustado por normalidad.
def mad(values):
    return 1.4826*np.median(np.absolute(np.array(values) - np.median(values)))

# Arma una grilla equiespaciada logaritmicamente.
def grillaLogaritmica(base, tope, cantidad):
    baseLog = math.log(base)
    topeLog = math.log(tope)
    grillaLog = np.linspace(baseLog,topeLog, cantidad)
    return np.exp(grillaLog)    


####################################### SOLVERS ###########################

# Se deberia hacer un GCV con esto para elegir el omega.
# https://en.wikipedia.org/wiki/Tikhonov_regularization
def fastTychonovProcedure(filas,columnas, A, y, labelsHor, omega, kernel):
    if omega == 0.0:
        x = linalg.lstsq(A,y)[0]
    else:
        grilla = Grilla(filas, columnas)
        print ("Labels Hor: ", labelsHor)
        grilla.loadLabelsHorizontal(labelsHor)
        H = omega*grilla.getTychonovH(kernel)
        x = linalg.inv(A.T.dot(A) + H.T.dot(H)).dot(A.T).dot(y)
    elementos = []
    cantLabels = np.max(labelsHor) + 1
    means = np.zeros(cantLabels)
    variances = np.zeros(cantLabels)
    for i in range(cantLabels):
        elementos.append([])
    for i in range(len(x)):
        elementos[labelsHor[i]].append(x[i])
    for i in range(len(elementos)):
        if len(elementos[i]) > 1:
            variances[i] = np.var(elementos[i])
        else:
            variances[i] = 0.0 #imputacion caprichosa...
        if len(elementos[i]) > 0:
            means[i] = np.mean(elementos[i])
        else:
            means[i] = 0.0 #imputacion caprichosa...
    return x, means, variances
# Anda sorprendentemente bien en bandas...


# No se esta recalculando el H... NO ES NECESARIO, NO TIENE QUE VER CON ELIPSES.
def CV_Tychonov(filas,columnas, A, y, labelsHor, omegas, kernel):
    cvValues = np.zeros(len(omegas))
    cantElipses = A.shape[0]
    grilla = Grilla(filas, columnas)
    print "Labels Hor: ", labelsHor
    grilla.loadLabelsHorizontal(labelsHor)
    Hbase = grilla.getTychonovH(kernel)
    y = y.reshape((len(y), 1))
    for i in range(len(omegas)):
        omega = omegas[i]
        print "Ensayando omega ", omega
        accum = 0.0
        for fila in range(cantElipses):
            Ared = np.delete(A, [fila], axis=0)
            yred = np.delete(y, [fila], axis=0)
            H = omega*Hbase
            x_red = linalg.inv(Ared.T.dot(Ared) + H.T.dot(H)).dot(Ared.T).dot(yred)
            accum = accum + linalg.norm(A[fila,:].dot(x_red) - y[fila] )
        print "Value obtenido: ", accum
        cvValues[i] = accum
    bestIndex = np.argmin(cvValues)
    bestOmega = omegas[bestIndex]
    return fastTychonovProcedure(filas, columnas, A,y,labelsHor, bestOmega, kernel)    


# Generalized Cross Validation. Presupone ser mas rapido.
# Mezcla de pagina 111 del libro "Parameter Estimation and Inverse Problems"
# (2ed) de Aster et al. y pagina 117 del mismo libro.
def GCV_Tychonov(filas,columnas, A, y, labelsHor, omegas, kernel):
    cvValues = np.zeros(len(omegas))
    cantElipses = A.shape[0]
    grilla = Grilla(filas, columnas)
    print "Labels Hor: ", labelsHor
    grilla.loadLabelsHorizontal(labelsHor)
    Hbase = grilla.getTychonovH(kernel)
    y = y.reshape((len(y), 1))
    for i in range(len(omegas)):
        omega = omegas[i]
        Asharp = linalg.inv(A.T.dot(A) + (omega**2)*Hbase.T.dot(Hbase)).dot(A.T)
        H = omega*Hbase
        x = linalg.inv(A.T.dot(A) + H.T.dot(H)).dot(A.T).dot(y)
        I = np.eye(cantElipses)
        value = cantElipses*(linalg.norm(A.dot(x) - y))**2 / (np.trace(I - A.dot(Asharp)))**2
        cvValues[i] = value
        print ("GCV Value obtenido: ", value)
    bestIndex = np.argmin(cvValues)
    bestOmega = omegas[bestIndex]
    return fastTychonovProcedure(filas, columnas, A,y,labelsHor, bestOmega, kernel)    

# Es muy rapido y parece andar muy bien.

############# TESTING #############################################

# Escenario de "banda extendida"
# Usando el metodo de Tychonov con CV.
def test24():
    # Armamos la grilla.
    grilla = Grilla(20,20)
    filas = grilla.getCantidadFilas()
    cols = grilla.getCantidadColumnas()
    for i in range(0,filas):
        for j in range(0,cols):
            grilla.setValue(i,j,norm.rvs(loc=100, scale=4)) # scale es varianza o desvio estandar? Es aparentemente desvio estandar.
            grilla.setLabel(i,j,0) # Agua
    # Fabricamos una cierta cantidad de "islas", que son pixels en principio.
    for i in range(8,14):
        for j in range(0, cols):
            grilla.setValue(i,j,norm.rvs(loc=160, scale=9)) # scale es varianza o desvio estandar? Es aparentemente desvio estandar.
            grilla.setLabel(i,j,1) # Tierra
    print ("Pcolor grilla original")
    grilla.pcolor()
    # Obtenemos los kernels y armamos el sistema asociado            
    coordKernels = kernelizar(grilla, BlockKernel, 150)
    #coordKernels = kernelizar(grilla, BlockKernel, 50) # Da igual bastante bien con 50.
    #coordKernels = kernelizar(grilla, BlockNonUniformKernel, 150)
    A,y = grilla.constructSystem(coordKernels)
    # Contaminamos y por presencia de ruido instrumental.
    #lamb = 1.0 # Esto es VARIANZA, no desvio estandar. Si esto vale cero se degenera en matriz singular.
    #varNoise = 1.0
    varNoise = 0.0
    y = y + norm.rvs(loc=0, scale=np.sqrt(varNoise), size=len(y))
    # Preparamos todo para resolver por medias.
    labelsHor = grilla.getLabelsHorizontal()
    #x, means, variances = experimentalIterativeTwoStepsProcedureTolerance(filas,cols, A, y, labelsHor, iterations = 1000,method="Classic", verbose=True, toleranceX = 1e-3)
    omegas = grillaLogaritmica(1e-6, 1e3, 10)
    #omega = 0.0
    kernelDisc = BlockKernel()
    #x, means, variances = tychonovProcedure(filas,cols, A, y, labelsHor, omega, kernelDisc)
    #x, means, variances= fastTychonovProcedure(filas,cols, A, y, labelsHor, omega, kernelDisc)
    #x, means, variances= CV_Tychonov(filas,cols, A, y, labelsHor, omegas, kernelDisc)
    x, means, variances= GCV_Tychonov(filas,cols, A, y, labelsHor, omegas, kernelDisc)
    #sys.exit(1)
    grillaNueva = Grilla(filas,cols)
    grillaNueva.loadX(x)
    grillaNueva.loadLabelsHorizontal(labelsHor)
    #sys.exit(1)
    print "Medias: ", means
    print "Varianzas: ", variances
    print "Pcolor grilla obtenida"
    grillaNueva.pcolor()
    # Diferencias.
    print "Pcolor de diferencias absolutas"
    diferencia = grillaNueva.getValues()- grilla.getValues()
    plt.pcolor(diferencia, cmap='jet', shading = 'interp')
    plt.colorbar()
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.show()
    # Diferencias relativas.
    for i in range(diferencia.shape[0]):
        for j in range(diferencia.shape[1]):
            diferencia[i,j] = np.absolute(diferencia[i,j] / grilla.getValue(i,j))
    print "Pcolor de diferencias relativas"
    plt.pcolor(diferencia, cmap='jet', shading = 'interp')
    plt.colorbar()
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.show()
    print "Error promedio: ", np.mean(diferencia)
# Anduvo bastante bien.



# Escenario de "damero extendido"
# Usando el metodo de Tychonov con CV.
def test25():
    # Armamos la grilla.
    grilla = Grilla(20,20)
    filas = grilla.getCantidadFilas()
    cols = grilla.getCantidadColumnas()
    for i in range(0,filas):
        for j in range(0,cols):
            grilla.setValue(i,j,norm.rvs(loc=100, scale=4)) # scale es varianza o desvio estandar? Es aparentemente desvio estandar.
            grilla.setLabel(i,j,0) # Agua
    # Fabricamos una cierta cantidad de "islas", que son pixels en principio.
    for i in range(0,filas):
        for j in range(0, cols):
            if (i+j) % 2 == 0:
                grilla.setValue(i,j,norm.rvs(loc=160, scale=9)) # scale es varianza o desvio estandar? Es aparentemente desvio estandar.
                grilla.setLabel(i,j,1) # Tierra
    #kernelDisc = CrossKernel()
    kernelDisc = BlockKernel()
    disc = grilla.penalizacionDiferenciaMismoTipoGlobalKernel(kernelDisc)
    print "Penalizacion grilla: ", disc
    print "Pcolor grilla original"
    grilla.pcolor()
    #sys.exit(1)
    # Obtenemos los kernels y armamos el sistema asociado            
    #coordKernels = kernelizar(grilla, BlockKernel, 150)
    #coordKernels = kernelizar(grilla, BlockKernel, 50) # Da igual bastante bien con 50.
    coordKernels = kernelizar(grilla, BlockKernel, 100)
    #coordKernels = kernelizar(grilla, BlockKernel, 300)
    #coordKernels = kernelizar(grilla, BlockNonUniformKernel, 150)
    A,y = grilla.constructSystem(coordKernels)
    # Contaminamos y por presencia de ruido instrumental.
    #lamb = 1.0 # Esto es VARIANZA, no desvio estandar. Si esto vale cero se degenera en matriz singular.
    #varNoise = 1.0
    varNoise = 0.0
    y = y + norm.rvs(loc=0, scale=np.sqrt(varNoise), size=len(y))
    # Preparamos todo para resolver por medias.
    labelsHor = grilla.getLabelsHorizontal()
    # Omega se deberia ajustar por CV.
    omegas = grillaLogaritmica(1e-6, 1e3, 10)
    kernelDisc = BlockKernel()
    #x, means, variances = tychonovProcedure(filas,cols, A, y, labelsHor, omega, kernelDisc)
    #x, means, variances= fastTychonovProcedure(filas,cols, A, y, labelsHor, omega, kernelDisc)
    #x, means, variances= CV_Tychonov(filas,cols, A, y, labelsHor, omegas, kernelDisc)
    x, means, variances= GCV_Tychonov(filas,cols, A, y, labelsHor, omegas, kernelDisc)
    grillaNueva = Grilla(filas,cols)
    grillaNueva.loadX(x)
    grillaNueva.loadLabelsHorizontal(labelsHor)
    #sys.exit(1)
    print "Medias: ", means
    print "Varianzas: ", variances
    print "Pcolor grilla obtenida"
    grillaNueva.pcolor()
    # Diferencias.
    print "Pcolor de diferencias absolutas"
    diferencia = grillaNueva.getValues()- grilla.getValues()
    plt.pcolor(diferencia, cmap='jet', shading = 'interp')
    plt.colorbar()
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.show()
    # Diferencias relativas.
    for i in range(diferencia.shape[0]):
        for j in range(diferencia.shape[1]):
            diferencia[i,j] = np.absolute(diferencia[i,j] / grilla.getValue(i,j))
    print "Pcolor de diferencias relativas"
    plt.pcolor(diferencia, cmap='jet', shading = 'interp')
    plt.colorbar()
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.show()
    print "Error promedio: ", np.mean(diferencia)
# Anda mas o menos bien el CV.
# El GCV anda tambien bastante bien, y muy rapido.
# Con 300 bloques es muy buena la reobtencion.


#test24() # (banda) CV y GCV andan muy bien.
test25() # (damero) Algo peor que con banda, pero respetable.



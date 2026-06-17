import numpy as np
from Functions.Utils_RTLO import *

class RTLO:
    def __init__(self, nI,nR,nO,ηS=[0.1,0.1,0.1], τ=10,mode='past'):
        np.random.seed(42)
        self.k = 1
        self.j = nI-1
        self.t = np.array([])
        self.start = 0
        self.ref = None
        self.act = 'tanh'
        self.flw = mode
        self.n = -1
        self.nI, self.nR, self.nO = nI, nR, nO

        self.ηS = np.array(ηS)
        self.τ = τ
        self.ρ = 0.003

        self.x = np.zeros(nI)
        self.hP, self.hU, self.hL = [0.1*np.ones(nR) for i in range(3)]
        self.hP2, self.hU2, self.hL2 = [0.1*np.ones(nR) for i in range(3)]
        self.uP, self.uP2 =  [np.zeros(nR) for i in range(2)]
        self.pS = np.zeros((self.nR, self.nR))
        self.qS = np.zeros((self.nR, self.nI))
        
        self.wI = XavierUniform([nR, nI],sd=42)
        self.wR = XavierUniform([nR, nR],sd=41)
        self.wO = XavierUniform([nO, nR],sd=40)
        self.BS = XavierUniform([nR, nO],sd=39)
    
        self.yP, self.yR, self.yL, self.yU = [np.array([]) for i in range(4)]
        self.yP_hist = np.zeros(self.nI)
        self.εY, self.εM, self.εR, self.εE, self.eP, self.eR, self.ΣW = [0 for i in range(7)]
        self.εM_hist,self.εR_hist, self.eR_hist, self.eP_hist = [np.array([]) for i in range(4)]
        self.wR_hist = []
        self.wI_hist = []
        self.wO_hist = []

        self.rR = 1e-9
        self.rP = 1e-10
        self.rL = 1e-11
        self.rU = 1e-12
        self.rRsum = 0
        self.rulR, self.rulP, self.rulL, self.rulU = [np.array([]) for i in range(4)]


    def PredSingle(self,x):

        u = np.dot(self.wR, self.hP) + np.dot(self.wI, x)
        h = self.hP + (-self.hP + Activation(u,self.act))/self.τ
        y = np.dot(self.wO, h)

        return y

    def fit(self,xP,yR):
        if self.flw != 'past': self.n = 0
        
        η1,η2,η3 = self.ηS      
        uP = self.wR @ self.hP + self.wI @ xP
        hP = self.hP*(1-1/self.τ) + Activation(uP,self.act)/self.τ
        yP = self.wO @ hP
        eS = yR-yP

        self.pS = np.outer(dActivation(self.uP,self.act),self.hP)/self.τ + (1-1/self.τ)*self.pS
        self.qS = np.outer(dActivation(self.uP,self.act),self.x)/self.τ + (1-1/self.τ)*self.qS

        δOS = η1*np.outer(eS,hP)
        δRS = η2*np.outer((self.BS@eS),np.ones(self.nR))*self.pS
        δIS = η3*np.outer(np.dot(self.BS, eS),np.ones(self.nI))*self.qS

        self.wI = self.wI + δIS
        self.wR = self.wR + δRS
        self.wO = self.wO + δOS

        self.wR_hist.append(self.wR.flatten())
        self.wI_hist.append(self.wI.flatten())
        self.wO_hist.append(self.wO.flatten())

        self.hP = hP
        self.hP2 = hP
        self.hL = hP
        self.hU = hP
        self.uP = uP
        self.x = xP

        self.k = self.k+1
        #self.ηS = self.ηS/(1 + self.decay*self.k
    
    def Predict(self, x):
        if self.flw != 'past': self.n = 0
        xP = x.copy()
       
        
        hP = self.hP2.copy()
        #print('P S',hP[:5])
        uP = (self.wR @ hP) + (self.wI @ xP)
        hP = hP*(1-1/self.τ) + Activation(uP,self.act)/self.τ
        yP = (self.wO @ hP)[self.n]
        self.hP2 = hP
        return yP
    
    def PredictIntr(self,xP,xL,xU,ep,show=False):
        if self.flw != 'past': self.n = 0
        wR,wI,wO = self.wR,self.wI,self.wO
        hP,hU,hL = self.hP2.copy(),self.hU2.copy(),self.hL2.copy()

        if show:
            print('hP antes:', hP[-5:])

        wRU, wRL = np.maximum((1+ep)*wR, wR/(1+ep)), np.minimum((1+ep)*wR, wR/(1+ep))
        wIU, wIL = np.maximum((1+ep)*wI, wI/(1+ep)), np.minimum((1+ep)*wI, wI/(1+ep))
        wOU, wOL = np.maximum((1+ep)*wO, wO/(1+ep)), np.minimum((1+ep)*wO, wO/(1+ep))
        
        uP = ( wR @ hP) + ( wI @ xP)
        uL = (wRL @ hL) + (wIL @ xL)
        uU = (wRU @ hU) + (wIU @ xU)

        
        uU, uL = np.maximum(uU,uL), np.minimum(uU,uL)
        
        hP = hP*(1-1/self.τ) + Activation(uP,self.act)/self.τ
        hL = hL*(1-1/self.τ) + Activation(uL,self.act)/self.τ
        hU = hU*(1-1/self.τ) + Activation(uU,self.act)/self.τ
        hU, hL = np.maximum(hU,hL), np.minimum(hU,hL)

        yP = ( wO @ hP)
        yL = (wOL @ hL)
        yU = (wOU @ hU)
        yU, yL = np.maximum(yU, yL), np.minimum(yU, yL)

        yP = yP[self.n]
        yL = yL[self.n]
        yU = yU[self.n]

        if show:
            print('uP:',uP[-5:])
            print('hP depois:',hP[-5:])
            print('xP antes:', xP[-3:],'yP:',yP)
            print('-----------------')

        

        self.hP2 = hP
        self.hL2 = hL
        self.hU2 = hU

        return np.array([yL,yP,yU])

    def Restore(self):
        self.hP2 = self.hP
        self.hL2 = self.hP
        self.hU2 = self.hP

    def ReturnParameters(self):

        return [self.wR,self.wI,self.wO,self.pS,self.qS,self.hP,self.hP2,self.hL2,self.hU2,self.x]
    
    def ReceiveParameters(self,vec):
        self.wR,self.wI,self.wO,self.pS,self.qS,self.hP,self.hP2,self.hL2,self.hU2,self.x = vec
        



 
        

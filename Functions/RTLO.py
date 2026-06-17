import numpy as np

class RTLO:
    def __init__(self, n_in, n_rec, n_out,N1,N2,N3, tau_m=1,decay=1e-5):
        np.random.seed(42)
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.xI = np.zeros(n_in)
        self.tau_m = tau_m
        self.n_spl = 1
        self.ni = np.array([N1,N2,N3])
        self.ni0 = np.array([N1,N2,N3])
        self.T = 1
        self.mErr = 0
        self.dw_in = 0
        self.dw_rec = 0
        self.dw_out = 0
        self.k = 0
        self.decay = decay
        self.ht = 0
        self.wit = 0
        self.wrt = 0
        self.wot = 0
    
        #self.w_in = np.zeros([n_rec, n_in])+1e-6
        #self.w_rec = np.zeros([n_rec, n_rec])+1e-6
        #self.w_out = np.zeros([n_out, n_rec])+1e-6
        #self.b = np.zeros([n_rec, n_out])+1e-6

        self.w_in = self.xavier_uniform([n_rec, n_in])
        self.w_rec = self.xavier_uniform([n_rec, n_rec])
        self.w_out = self.xavier_uniform([n_out, n_rec])
        self.b = np.random.randn(n_rec, n_out)/n_out**0.5

        self.u = 0
        self.uI = 0
        self.hI = np.zeros(self.n_rec)
        self.hF = np.zeros(self.n_rec)

        self.p = np.zeros((self.n_rec, self.n_rec))
        self.q = np.zeros((self.n_rec, self.n_in))

        #self.b = np.random.randn(n_rec, n_out)/n_out**0.5


    def adapt(self, x, yR,tip=1,parameters=False):
        self.ni = self.ni0/(1 + self.decay*self.k)
        N1, N2, N3 = self.ni[0], self.ni[1], self.ni[2]

        for tt in range(self.n_rec):
            self.u= np.dot(self.w_rec, self.hI) + np.dot(self.w_in, x)
            self.hF = self.hI + (-self.hI + self.phi(self.u))/self.tau_m
            yP = np.dot(self.w_out, self.hF)
            err = yR - yP  # readout error
            self.p = np.outer(self.dphi(self.uI),self.hI)/self.tau_m + (1-1/self.tau_m)*self.p

            self.q = np.outer(self.dphi(self.uI),self.xI)/self.tau_m + (1-1/self.tau_m)*self.q

            #self.dw_out = (eta1*np.outer(err, self.hF))
            #self.dw_rec = eta2*np.outer(np.dot(self.b, err),np.ones(self.n_rec))*self.p
            #self.dw_in = eta3*np.outer(np.dot(self.b, err),np.ones(self.n_in))*self.q
            
            self.dw_out = (self.dw_out*(self.T-1)/self.T)+((N1*np.outer(err, self.hF))/self.T)*tip
            self.dw_rec = (self.dw_rec*(self.T-1)/self.T)+(N2*np.outer(np.dot(self.b, err),np.ones(self.n_rec))*self.p/self.T)*tip
            self.dw_in = (self.dw_in*(self.T-1)/self.T)+(N3*np.outer(np.dot(self.b, err),np.ones(self.n_in))*self.q/self.T)*tip

            self.w_out = self.w_out + self.dw_out
            self.w_rec = self.w_rec + self.dw_rec
            self.w_in = self.w_in + self.dw_in

            if parameters:
                print('x:',x,'y:',yR)
                print('u:',self.u)
                print('hI:',self.hI)
                print('hF:',self.hF)
                print('yP:',yP)
                print('err:',err)
                print('p:', self.p)
                print('q:', self.q)
                print("self.dw_out:", self.dw_out)
                print("self.dw_rec:", self.dw_rec)
                print("self.dw_in:", self.dw_in)
                print('w_out:',self.w_out)
                print('w_rec:',self.w_rec)
                print('w_in:',self.w_in)

        self.ht = self.hF
        self.ht2 = self.hF
        self.wit = self.w_in
        self.wrt = self.w_rec
        self.wot = self.w_out
        self.hI = self.hF
        self.xI = x
        self.uI = self.u
        self.T = self.T+1  
        self.k = self.k+1 
        return
    
    def predict(self,x):
        #print(self.ht)
        #print('.......')
        u = np.dot(self.wrt, self.ht) + np.dot(self.wit, x)
        h = self.ht + (-self.ht + self.phi(u))/self.tau_m
        y = np.dot(self.wot, h)
        self.ht = h
        return y
    
    def predict2(self,x):
        #print(self.ht)
        #print('.......')
        u = np.dot(self.wrt, self.ht2) + np.dot(self.wit, x)
        h = self.ht2 + (-self.ht2 + self.phi(u))/self.tau_m
        y = np.dot(self.wot, h)
        self.ht2 = h
        return y
    
    def restore(self):
        self.ht = self.hF
        self.ht2 = self.hF

    def restore_ni(self):
        self.ni = self.ni0
        self.k = 0
        self.T = 1
    
    def restore_ni2(self):
        self.ni = self.ni0
        self.k = 0
        #self.T = 1

    def phi(self,x):
        return np.tanh(x)
    
    def dphi(self,x):
        return 1/np.cosh(10*np.tanh(x/10))**2  # the tanh prevents oveflow
    
    def xavier_uniform(self,shape):
        np.random.seed(42)
        n_in, n_out = shape
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(-limit, limit, size=shape)


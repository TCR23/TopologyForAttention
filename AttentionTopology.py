# Methods for computing measures used in "Topological insights into the neural basis of flexible behavior"

import numypy as np
import scipy
from scipy.io import loadmat
from scipy.integrate import quad_vec
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve


# x -> birth time of the persistence diagram for a homology group
# y -> death time of the persistence diagram for a homology group
Total_Persistence = lambda x,y: np.sum(y-x)

def SC_Extraction(filename,sessionIdx):
    decoder = loadmat(filename)['decoder']
    dprime_c = loadmat(filename)['dprimeCt']
    dprime_u = loadmat(filename)['dprimeUt']
    attinside = loadmat(filename)['attINside']
    # get spike count
    sc_prev = decoder[0,sessionIdx][0].T
    attlocAll = decoder[0,sessionIdx][11]
    # separate attend in and attend out data
    attin_idx = np.nonzero(attlocAll == attinside)[0]
    attout_idx = np.nonzero(attlocAll!= attinside)[0]
    attin_sc = sc_prev[:,attin_idx]
    attout_sc = sc_prev[:,attout_idx]
    return attin_sc,attout_sc

def PersistenceHomologyCalcs(sc):
    VR = VietorisRipsPersistence(metric = "precomputed",homology_dimensions= [0,1,2])
    BC = BettiCurve()
    distance_mat = 1-np.corrcoef(sc)
    diag = VR.fit_transform(distance_mat[None,:,:])
    BettiData = BC.fit_transform(diag)
    return diag,BettiData,Total_Persistence(diag[0,:,:2])

def AdjacencyNormalization(mat):
    u,v = np.linalg.eig(mat)
    newmat = mat/(1+np.max(np.abs(u)))
    return newmat

def ModalControllability(mat):
    modalvect = np.zeros((mat.shape[0],))
    u,vl,vr = scipy.linalg.eig(mat,left = True)
    eigval = u
    for i in range(eigval.shape[0]):
        for j in range(eigval.shape[0]):
            modalvect[i] += ((1-eigval[j]**2)*(vl[jj,ii]**2))
    return modalvect

def ControlGramianCalc(mat,timeHorizon):
    B = np.eye(mat.shape[0])
    tmpmat = np.matmul(B,B.T)
    f = lambda tau: np.matmul(np.matmul(scipy.linalg.expm(mat*tau),tmpmat),scipy.linalg.expm(mat.T*tau))
    y,err = quad_vec(f,0,timeHorizon)
    determinant = scipy.linalg.det(y)
    eigvals,eigvects = scipy.linalg.eig(y)
    return (y,determinant,eigvals,eigvects)

def DistributionOfCorrelations(corr_mat):
    idx = np.triu_indices(n = corr_mat.shape[0],k=1,m = corr_mat.shape[1])
    offdiag = corr_mat[idx]
    meanval = np.mean(offdiag)
    varval = np.var(offdiag)
    return (offdiag,meanval,varval)

def ExampleCalculations(filename):
    sessionIdx = 1
    # get the spike count data
    attin_sc,attout_sc = SC_Extraction(filename,sessionIdx)
    corr_mat_attIn = np.corrcoef(attin_sc)
    corr_mat_attOut = np.corrcoef(attout_sc)
    # compute the homology
    persDiagram_attIn,BettiData_attIn,totalPersistence_attIn = PersistenceHomologyCalcs(attin_sc)
    peakBetti_attIn = np.max(BettiData_attIn[0,:,:],axis = 1)
    persDiagram_attOut,BettiData_attOut,totalPersistence_attOut = PersistenceHomologyCalcs(attout_sc)
    peakBetti_attOut = np.max(BettiData_attOut[0,:,:],axis = 1)
    # compute the controllability measures
    adjmat_in = np.copy(corr_mat_attIn)
    adjmat_in[np.diag_indices(adjmat_in.shape[0])] = 0 # zero out the diagonal to remove self connections
    adjmat_out = np.copy(corr_mat_attOut)
    adjmat_out[np.diag_indices(adjmat_out.shape[0])] = 0
    W_attin,det_attin,eigvals_attin,eigvects_attin = ControlGramianCalc(AdjacencyNormalization(adjmat_in),1)
    W_attout,det_attout,eigvals_attout,eigvects_attout = ControlGramianCalc(AdjacencyNormalization(adjmat_out),1)
    # average control
    avgctrl_attin = np.trace(W_attin)
    avgctrl_attout = np.trace(W_attout)
    # modal control
    avg_modalctrl_attin = np.mean(ModalControllability(AdjacencyNormalization(adjmat_in)))
    avg_modalctrl_attout = np.mean(ModalControllability(AdjacencyNormalization(adjmat_out)))
    # compute the average noise correlation
    offdiag_attin,avg_rsc_attin,var_attin = DistributionOfCorrelations(corr_mat_attIn)
    offdiag_attout,avg_rsc_attout,var_attout = DistributionOfCorrelations(corr_mat_attOut)
    return 0

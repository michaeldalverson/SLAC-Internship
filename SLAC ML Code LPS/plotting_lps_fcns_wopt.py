#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:01:27 2018

@author: cemma
"""

def plot_lps_vs_prediction_lucretia(lps,predicted_lps,x,y):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage    
    ns = int(lps.shape[1]+lps.shape[1]*(0.5*(2*np.random.rand(1,1))-1));        
    fig, (ax, ax2) = plt.subplots(1,2)
    rotated_image = ndimage.rotate(np.transpose(lps[ns,:]),90)
    ax.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    #ax.imshow(rotated_image)
    ax.set_aspect(0.4)
    ax.set_xlabel('z [$\mu$ m]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax.set_xticks([-50,-25,0,25,50])
    rotated_image = ndimage.rotate(np.transpose(predicted_lps[ns,:]),90)
    ax2.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    #ax2.imshow(rotated_image)    
    ax2.set_xlabel('z [$\mu$ m]')
    ax2.set_xticks([-50,-25,0,25,50])
    ax2.set_aspect(0.4)
    plt.show()
    
def plot_cur_prediction_vs_lucretia(tvector,curprof,predicted_curprof,Imax):
    import matplotlib.pyplot as plt
    import numpy as np
    charge = 2e-9;    
    ns = int(curprof[:,0].shape+curprof[:,0].shape*(0.5*(2*np.random.rand(1,1))-1))
    conv = charge/np.trapz(curprof[ns,:],x=tvector[ns,:]*1e-6/3e8)*1e-3;
    plt.figure()
    plt.plot(tvector[ns,:],curprof[ns,:]*conv,label='Simulation');
    plt.plot(tvector[ns,:],predicted_curprof[ns,:]*conv,'r--',label='Neural Net');
    plt.xlabel('z [um]')
    plt.ylabel('I [kA]')
    plt.legend()
    plt.xlim([-100,100])
    plt.show()
    
def plot_lps_and_current_lucretia(lps,x,y,tvector,curprof,Imax):
    import matplotlib.pyplot as plt
    import numpy as np  
    from scipy import ndimage
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    fig, (ax, ax2) = plt.subplots(1,2,figsize=(10,3))
    rotated_image = ndimage.rotate(lps[ns,:],90)
    ax.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),aspect = "auto")
    ax.set_xlabel('z [$\mu$m]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax2.plot(tvector[ns,:],curprof[ns,:]*Imax)
    ax2.set_xlabel('s [$\mu$m]')
    ax2.set_ylabel('Current [kA]')
    ax2.set_aspect("auto")
    plt.show()

def plot_inverse_model_vs_data(X,X_pred):
    import matplotlib.pyplot as plt
    plt.subplot(231)
    plt.plot(X[:,0]-X_pred[:,0])
    plt.ylabel('L1S phase difference')

    plt.subplot(232)
    plt.plot(X[:,1]-X_pred[:,1])
    plt.ylabel('L2 phase difference')
            
    plt.subplot(233)
    plt.plot(X[:,2]-X_pred[:,2])
    plt.ylabel('L1S amp difference')    

    plt.subplot(234)
    plt.plot(X[:,3]-X_pred[:,3])
    plt.ylabel('L2 amp difference')   

    plt.subplot(235)
    plt.plot(X[:,4]-X_pred[:,4])
    plt.ylabel('BC1 peak current diff.')
    
    plt.subplot(236)
    plt.plot(X[:,6])
    plt.ylabel('L1X phase [deg]')  
    
    plt.tight_layout()
    plt.show()


def preset_plot_lps_vs_prediction_lucretia(lps,predicted_lps,x,y,ns):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage    
    #ns = int(lps.shape[1]+lps.shape[1]*(0.5*(2*np.random.rand(1,1))-1));      
    ns = int(ns)
    fig, (ax, ax2) = plt.subplots(1,2)
    rotated_image = ndimage.rotate(np.transpose(lps[ns,:]),90)
    ax.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    #ax.imshow(rotated_image)
    ax.set_aspect(0.4)
    ax.set_xlabel('z [$\mu$ m]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax.set_xticks([-50,-25,0,25,50])
    rotated_image = ndimage.rotate(np.transpose(predicted_lps[ns,:]),90)
    ax2.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    #ax2.imshow(rotated_image)    
    ax2.set_xlabel('z [$\mu$ m]')
    ax2.set_xticks([-50,-25,0,25,50])
    ax2.set_aspect(0.4)
    plt.show()

def grab_ns_plot_lps_vs_prediction_lucretia(lps,predicted_lps,x,y):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage    
    ns = int(lps.shape[1]+lps.shape[1]*(0.5*(2*np.random.rand(1,1))-1));        
    fig, (ax, ax2) = plt.subplots(1,2)
    rotated_image = ndimage.rotate(np.transpose(lps[ns,:]),90)
    ax.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    #ax.imshow(rotated_image)
    ax.set_aspect(0.4)
    ax.set_xlabel('z [$\mu$ m]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax.set_xticks([-50,-25,0,25,50])
    rotated_image = ndimage.rotate(np.transpose(predicted_lps[ns,:]),90)
    ax2.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    #ax2.imshow(rotated_image)    
    ax2.set_xlabel('z [$\mu$ m]')
    ax2.set_xticks([-50,-25,0,25,50])
    ax2.set_aspect(0.4)
    plt.show()
    return ns

#Essentially is plot_vs_prediction_lucretia with a predetermine ns value
def optimization_plot_lps_vs_prediction_lucretia(lps,predicted_lps,x,y,ns):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage    
    #ns = int(lps.shape[1]+lps.shape[1]*(0.5*(2*np.random.rand(1,1))-1));      
    ns = int(ns)
    fig, (ax, ax2) = plt.subplots(1,2)
    rotated_image = ndimage.rotate(np.transpose(lps[ns,:]),90)
    ax.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    #ax.imshow(rotated_image)
    ax.set_aspect(0.4)
    ax.set_xlabel('z [$\mu$ m]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax.set_xticks([-50,-25,0,25,50])
    rotated_image = ndimage.rotate(np.transpose(predicted_lps[0]),90)
    ax2.imshow(rotated_image,extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    #ax2.imshow(rotated_image)    
    ax2.set_xlabel('z [$\mu$ m]')
    ax2.set_xticks([-50,-25,0,25,50])
    ax2.set_aspect(0.4)
    plt.show()
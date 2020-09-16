#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt as wt
from scipy import signal
import scipy.io as matlab
from os import scandir, getcwd
import os.path
import scipy.io.wavfile as waves
import csv
from ipywidgets import * 
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[12]:


def norm(arr):
    arr = arr - min(arr)
    arr = arr/max(arr)
    return arr

def fft_signal(arr, Fs):
    L = len(arr)
    
    Y = np.fft.fft(arr);
    P2 = abs(Y/L);
    P1 = P2[1:int(L/2+1)];
    P1[1:len(P1)-1] = 2*P1[1:len(P1)-1];
    f = Fs*(np.arange(0,(L/2)))/L;

    return f, P1

def ls(ruta = getcwd()):
    return [arch.name for arch in scandir(ruta)] #if arch.is_file()]

def save_csv(data, ruta):
    with open(ruta, 'w') as save_File:
        writer = csv.writer(save_File)
        writer.writerows(data)
        
def open_csv(ruta):
    senal = []
    try:
        df = pd.read_csv(ruta, sep=',',header=None)
        if(df[1][0] == "Channel 1 (V)"):
            senal = np.array(df[1][1:]).astype(np.float32)
        else:
            senal = df[1][:16000]                    
    except:
        data = []
        with open(ruta, newline='') as File:  
            reader = csv.reader(File)
            for row in reader:
                data.append(row)

            data = data[6:]
            data = [data[i][1] for i in range(len(data))]
            senal = np.array(data).astype(np.float32)
    
    return senal


# # FFT-Mismo paciente con diferente calzado

# # Melissa 

# In[43]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Melissa/Normal/WU/WU_M_N1_Each A.csv" #Modificar el modo WU o FB en las rutas y el numero en la N puede ser N1,N2,N3 refiriendose al numero de muestra.
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[44]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Melissa/Tenis/WU/WU_M_T1_Each A.csv" #Modificar el modo WU o FB en las rutas y el numero en la N puede ser T1,T2,T3 refiriendose al numero de muestra.
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio1 = (df.values[:,1].astype(float))


# In[45]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Melissa/Zapatillas/WU/WU_M_Z1_Each A.csv" #Modificar el modo WU o FB en las rutas y el numero en la N puede ser Z1,Z2,Z3 refiriendose al numero de muestra.
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio2 = (df.values[:,1].astype(float))


# In[46]:


import plotly.graph_objects as go
import numpy as np
f,p =fft_signal(audio,Fs)
f1,p1 =fft_signal(audio1,Fs)
f2,p2 =fft_signal(audio2,Fs)

fig = go.Figure()
fig.add_trace(go.Scatter(x=f, y=p, name='Melissa Normal'))
fig.add_trace(go.Scatter(x=f1, y=p1, name='Melissa Tenis'))
fig.add_trace(go.Scatter(x=f2, y=p2, name='Melissa Zapatillas'))
fig.update_layout(title="Muestra 1 Melissa Warm Up Descalza_Tenis_Zapatillas", hoverlabel_align = 'auto') #Cambiar el titulo dependiendo el numero de muestra(puede ser muestra 1, 2, 3, etc.) y el modo ya sea Warm Up o Fat Burn.
fig.update_layout(xaxis={'title':'Frecuencia'}, yaxis={'title':'Amplitud'})
fig.show()


# # Hermosillo

# In[22]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Hermosillo/Normal/WU/WU_LH_N1_Each A.csv" #Modificar el modo WU o FB en las rutas y el numero en la N puede ser N1,N2,N3 refiriendose al numero de muestra.
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[23]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Hermosillo/Tenis/WU/WU_LH_T1_Each A.csv" #Modificar el modo WU o FB en las rutas y el numero en la T puede ser T1,T2,T3 refiriendose al numero de muestra.
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[24]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Hermosillo/Zapatos/WU/WU_LH_Z1_Each A.csv" #Modificar el modo WU o FB en las rutas y el numero en la N puede ser Z1,Z2,Z3 refiriendose al numero de muestra.
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[26]:


f,p =fft_signal(audio,Fs)
f1,p1 =fft_signal(audio1,Fs)
f2,p2 =fft_signal(audio2,Fs)

fig = go.Figure()
fig.add_trace(go.Scatter(x=f, y=p, name='Hermosillo Normal'))
fig.add_trace(go.Scatter(x=f1, y=p1, name='Hermosillo Tenis'))
fig.add_trace(go.Scatter(x=f2, y=p2, name='Hermosillo Zapatillas'))
fig.update_layout(title="Muestra 1 Hermosillo Warm Up Descalzo_Tenis_Zapatos", hoverlabel_align = 'auto') #Cambiar el titulo dependiendo el numero de muestra(puede ser muestra 1, 2, 3, etc.) y el modo ya sea Warm Up o Fat Burn.
fig.update_layout(xaxis={'title':'Frecuencia'}, yaxis={'title':'Amplitud'})
fig.show()


# # FERNANDA

# In[28]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Fernanda/Normal/WU/WU_F_N1_Each A.csv"
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[29]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Fernanda/Tenis/WU/WU_F_T1_Each A.csv"
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[30]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/Fernanda/Zapatillas/WU/WU_F_Z1_Each A.csv"
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[31]:


f,p =fft_signal(audio,Fs)
f1,p1 =fft_signal(audio1,Fs)
f2,p2 =fft_signal(audio2,Fs)

fig = go.Figure()
fig.add_trace(go.Scatter(x=f, y=p, name='Fernanda Normal'))
fig.add_trace(go.Scatter(x=f1, y=p1, name='Fernanda Tenis'))
fig.add_trace(go.Scatter(x=f2, y=p2, name='Fernanda Zapatillas'))
fig.update_layout(title="Muestra 1 Fernanda Warm Up Descalza_Tenis_Zapatillas", hoverlabel_align = 'auto')
fig.update_layout(xaxis={'title':'Frecuencia'}, yaxis={'title':'Amplitud'})
fig.show()


# # César

# In[32]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/César/Normal/WU/WU_Cr_N1_Each A.csv"
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[33]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/César/Tenis/WU/WU_Cr_T1_Each A.csv"
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[35]:


ruta = "C:/Users/cesar/Desktop/Muestras Marcha/24-08-20/César/Zapatos/WU/WU_Cr_Z1_Each A.csv"
df=pd.read_csv(ruta, header=None)
Fs = 8000
Ts = 1/Fs
t = np.linspace(Ts, Ts*len(df.values[:,0].astype(float)), len(df.values[:,0].astype(float)))
audio = (df.values[:,1].astype(float))


# In[37]:


f,p =fft_signal(audio,Fs)
f1,p1 =fft_signal(audio1,Fs)
f2,p2 =fft_signal(audio2,Fs)

fig = go.Figure()
fig.add_trace(go.Scatter(x=f, y=p, name='César Normal'))
fig.add_trace(go.Scatter(x=f1, y=p1, name='César Tenis'))
fig.add_trace(go.Scatter(x=f2, y=p2, name='César Zapatos'))
fig.update_layout(title="Muestra 1 César Warm Up Descalzo_Tenis_Zapatillas", hoverlabel_align = 'auto')
fig.update_layout(xaxis={'title':'Frecuencia'}, yaxis={'title':'Amplitud'})
fig.show()


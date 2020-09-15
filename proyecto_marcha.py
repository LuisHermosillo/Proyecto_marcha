import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import scandir, getcwd
import os.path
import csv
import plotly.graph_objects as go
%matplotlib inline 

## Funciones auxiliares

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
  
# Muestras del 24/08/2020
## Diferencia entre velocidades pero con la misma persona
ruta = "D:/hermo/Escritorio/datos/Muestras 24-08-2020/"
Fs = 8000
for px in ls(ruta):
    for calz in ls(os.path.join(ruta, px)):
        fig = go.Figure()

        for vel in ls(os.path.join(ruta, px, calz)):
            try:
                signal = []
                for archivo in ls(os.path.join(ruta, px, calz, vel)):
                    signal_aux = open_csv(os.path.join(ruta, px, calz, vel, archivo))
                    signal = np.concatenate((signal, signal_aux))

                p, f = fft_signal(signal, Fs)

                fig.add_trace(go.Scatter(x=p, y=f, name=vel, line=dict(dash='dot', width=3)))
            except Exception as e:
                print("Error en al lectura de la carpeta: "+str(os.path.join(ruta, px, calz, vel, archivo)))
                print(e)

        fig.update_layout(
            title="Persona: "+px+", Calzado: "+calz, hoverlabel_align = 'auto',
            xaxis={'title':'Frecuencia'}, yaxis={'title':'Magnitud'}
        )
        fig.show()
        
## Diferencia entre velocidades pero con diferentes personas
#Recopilacion de informacion ---------------------------------------
ruta_2 = "D:/hermo/Escritorio/datos/Muestras 23-07-2020/"

i = 1
Fs = 8000
dic = {}
clave = []
for persona in ls(ruta_2):
    for tipo in ls(ruta_2+persona):
        for modo in ls(ruta_2+persona+"/"+tipo):
            for archivo in ls(ruta_2+persona+"/"+tipo+"/"+modo):
                ruta_final = ruta_2+persona+"/"+tipo+"/"+modo+"/"+archivo
                senal = open_csv(ruta_final)
                dic[persona+" "+tipo+" "+modo] = senal
                try:
                    clave.index(tipo+" "+modo)
                except:
                    clave.append(tipo+" "+modo)
                break

#Creacion de graficas -----------------------------------------------
for i in range(len(clave)):
    fig = go.Figure()
    
    for key in dic:
        if key.find(clave[i]) >= 0:
            p, f = fft_signal(dic[key], 8000)
            fig.add_trace(go.Scatter(x=p, y=f, name=key, line=dict(dash='dot', width=2)))
            
    fig.update_layout(
        title=clave[i], hoverlabel_align = 'auto',
        xaxis={'title':'Frecuencia'}, yaxis={'title':'Magnitud'}
    )
    fig.show()
    
    
## Promedio con desviacion
#Recopilacion de informacion ---------------------------------------
ruta_3 = "D:\hermo\Escritorio\datos\Muestras 24-08-2020"

i = 1
Fs = 8000
dic_2 = {}
clave_2 = []
eje_x = []
for px in ls(ruta_3):
    for calz in ls(os.path.join(ruta_3, px)):
        for vel in ls(os.path.join(ruta_3, px, calz)):
            senales = 0
            for archivo in ls(os.path.join(ruta_3, px, calz, vel)):
                senal = open_csv(os.path.join(ruta_3, px, calz, vel, archivo))
                p, f = fft_signal(senal, Fs)
                senales = f +senales
                eje_x = p
                
            dic_2[px+" "+vel+" "+calz] = [senales-np.std(senales), senales, senales+np.std(senales)]
            try:
                clave_2.index(px+" "+vel)
            except:
                clave_2.append(px+" "+vel)

#Creacion de graficas -----------------------------------------------
Colors = ["red", "blue", "green", "orange"]

for i in range(len(clave_2)):
    fig = go.Figure()
    count = 0
    
    for key in dic_2:
        if key.find(clave_2[i]) >= 0:
            fig.add_trace(go.Scatter(x=eje_x, y=dic_2[key][0], showlegend=False, line=dict(color=Colors[count],dash='dot', width=2)))
            fig.add_trace(go.Scatter(x=eje_x, y=dic_2[key][1], name=key, line=dict(color=Colors[count],dash='dot', width=2)))
            fig.add_trace(go.Scatter(x=eje_x, y=dic_2[key][2], showlegend=False, line=dict(color=Colors[count],dash='dot', width=2)))
            count = count + 1
            
    fig.update_layout(
        title=clave[i], hoverlabel_align = 'auto',
        xaxis={'title':'Frecuencia'}, yaxis={'title':'Magnitud'}
    )
    fig.show()

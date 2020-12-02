# -*- coding: utf-8 -*-
"""
@author: Navid Hasanzadeh
@email: hasanzadeh.navid@gmail.com
"""
import glob
import numpy as np
from scipy.io import loadmat
from sklearn.svm import LinearSVC
import random
from sklearn import preprocessing
from scipy import signal
import sys
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

#####################################################
# Default Parameters                                #
#####################################################
subject_id = 1
session = 1
face_ids = [13,14,15,16,17,18,19,20,21,22,23,24]
object_ids = [73,74,75,76,78,79,80,83,84,85,88,89]
data_folder = 'data'
save_folder = 'results'
averaging_K = -1 #-1 for using all trials
fs = 1000 # MEG Sampling Rate
fc = 20 # Cut-off frequency of the filter
PLOT_decoding_time_series = True
PLOT_generalization = False
show_plots = True
save_plots = False
#####################################################
real_path = os.path.abspath(os.path.dirname(sys.argv[0]))
data_folder = real_path + '/' + data_folder
save_folder = real_path + '/' + save_folder
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
# IIR Butterworth Filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')

#
subject_id =  '%02d'%(int('00')+subject_id)
session = '%02d'%(int('00')+session)
# read faces data
faces_MEG = []
print('* Reading the MEG data - Faces')
for findx,face_id in enumerate(face_ids):
    face_MEG = []
#    print('Face_id: {}'.format(face_id))
    update_progress(findx/(len(face_ids)-1))
    face_id = '%04d'%(int('0000') + face_id)
    trials_path = glob.glob('{}/subj{}/sess{}/cond{}/*.mat'.format(data_folder,subject_id,session,face_id))
    if averaging_K>0:
        K_selected_trials = random.sample(trials_path,averaging_K)
    else:
        K_selected_trials = trials_path
    for trial_path in K_selected_trials:
        MEG_trial = loadmat(trial_path)
        MEG_trial_signals = MEG_trial['F']
        # Filter signal
        MEG_filtered = [signal.filtfilt(b, a, ch-np.mean(ch[0:int(0.1*fs)])) for ch in MEG_trial_signals]
        face_MEG.append(np.array(MEG_filtered))
    faces_MEG.append(list(np.mean(face_MEG,0)))

# read objects data
objects_MEG = []

print('* Reading the MEG data - Objects')
for oindx,object_id in enumerate(object_ids):
    object_MEG = []
#    print('Object_id: {}'.format(object_id))
    update_progress(oindx/(len(object_ids)-1))
    object_id = '%04d'%(int('0000') + object_id)
    trials_path = glob.glob('{}/subj{}/sess{}/cond{}/*.mat'.format(data_folder,subject_id,session,object_id))
    if averaging_K>0:
        K_selected_trials = random.sample(trials_path,averaging_K)
    else:
        K_selected_trials = trials_path
    for trial_path in K_selected_trials:
        MEG_trial = loadmat(trial_path)
        MEG_trial_signals = MEG_trial['F']
        # Filter signal
        MEG_filtered = [signal.filtfilt(b, a, ch-np.mean(ch[0:int(0.1*fs)])) for ch in MEG_trial_signals]
        object_MEG.append(np.array(MEG_filtered))
    objects_MEG.append(list(np.mean(object_MEG,0)))

# a sample plot of filtering signal
plt.figure(figsize=(10,6))
x_values = np.linspace(-100,1200,1301)
plt.subplot(211)
plt.plot(x_values,MEG_trial_signals[1],label='Raw Signal')
plt.ylabel('Amplitude')
plt.xlim(-100,1200)
plt.axvline(x=0,ls='--',c='k',linewidth=1)
plt.grid()
plt.legend()
plt.subplot(212)
plt.plot(x_values,MEG_filtered[1],'r',label='Normalized and Filtered Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim(-100,1200)
plt.axvline(x=0,ls='--',c='k',linewidth=1)
plt.grid()
plt.legend()
os.makedirs(save_folder, exist_ok=True)
if save_plots:
    plt.savefig(save_folder + '/' + 'sample_filtering_{}_{}.png'.format(subject_id,session))
    print('sample_filtering_{}_{}.png'.format(subject_id,session))
if show_plots:
    plt.show()

# Cleaning!
del object_MEG
del face_MEG
del MEG_trial
del trials_path

MEG_duration = np.shape(objects_MEG[0])[1]
MEG_channels = np.shape(objects_MEG[0])[0]
MEGs = np.array(faces_MEG + objects_MEG)
#MEGs = np.array(MEGs).reshape(np.shape(MEGs)[2],np.shape(MEGs)[0],np.shape(MEGs)[1])
MEGs_labels = np.array([0 for face in faces_MEG] + [1 for obj in objects_MEG])
if PLOT_decoding_time_series:
    accs = []
    print('* Training SVM models in order to make a Decoding Time Series plot')
    for t in range(MEG_duration):
        update_progress(t/(MEG_duration-1))
        #LOO
        t_accs = []
        for test_id in range(0,len(MEGs_labels)):
            train_indices = [i for i in range(0,len(MEGs_labels)) if i != test_id]
            test_x = MEGs[test_id,:,t].reshape(1,MEG_channels)
            test_y = MEGs_labels[test_id]
            train_x = MEGs[train_indices,:,t]
            train_y = MEGs_labels[train_indices]
            scaler = preprocessing.StandardScaler().fit(train_x)
            train_x = scaler.transform(train_x) 
            test_x = scaler.transform(test_x)
            model = LinearSVC(C=1)
            model.fit(train_x,train_y)
            y_predict = model.predict(test_x)
#            print(y_predict)
            if test_y==y_predict[0]:
                t_accs.append(1)
            else:
                t_accs.append(0)
        accs.append(np.mean(t_accs))
    accs_100 = 100 * np.array(accs)
    x_values = np.linspace(-100,1200,1301)
    plt.figure(figsize=(10,6))
    plt.plot(x_values,accs_100,c='indianred', label='Sig')
    accs_padded = 50*[accs_100[0]]
    accs_padded.extend(accs_100)
    accs_padded.extend(50*[accs_100[-1]])
    smoothed_accs = np.convolve(accs_padded, np.ones((50,))/50, mode='same')
    smoothed_accs = smoothed_accs[50:-50]
    max_loc = np.argmax(smoothed_accs)
    plt.plot(x_values,smoothed_accs,c='dimgray',linewidth=4, label='Smoothed Sig')    
    plt.ylabel('Decoding accuracy (%)')
    plt.xlabel('Time (ms)')
    plt.xlim(-100,1200)
    plt.axvline(x=0,ls='--',c='k',linewidth=1)
    plt.axvline(x=500,ls='--',c='b',linewidth=1)
    plt.axvline(x=max_loc-100,ls='--',c='g',linewidth=1)
#    plt.gca().annotate('local max', xy=(2, 1), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.annotate(
    '', xy=(max_loc-100, 30), xycoords='data',
    xytext=(0, 30), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
    plt.annotate(
    'Peak: {}ms'.format(max_loc-100), xy=(int((max_loc-100)/8), 32), xycoords='data',
    xytext=(5, 0), textcoords='offset points')
    plt.scatter(x_values[max_loc],smoothed_accs[max_loc],s=300,c='g')
    plt.legend()
    os.makedirs(save_folder, exist_ok=True)
    if save_plots:
        plt.savefig(save_folder + '/' + 'decoding_time_series_{}_{}.png'.format(subject_id,session))    
        print('decoding_time_series_{}_{}.png saved.'.format(subject_id,session))
    if show_plots:
        plt.show()


if PLOT_generalization:
    import cv2
    print('* Training SVM models in order to make a Temporal Generalization plot')
    tempo_gen = np.zeros([900,900])
    for t in range(0,900):
        update_progress(t/899)
        for t2 in range(0,900):
            t_accs = []
            for test_id in range(0,len(MEGs_labels)):
                train_indices = [i for i in range(0,len(MEGs_labels)) if i != test_id]
                test_x = MEGs[test_id,:,t2].reshape(1,MEG_channels)
                test_y = MEGs_labels[test_id]
                train_x = MEGs[train_indices,:,t]
                train_y = MEGs_labels[train_indices]
                scaler = preprocessing.StandardScaler().fit(train_x)
                train_x = scaler.transform(train_x) 
                test_x = scaler.transform(test_x)
                model = LinearSVC(C=1)
                model.fit(train_x,train_y)
                y_predict = model.predict(test_x)    
                if test_y==y_predict[0]:
                    t_accs.append(1)
                else:
                    t_accs.append(0)
            tempo_gen[t][t2] = np.mean(t_accs)
    import pickle
    os.makedirs(save_folder, exist_ok=True)
    with open(save_folder + '/' + 'TempoGen_{}_{}.pkl'.format(subject_id,session), 'wb') as f:
        pickle.dump([tempo_gen], f)
    tempo_gen_50 = 100*(tempo_gen*(tempo_gen>0.5) + 0.5*(tempo_gen<=0.5))
    kernel = np.ones((50,50),np.float32)/(50*50)
    tempo_gen_50_filtered = cv2.filter2D(tempo_gen_50,-1,kernel)    
    fig = plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.imshow(tempo_gen_50,cmap='copper',extent=[-100,800,800,-100])
    plt.gca().invert_yaxis()
    plt.ylabel('Train time (ms)',fontweight='bold')
    plt.xlabel('Test time (ms)\n(a)',fontweight='bold')
    plt.subplot(122)
    plt.imshow(tempo_gen_50_filtered,cmap='copper',extent=[-100,800,800,-100])
    plt.gca().invert_yaxis()
    #plt.ylabel('Train time (ms)')
    plt.xlabel('Test time (ms)\n(b)',fontweight='bold')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
    clb = plt.colorbar(orientation="vertical", cax=cbar_ax)
    clb.set_label('Decoding accuracy (%)', labelpad=15)
    if save_plots:
        plt.savefig(save_folder + '/' + 'temporal_generalization_{}_{}.png'.format(subject_id,session))
        print('temporal_generalization_{}_{}.png saved.'.format(subject_id,session))
    if show_plots:
        plt.show()
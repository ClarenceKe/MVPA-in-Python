% By Navid Hasanzadeh
% Email: hasanzadeh.navid@gmail.com
close all, clear all
data_folder = 'data';
face_ids = [13,14,15,16,17,18,19,20,21,22,23,24];
object_ids = [73,74,75,76,78,79,80,83,84,85,88,89];
%%
load([data_folder  '/'  'visual_stimuli.mat'])
figure
suptitle('Images of Face')
for i=1:12
    subplot(3,4,i)
    imshow(visual_stimuli(face_ids(i)).pixel_values)
    xlabel(num2str(face_ids(i)))
end
figure
suptitle('Images of Objects')
for i=1:12
    subplot(3,4,i)
    imshow(visual_stimuli(object_ids(i)).pixel_values)
    xlabel(num2str(object_ids(i)))
end
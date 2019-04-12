

clc; clear all; close all;

load Face_data.mat;

figure;
for k = 54 : size(images,2)
    k
    label = [lights(k)' poses(:,k)'];
    imagesc(reshape(images(:,k),64,64)); colormap gray
    drawnow
    pause%(.2)
end

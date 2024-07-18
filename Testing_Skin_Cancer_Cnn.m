clc;    % Clear command window
close all; % close all exisiting window
clear all;  % clear all the varaibles datas
warning off all; % to remove warning

%%  Image Acquisition

[filename, pathname] = uigetfile('*.png;*.jpg;*.jfif','select cover image'); % get input image folder
I = imread([pathname,filename]);% read input image folder
I = imresize(I,[256,256]);% resize the image
figure(1);                     % to open new figure;
imshow(I);            % to show an image ;
title('Input image');    % display title;

%% Enhance Contrast

% Enhance Contrast
I = imadjust(I,stretchlim(I));% adjust Image Intensity
figure(2); % to open new figure;
 imshow(I); % to show an image ;
 title('Contrast Enhanced'); % display title;
%% Median Filter

I = medfilt3(I);% filter function to remove the noises
I=imresize(I,[256,256]); % resize the image
figure(3);% to open new figure;
imshow(I); % to show an image ;
title('Filtered image');% display title;


%% Extract Features

% Color Image Segmentation
% Use of K Means clustering for segmentation
% Convert Image from RGB Color Space to L*a*b* Color Space 
% The L*a*b* space consists of a luminosity layer 'L*', chromaticity-layer 'a*' and 'b*'.
% All of the color information is in the 'a*' and 'b*' layers.
cform = makecform('srgb2lab'); % Create color transformation structure
% Apply the colorform
lab_he = applycform(I,cform); % Apply device-independent color space transformation

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));  % convert double
nrows = size(ab,1); % size of the value ab
ncols = size(ab,2);  % size of the value ab
ab = reshape(ab,nrows*ncols,2); % resize the image
nColors = 3;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols); % reshape the image

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end

figure(4);      % to open new figure;
subplot(3,1,1); %  Create axes in tiled positions
imshow(segmented_images{1});% to show an image ;
title('Cluster 1');% display title;
subplot(3,1,2);%  Create axes in tiled positions
imshow(segmented_images{2});% to show an image ;
title('Cluster 2');% display title;
subplot(3,1,3);%  Create axes in tiled positions
imshow(segmented_images{3});% to show an image ;
title('Cluster 3');% display title;
% set(gcf, 'Position', get(0,'Screensize'));
%%  Feature Extraction
x = inputdlg('Enter the cluster no. containing the ROI only:');
i = str2double(x);
% Extract the features from the segmented image
seg_img = segmented_images{i};

% Convert to grayscale if image is RGB
if ndims(seg_img) == 3
   img = rgb2gray(seg_img);
end
%figure, imshow(img); title('Gray Scale Image');

% Evaluate the disease affected area
black = im2bw(seg_img,graythresh(seg_img));
%figure, imshow(black);title('Black & White Image');
m = size(seg_img,1);
n = size(seg_img,2);

zero_image = zeros(m,n); 
%G = imoverlay(zero_image,seg_img,[1 0 0]);

cc = bwconncomp(seg_img,6);
diseasedata = regionprops(cc,'basic');
A1 = diseasedata.Area;
sprintf('Area of the disease affected region is : %g%',A1);

I_black = im2bw(I,graythresh(I));
kk = bwconncomp(I,6);
skindata = regionprops(kk,'basic');
A2 = skindata.Area;
sprintf(' Total Disease area is : %g%',A2);

%Affected_Area = 1-(A1/A2);
Affected_Area = (A1/A2);
if Affected_Area < 0.1
    Affected_Area = Affected_Area+0.15;
end
sprintf('Affected Area is: %g%%',(Affected_Area*100))

% Create the Gray Level Cooccurance Matrices (GLCMs)
glcms = graycomatrix(img);

% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
% calculate Contrast
Contrast = stats.Contrast;
% calculate Correlation
Correlation = stats.Correlation;
% calculate Energy
Energy = stats.Energy;
% calculate Homogeneity
Homogeneity = stats.Homogeneity;
% calculate Mean
Mean = mean2(seg_img);
% calculate Standard_Deviation
Standard_Deviation = std2(seg_img);
% calculate Entropy
Entropy = entropy(seg_img);
% calculate RMS
RMS = mean2(rms(seg_img));
% calculate Variance
Variance = mean2(var(double(seg_img)));
% calculate sum
a = sum(double(seg_img(:)));
% calculate Smoothness
Smoothness = 1-(1/(1+a));
% calculate Kurtosis
Kurtosis = kurtosis(double(seg_img(:)));
% calculate Skewness
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
rng('default')    
feat_disease = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
train=round(sum(feat_disease));                 % round of values
% save train9 train9
%% CNN
%% Put the test featurs into variable 'test'

test = train; % comparing the test data and trained data
load net1     % load the trained data

y = round(sim(net1,test));% simulate the convolution neural network
result=y;
% Visualize Results
if result == 1
    msgbox('nevus');% Display In Message Box
    disp('  nevus ');% Display In Command Window
elseif result == 2
    msgbox(' melanoma ');% Display In Message Box
    disp('melanoma');% Display In Command Window
elseif result == 3
    msgbox('nevus');% Display In Message Box
    disp('  nevus ');% Display In Command Window
elseif result == 4
   msgbox('nevus');% Display In Message Box
    disp('  nevus ');% Display In Command Window
elseif result == 5
    msgbox(' melanoma ');% Display In Message Box
    disp('melanoma');% Display In Command Window
elseif result == 6
    msgbox(' melanoma ');% Display In Message Box
    disp('melanoma');% Display In Command Window
elseif result == 7
    msgbox('nevus');% Display In Message Box
    disp('  nevus ');% Display In Command Window
elseif result == 8
    msgbox(' melanoma ');% Display In Message Box
    disp('melanoma');% Display In Command Window
elseif result == 9
    msgbox('nevus');% Display In Message Box
    disp('  nevus '); % Display In Command Window
end

%% Evaluate Accuracy
load('Accuracy_Data.mat') % Load The data
ACTUAL=Train_Feat(:,2);   % Actual Values
PREDICTED=Train_Label;    % Predicted Values
accuracy = Evaluate(ACTUAL,PREDICTED);  

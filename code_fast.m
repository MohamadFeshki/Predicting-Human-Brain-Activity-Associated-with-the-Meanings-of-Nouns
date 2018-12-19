%% Mohamad Feshki
clc,clear all;
close all;
%% initializing
load('data-science-P7.mat');
load('Fw.mat');
load('val.mat');
word=cell2mat(value);
load('verbarenged.mat');

voxels=19750;
TrialNum=60
a=meta.coordToCol;
a= permute(a,[2 1 3]);
[X Y Z]=size(a);
data7_3d=zeros( X ,Y ,Z ,TrialNum);
data7_cuts=zeros(5*X,5*Y,TrialNum);
data7=zeros(TrialNum,voxels);
%% visualization of nouns by FMRI signals 
for num=1:TrialNum
data71(num,:)=data{num, 1};
data7(num,:)=data71(num,1:voxels);

% for ploting initial FMRIs decode next lines

for i=1 : voxels
    [x,y,z] = ind2sub(size(a),find(a == i));
 data7_3d(x,y,z,num) = data7(num,i);
    
end
%  data7_3d= permute(data7_3d,[2 1 3 4]);

m=1;
for i=1:5
    for j=1:5
        if( m<24)
            data7_cuts(((i-1)*X)+1:(i*X),((j-1)*Y)+1:(j*Y),num)=data7_3d(:,:,m,num);
            m=m+1;
        end

    end
end
 B=data7_cuts(:,:,num);
figure
imagesc(B,[-3,3]);
colorbar;
colormap(jet);
title([ 'participant 7 FMRI data  \newline trial number: ', num2str(num),'. Word: ',info(1,num).word]);
end
%% data sorting base on verb rank
data7arrenged=zeros(TrialNum,voxels);


for i=1:TrialNum 
    for j=1:TrialNum
        trialinfo=info(1,j).word;
        if ((i==33 || i==51)&&( size(trialinfo,2) > 3)    )
            if(trialinfo(2:4) == word(i,1:end))
                data7arrenged(i,:)=data7(j,:);
                break;
            end
        end
        if(trialinfo(1:3) == word(i,1:end))
            data7arrenged(i,:)=data7(j,:);
        end
          
    end

end
%% test and train data segmentation
Ftrain=zeros(25,58);
Ftest1=zeros(25,1);
Ftrain(:,1:51)=Fw(:,1:51);
Ftest1(:,1)=Fw(:,52);
Ftrain(:,52:54)=Fw(:,53:55);
Ftest2(:,1)=Fw(:,56);
Ftrain(:,55:58)=Fw(:,57:TrialNum);
Ftest1=Ftest1';
Ftrain=Ftrain';

data7train=zeros(58,voxels);
data7test1=zeros(1,voxels);
data7train(1:51,:)=data7arrenged(1:51,:);
data7test1(1,:)=data7arrenged(52,:);
data7train(52:54,:)=data7arrenged(53:55,:);
data7test2(1,:)=data7arrenged(56,:);
data7train(55:58,:)=data7arrenged(57:TrialNum,:);

trialinfo=info(1:TrialNum).word;

%% linear fitting model and error calculation
load('Coefficient1.mat');
load('CeleryPredicted');
load('ErrorTestMSE');

% for retrain and retest the system denote next "for" and note 3 top lines
% witch take 10 minutes time
% 
% for vox=1:voxels  
% 
% rand2 = randperm(size(Ftest1, 1));
% trainX = Ftrain(:, :);
% trainY = data7train(:, vox);
% testX = Ftest1(1,:);
% testY = data7test1(1,vox);
% model = LinearModel.fit(trainX, trainY);
% C(:,vox)=table2array(model.Coefficients(2:end,1));
% % %by using C.mat its change with top line 
% Ypredict(vox)=model.predict(testX);
% ErrorTestMSE(1,vox) = mse(Ypredict(vox) - testY); 
% ErrorTestMSE(2,vox) = mse(Ypredict(vox) - data7test2(vox)); 
% 
% 
% end

%% cosine similarity and normalize MSE 
DISTANCEcelery2celery=pdist([Ypredict;data7test1(1:voxels)],'cosine');
DISTANCEairplane2celery=pdist([Ypredict;data7test2(1:voxels)],'cosine');

MSEcelery2celery=sum(ErrorTestMSE(1,:))/voxels;
MSEairplane2celery=sum(ErrorTestMSE(2,:))/voxels;


%%  visualization of trained verbs FMRI signals and compration between initial and pridictds nouns FMRI
verb7_3d=zeros( X ,Y ,Z ,25);
verb7_cuts=zeros(5*X,5*Y,25);
%n choose number of verb that plots!
n=1;
for ver=1:n
verb7(ver,:)=C(ver,1:voxels);

for i=1 : voxels
    [x,y,z] = ind2sub(size(a),find(a == i));
 verb7_3d(x,y,z,ver) =C(ver,i);
    
end
%  data7_3d= permute(data7_3d,[2 1 3 4]);

m=1;
for i=1:5
    for j=1:5
        if( m<24)
            verb7_cuts(((i-1)*X)+1:(i*X),((j-1)*Y)+1:(j*Y),ver)=verb7_3d(:,:,m,ver);
            m=m+1;
        end

    end
end
 B=verb7_cuts(:,:,ver);
figure
imagesc(B,[-3,3]);
colorbar;
colormap(jet);
title([ 'Learned signature for intermediate semantic feature:  verb: ',value(ver)]);
end
MSEairplane2celery
MSEcelery2celery
DISTANCEairplane2celery
DISTANCEcelery2celery

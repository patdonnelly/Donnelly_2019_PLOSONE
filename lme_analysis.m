%% LME Analysis Script for Donnelly 2019 PLOS ONE
% Patrick M. Donnelly
% University of Washington
% July 7, 2020

%% Data processing
% read in data 
data = readtable('data/matlab_data.csv');

% rename variables for ease in analysis
data.Properties.VariableNames = {'Var1', 'id', 'age', 'session', 'group', ...
    'study_name', 'word1_acc', 'word2_acc', 'pseudo1_acc', ...
    'pseudo2_acc', 'first_acc', 'second_rate', 'wj_brs', 'twre_index', ...
    'ctopp_rapid','ctopp_pa','wasi_fs2','practice'};

% make time and group variables categorical
categorical(data.session);
categorical(data.group);

% calculate difference scores and create new variables
data.worddiff = data.word2_acc - data.word1_acc;
data.pseudodiff = data.pseudo2_acc - data.pseudo1_acc;

%extend practice variable to both visits for LME analysis
for sub = 1:length(data.id)
   if isnan(data.practice(sub))
       data.practice(sub) = data.practice(sub+1);
   end
end

% create new stacked dataset for wordlist data analysis
data_stacked = stack(data,{'word1_acc','word2_acc',...
    'pseudo1_acc','pseudo2_acc'},'NewDataVariableName','acc');
data_stacked.type = data_stacked.acc_Indicator == 'word1_acc' | ...
    data_stacked.acc_Indicator=='word2_acc';

%% LME analysis {preregistered}

%% Real word analysis

model = 'acc ~ 1 + group*session + (1-session|id)+ (1|acc_Indicator)';

% focus in on dataset
real_data = data_stacked(data_stacked.type==true,:);

% run model fit
lme_realword = fitlme(real_data, model, 'FitMethod', 'REML'); 


%% Pseudo word analysis

model = 'acc ~ 1 + group*session + (1-session|id)+ (1|acc_Indicator)';


pseudo_data = data_stacked(data_stacked.type==false,:);

% run model fits
lme_pseudoword = fitlme(pseudo_data, model, 'FitMethod', 'REML');


%% Passage data

% accuracy model
model = 'first_acc ~ 1 + group*session + (1-session|id)';


% run model fits
lme_accuracy = fitlme(data, model, 'FitMethod', 'REML');



% rate model
model = 'second_rate ~ 1 + group*session + (1-session|id)';


lme_rate = fitlme(data, model, 'FitMethod', 'REML');








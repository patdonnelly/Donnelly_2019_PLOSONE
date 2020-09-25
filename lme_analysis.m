%% LME Analysis Script for Donnelly 2019 PLOS ONE
% Patrick M. Donnelly
% University of Washington
% September 23, 2020

%% Data processing
% read in data 
data = readtable('data/matlab_data.csv');

% rename variables for ease in analysis
data.Properties.VariableNames = {'Var1', 'id', 'age', 'session', 'group', ...
    'study_name', 'word1_acc', 'word2_acc', 'pseudo1_acc', ...
    'pseudo2_acc', 'word_acc', 'pseudo_acc', 'first_acc', ...
    'second_rate', 'wj_brs', 'twre_index', 'ctopp_rapid', ...
    'ctopp_pa','wasi_fs2','practice'};

% make time and group variables categorical
session = categorical(data.session);
group = categorical(data.group);

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

%create by group data sets for looking at by group analyses
int_data = data_stacked(data_stacked.group==1,:);
cntrl_data = data_stacked(data_stacked.group==0,:);
real_int = int_data(int_data.type==true,:);
real_control = cntrl_data(cntrl_data.type==true,:);
pseudo_int = int_data(int_data.type==false,:);
pseudo_control = cntrl_data(cntrl_data.type==false,:);


%% LME analysis {preregistered}

%% Real word analysis
% focus in on dataset
real_data = data_stacked(data_stacked.type==true,:);

%By group analysis
% paired t-tests for each group
int_data = data(data.group==1,:);
cntrl_data = data(data.group==0,:);


[h,p,ci,stats] = ttest(int_data.word_acc(int_data.session == 2), ...
    int_data.word_acc(int_data.session == 1));
[h,p,ci,stats] = ttest(cntrl_data.word_acc(cntrl_data.session == 2), ...
    cntrl_data.word_acc(cntrl_data.session == 1));

%random effects model fitting
model_1 = 'acc ~ 1 + group*session + (1|id)';
model_2 = 'acc ~ 1 + group*session + (session|id)';
model_3 = 'acc ~ 1 + group*session + (1|id) + (1|acc_Indicator)';
model_4 = 'acc ~ 1 + group*session + (session|id) + (1|acc_Indicator)';

%try adding age and phonological awareness
model_5 = 'acc ~ 1 + age + ctopp_pa + group*session + (1|id) + (1|acc_Indicator)';

fit_1 = fitlme(real_data, model_1, 'FitMethod', 'REML'); 
fit_2 = fitlme(real_data, model_2, 'FitMethod', 'REML');
compare(fit_1, fit_2) %fit_1 is better
fit_3 = fitlme(real_data, model_3, 'FitMethod', 'REML');
compare(fit_1, fit_3) %fit_3 is better
fit_4 = fitlme(real_data, model_4, 'FitMethod', 'REML');
compare(fit_3, fit_4) %fit_3 is better
fit_5 = fitlme(real_data, model_5, 'FitMethod', 'REML');
compare(fit_3, fit_5) % fit_3 is better
%indicates no greater model fit with added covariates
%albeit not a good fit, model statistics suggest an effect of age
%(0.10, p = 0.045)

%plot residuals to check heteroscedasticity
figure();
plotResiduals(fit_3)%reveals normally distributed residuals

%plot the fitted response versus the observed response and residuals
F = fitted(fit_3);
R = response(fit_3);
figure();
plot(R,F,'rx')
xlabel('Response')
ylabel('Fitted')
%indicates good fit

%This model will be used for both real words and pseudo words for
%consistency in model interpretation


% Run the analysis
lme_realword = fitlme(real_data, model_3, 'FitMethod', 'REML'); 


%% Pseudo word analysis

%look at dataset
pseudo_data = data_stacked(data_stacked.type==false,:);

%By group analysis
% paired t-tests for each group
[h,p,ci,stats] = ttest(int_data.pseudo_acc(int_data.session == 2), ...
    int_data.pseudo_acc(int_data.session == 1));
[h,p,ci,stats] = ttest(cntrl_data.pseudo_acc(cntrl_data.session == 2), ...
    cntrl_data.pseudo_acc(cntrl_data.session == 1));

%Same model as with real words for consistency
model_3 = 'acc ~ 1 + group*session + (1|id) + (1|acc_Indicator)';

%model fit
fit_3 = fitlme(pseudo_data, model_3, 'FitMethod', 'REML');

%try adding age and phonological awareness
model_5 = 'acc ~ 1 + age + ctopp_pa + group*session + (1|id) + (1|acc_Indicator)';

% checking for role of covariates of age and ctopp
fit_3 = fitlme(pseudo_data, model_3, 'FitMethod', 'REML');
fit_5 = fitlme(pseudo_data, model_5, 'FitMethod', 'REML');
compare(fit_3, fit_5) 
%indicates that the age/ctopp model not a better fit
%indicates a suggestive, yet non-signifanct effect of age 
%(0.115, p = 0.049)

%plot residuals to check heteroscedasticity
figure();
plotResiduals(fit_3) %residuals are normally distributed

%plot the fitted response versus the observed response and residuals
F = fitted(fit_3);
R = response(fit_3);
figure();
plot(R,F,'rx')
xlabel('Response')
ylabel('Fitted')
%indicates a pretty good fit


% run model
lme_pseudoword = fitlme(pseudo_data, model_3, 'FitMethod', 'REML');


%% Passage Accuracy


%By group analysis
% paired t-tests for each group
[h,p,ci,stats] = ttest(int_data.first_acc(int_data.session == 2), ...
    int_data.first_acc(int_data.session == 1));
[h,p,ci,stats] = ttest(cntrl_data.first_acc(cntrl_data.session == 2), ...
    cntrl_data.first_acc(cntrl_data.session == 1));

% use model fit from word list analyses but not with word list random
% effect
model_3 = 'first_acc ~ 1 + group*session + (1|id)'; % best fit

%try adding age and phonological awareness
model_5 = 'first_acc ~ 1 + age + ctopp_pa + group*session + (1|id)'; 
%model fit
fit_3 = fitlme(data, model_3, 'FitMethod', 'REML'); 
fit_5 = fitlme(data, model_5, 'FitMethod', 'REML'); 
compare(fit_3, fit_5) %fit_3 is better
%not better fit and no near significant fixed effects

%plot residuals to check heteroscedasticity
figure();
plotResiduals(fit_3) %residuals are normally distributed

%plot the fitted response versus the observed response and residuals
F = fitted(fit_3);
R = response(fit_3);
figure();
plot(R,F,'rx')
xlabel('Response')
ylabel('Fitted')
%indicates a pretty good fit


% run accuracy model
lme_accuracy = fitlme(data, model_3, 'FitMethod', 'REML');





%% Pssage Reading Rate

%By group analysis
% paired t-tests for each group
[h,p,ci,stats] = ttest(int_data.second_rate(int_data.session == 2), ...
    int_data.second_rate(int_data.session == 1));
[h,p,ci,stats] = ttest(cntrl_data.second_rate(cntrl_data.session == 2), ...
    cntrl_data.second_rate(cntrl_data.session == 1));

% rate model
% model fit same as accuracy
model_3 = 'second_rate ~ 1 + group*session + (1|id)'; 

%try adding age and phonological awareness
model_5 = 'second_rate ~ 1 + age + ctopp_pa + group*session + (1|id)'; 

fit_3 = fitlme(data, model_3, 'FitMethod', 'REML'); 
fit_5 = fitlme(data, model_5, 'FitMethod', 'REML');
compare(fit_3, fit_5)
%not better fit and no near significant fixed effects

%plot residuals to check heteroscedasticity
figure();
plotResiduals(fit_3) %residuals are normally distributed

%plot the fitted response versus the observed response and residuals
F = fitted(fit_3);
R = response(fit_3);
figure();
plot(R,F,'rx')
xlabel('Response')
ylabel('Fitted')
%indicates a good fit


%run rate model
lme_rate = fitlme(data, model_3, 'FitMethod', 'REML');



%% Correlation analyses

% Real word decoding





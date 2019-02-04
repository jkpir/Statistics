function [p_value,p_partial] = perm_multi_fisher(Dataset,Control,B)
%% ------------------------------------------------------------------------
%% Non-parametric multivariable permutation test for location problem of
%  two independent samples using Fisher's method for combination of partial
%  tests.
%  
%  References:
%  Bonnini, Stefano & Salmaso, Luigi & Solari, Aldo. Multivariate 
%  permutation tests for evaluating effectiveness of universities through 
%  the analysis of student dropouts.  Statistica & Applicazioni, Vol. 3,
%  2005, 37-44.
% 
%  Rosa Arboretti Giancristofaro and Stefano Bonnini. "Permutation Tests 
%  for Heterogeneity Comparisons in Presence of Categorical Variables with 
%  Application to University Evaluation", Metodolo嗅i zvezki, Vol. 4, No. 1,
%  2007, 21-36.
%  
%  Rosa Arboretti Giancristofaro, Mario Bolzan, Federico Campigotto Livio 
%  Corain and Luigi Salmaso. "Combination-based permutation testing in 
%  survival analysis", Quaderni di Statistica, Vol. 12, No 1, 2010, 15-38.
%
%  PESARIN, Fortunato; SALMASO, Luigi. The permutation testing approach: a 
%  review. Statistica, [S.l.], v. 70, n. 4, p. 481-509, dec. 2010. ISSN 
%  1973-2201. Available at: 
%  <https://rivista-statistica.unibo.it/article/view/3599/2950>. 
%  Date accessed: 19 dec. 2018. doi:https://doi.org/10.6092/issn.1973-2201/3599. 
%  
%  Dataset format:
%  First column: thw sample (Algorithm) ID (1 or 2).
%  Other columns: the values found for each function.
%  Lines: Each experiment.
%  Example:
%            ID    F1    F2     F3
%  Dataset = [1  0.012  0.011  0.42         Run 1 (Alg. 1)
%             1  0.013  0.013  0.43         Run 2 (Alg. 1)
%             1  0.013  0.012  0.44         Run 3 (Alg. 1)
%             1  0.013  0.011  0.42         Run 4 (Alg. 1)
%             1  0.012  0.013  0.43         Run 5 (Alg. 1)
%             2  0.013  0.013  0.45         Run 1 (Alg. 2)
%             2  0.013  0.012  0.42         Run 2 (Alg. 2)
%             2  0.012  0.012  0.43         Run 3 (Alg. 2)
%             2  0.014  0.011  0.43];       Run 4 (Alg. 2)
%
%  Input variable 'B' is the number of random permutations with repetitions
%  (Glivenko砲antelli theorem) instead of performing all the possible permutations
%  (it would be (n1+n2)! / (n1! * n2!) => 1.1826e+17 for n1=n2=30!!!)
%  The default value for 'B' is 10000.
%
%  To define the 'T value' calculation change line 67 as follow:
%  1: For 'Sum of the first group' use: T = @(g1,g2,n1,n2) calculate_T(g1)
%  2: For 'Difference of means' use:    T = @(g1,g2,n1,n2) calculate_T(g1,g2)
%  3: For 'Normal aproximation' use:    T = @(g1,g2,n1,n2) calculate_T(g1,g2,n1,n2)
%
%  The default method is 'Difference of means'. Normal aproximation is not
%  recommended.
%
%  The 'p_value' returned is the relative rank of combinet T value of the
%  original Dataset (not permuted). Reject the null hypothesis (equal means
%  of samples 1 and 2) if the 'p_value' calculated is lower than the
%  significance level desired (alpha).
%
%  Coded by: Juliano Pierezan (July, 2015)
%            juliano.pierezan@ufpr.br
%% ------------------------------------------------------------------------

%% Define the T value calculation procedure
T = @(g1,g2,n1,n2) calculate_T(g1,g2);

%% Considering two groups compared each time
if nargin <=1
    Control=Dataset(1,1);
end
if nargin <= 2
    B = 10000;
end
nvar     = size(Dataset,2)-1;
T_values = zeros(B,nvar);

%% Extract information form Dataset
samples     = unique(Dataset(:,1));
% Put the control in first place
samples     = [samples(samples==Control);samples(samples~=Control)];
n_samples   = size(samples,1);
per_sample  = zeros(n_samples,1);
for s=1:n_samples
    per_sample(s,1) = sum(Dataset(:,1)==samples(s,1));
end
n1 = per_sample(1,1);
n2 = per_sample(2,1);

ntotal = size(Dataset,1);
p_partial = ones(1,nvar);
    
%% T_value of the original Dataset (not permuted)
group1 = Dataset(Dataset(:,1)==samples(1,1),2:nvar+1);
group2 = Dataset(Dataset(:,1)==samples(2,1),2:nvar+1);
T_values(1,1:nvar) = T(group1,group2,n1,n2);

%% If the groups are equal, then there's no difference
if group1 == group2
    p_value = 1;
else
    for p=2:B
        %% Random sampling strategy
        permut = randperm(ntotal);
        group1 = Dataset(permut(1:n1),2:nvar+1);
        group2 = Dataset(permut(n1+1:n1+n2),2:nvar+1);
        
        %% T_value calculated for each each permutation
        T_values(p,1:nvar) = T(group1,group2,n1,n2);
    end
    
    %% For multivariate problems
    if nvar > 1
        [T_comb,p_partial]  = combine_T(T_values);
    else
        T_comb      = T_values;
    end
    
    %% The p_value is the realtive rank of the T_obs (the T_obs_rank/B)
    [~,inds]         = sort(T_comb,'descend');
    re_ranks(inds,1) = (1:B)'/B;
    p_value          = re_ranks(1,1);
end
end
function [T_comb,p_partial]  = combine_T(T_values)
%% Fisher weighted combination method (equal weights)
[B,nvar] = size(T_values);
w        = ones(B,nvar);

% Rank the 'T_values'
[~,inds] = sort(T_values,'descend');

% P_values of the partial tests (each variable)
rel_ranks = zeros(B,nvar);
for v=1:nvar
    rel_ranks(inds(:,v),v)=(1:B)'/B;
end
p_partial = rel_ranks(1,:);

%% Combination function (Fisher) with weights (w)
T_comb   = -2*sum(w.*log(rel_ranks),2);
end
function T_value = calculate_T(group1,group2,n1,n2)
if nargin<=0
    error('No input variables!');
elseif nargin <=1 % Sum of first group
    T_value  = sum(group1,1);
elseif nargin <=2 % Difference of means
    T_value  = abs(mean(group1,1) - mean(group2,1));
elseif nargin >= 4 % Normal aproximation (considering equal populations' variances)
    s1  = var(group1);
    s2  = var(group2);
    sp2 = ((n1-1).*s1.^2 + (n2-1).*s2.^2)./(n1+n2-2);
    T_value  = (mean(group1,1) - mean(group2,1))./sqrt(sp2.*(1./(n1+n2)));
end
end
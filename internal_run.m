function acc = internal_run(neural_dat, target_variable, nIter, permFlag, minTrials, T)
% INTERNAL_RUN Performs decoding over multiple iterations

    all_comb = unique(target_variable{1},'rows'); % calculate every combination of variables
    acc = zeros(nIter, T, size(all_comb,2), size(all_comb,1)); %initialize : iter x time x variable x cond 
    
    for ci = 1:size(all_comb,2)
       
        % abstract unique feature of each targeted variable
        [C,~,~] = unique(all_comb(:,ci));

        % extract 'row' of conditions based on each unique elements for target variable
        comb_matrix1 = find(all_comb(:,ci) == C(1));
        comb_matrix2 = find(all_comb(:,ci) == C(2));

        % Generate all possible 'row' combinations of 2 elements from each matrix
        combinations_cond = combvec(comb_matrix1', comb_matrix2')';

        for i2 = 1:size(combinations_cond,1)
            % which combination will be used as train?
            targ_comb = all_comb(combinations_cond(i2,:)',:);
            untarg_comb = all_comb(~ismember(all_comb,targ_comb,'rows'),:);

            parfor kk = 1:nIter
              [trainX, trainY] = collect_data(neural_dat, target_variable, targ_comb, minTrials);
              [testX, testY]   = collect_data(neural_dat, target_variable, untarg_comb, minTrials);

                for tt = 1:T
                    Xtrain = cell2mat(cellfun(@(x) x(:,tt), trainX, 'UniformOutput', false));
                    Xtest  = cell2mat(cellfun(@(x) x(:,tt), testX, 'UniformOutput', false));

                    if permFlag
                        [Xtrain, Xtest] = Geometric_random_model(Xtrain, trainY{1}, Xtest, testY{1});
                    end

                    Mdl = fitcecoc(Xtrain, trainY{1}(:,ci));
                    pred = predict(Mdl, Xtest);
                    acc(kk,tt,ci,i2) = mean(pred == testY{1}(:,ci)) * 100; 
                end
            end
        end
    end
end

function [Xcell, Ycell] = collect_data(neural_dat, target_variable, comb, mTrials)
% COLLECT_DATA Samples fixed number of trials per class for each unit

    N = numel(neural_dat);
    Xcell = cell(1,N);
    Ycell = cell(1,N);

    for i3 = 1:size(comb,1)

        % make targeted trials which corresponds to label of variable
        target_trials = cellfun(@(x) find(ismember(x,comb(i3,:),'rows')), target_variable, 'UniformOutput', false);

        % shuffle the order
        target_trials = cellfun(@(x) x(randperm(length(x))), target_trials, 'UniformOutput', false);

        % choose random trial among them in each neuron
        train_trials = cellfun(@(x) x(1:mTrials), target_trials, 'UniformOutput', false);

        % extract neural activity
        temp_neural = cellfun(@(x, y) x(y,:), neural_dat, train_trials, 'UniformOutput', false);

        % extract label
        temp_label = cellfun(@(x, y) x(y,:), target_variable, train_trials, 'UniformOutput', false);

        % concatenate matrix
        Xcell = cellfun(@(x, y) [x; y], Xcell, temp_neural, 'UniformOutput', false);
        Ycell = cellfun(@(x, y) [x; y], Ycell, temp_label, 'UniformOutput', false);
    end

end


parameters  = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1-1e-8];
np=length(parameters);
[trow,tcol]=size(X_test);


accuracy=zeros(np,1);
nnzero=zeros(np,1); %nnzero is the number of non-zero elements(number of selected features) in weights
auc=zeros(np,1); %auc records the area under the roc curve corresponding to each parameters
for i=1:np
    par=parameters(i);
    [w,c] = logistic_l1_train(X_train,y_train,par);
    nnzero(i)=nnz(w);
    
    labelp=zeros(trow,1);
    err=zeros(trow,1);
    for j=1:trow
        labelp(j)=w'*X_test(j,:)'+c;
        
%         if labelp(j)>=0 
%             labelp(j)=1;
%         else
%             labelp(j)=-1;
%         end
%         
%         if labelp(j)==y_test(j)
%             err(j)=0;
%         else
%             err(j)=1;
%         end
%       
        
    end
    
    %rescale labelp between -1 and 1
%     clear max
%     clear min
    maxx = max(labelp);
    minn = min(labelp);
    % if par=1 , then we got maxx=minn, and the rescaling caught error,
    % soln: change the last parameter into 1-1e-8
    for k=1:trow
        labelp(k)=(labelp(k)-((maxx+minn)/2))/((maxx-minn)/2);
    end
    
    %X returns the false positive rate; Y returns the true positive rate;
    %AUC returns the auc value
    [X,Y,~,AUC]=perfcurve(y_test,labelp,1);
    auc(i)=AUC;
   
%     accuracy(i)=(trow-sum(err))/trow;
%     
%     plot(parameters,accuracy)
    
           
end

auc

plot(parameters,nnzero)
xlabel('L1 regularization parameter'); ylabel('Number of selected features')
title('Number of features vary with different parameters');
    
figure;
plot(parameters,auc)
xlabel('L1 regularization parameter'); ylabel('Area under roc curve')
title('auc of models vary with different parameters')

    
    
    

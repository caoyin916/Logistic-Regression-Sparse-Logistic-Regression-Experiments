ntrain=[200,500,800,1000,1500,2000];

n=length(ntrain);
accuracy=zeros(n,1);

for i=1:n
    nn=ntrain(i);
    [weights,acc]=logistic_train(spam_data,spam_label,nn,1e-5,1000);
    accuracy(i)=acc;
end

accuracy

plot(ntrain,accuracy)
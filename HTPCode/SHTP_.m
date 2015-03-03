% SHTP_.m
% Basic implementation of the Simultaneous Hard Thresholding Pursuit
% algorithm after normalization of the measurement matrix
% Find the jointly sparse solutions of the K underdetermined mXN linear system Ax_1=y_1,...,Ax_K=y_K 
%
% Usage: [X,S,NormRes,NbIter] = SHTP_(Y,A,s,MaxNbIter,mu,X0,TolRes,Warnings,Eps)
%
% Y: mxK matrix whose K columns are measurement vectors y_1,...,y_K
% A: mxN measurement matrix
% s: sparsity level
% MaxNbIter: number of iterations not to be exceeded (optional, default=500)
% mu: factor of A*(Y-AX) in the support-identification step (optional, default=1) 
% X0: initial matrix (optional, default=zero)
% TolRes: tolerance on the Frobenius norm |Y-AX|/|Y| of the relative residual (optional, default=1e-4)
% Warnings: use 'On' to display the warnings, 'No' otherwise (optional,
% default='On')
% Eps: thresholding for sending to zero the rows of a matrix with
% norm smaller than Eps times the largest row of the matrix in norm
% (optional, default=1e-8)
%
% X: an s-row-sparse matrix which is the possible solution of AX=Y
% S: the row-support of X
% NormRes: the Frobenius norm |Y-AX| of the residual
% NbIter: the number of iterations performed until stationary outputs are reached
%         if stationary outputs are not reached, a warning is displayed and NbIter takes MaxNbIter 
%
% Written by Simon Foucart in February 2011
% Code proposed and used in the paper "Recovering jointly sparse vectors via Hard Thresholding Pursuit"
% Send comments to simon.foucart@centraliens.net

function [X,S,NormRes,NbIter] = SHTP_(Y,A,s,MaxNbIter,mu,X0,TolRes,Warnings,Eps)


%% set the default values
N=size(A,2);
K=size(Y,2);
if nargin<9
    Eps=1e-8; 
end
if nargin<8
    Warnings='On';
end
if nargin<7
    TolRes=1e-4;
end
if nargin<6
    X0=zeros(N,K);
	S0=1:s;
end
if nargin<5
    mu=1;
end
if nargin<4
    MaxNbIter=500;
end

%% renormalization of A
d=ones(1,N);
for j=1:N
   d(j)=1/(norm(A(:,j)));
end
A=A*diag(d);

%% define auxiliary quantities
B=A'*A;
Z=A'*Y;
[~,sorted_idx]=sort(sum(X0.*X0,2),'descend');
S0=sorted_idx(1:s);

%% initialization
X=X0;
S=S0; 
G=Z-B*X;
V=X+mu*G;
Norm2RowsV=sum(V.*V,2);
zero_idx=find(Norm2RowsV<Eps^2*max(Norm2RowsV));
Norm2RowsV(zero_idx)=zeros(size(zero_idx));
[~,sorted_idx]=sort(Norm2RowsV,'descend');
Snew=sort(sorted_idx(1:s)); 
Xnew=zeros(N,K);
Xnew(Snew,:)=A(:,Snew)\Y;
NbIter=1;

%% main loop
while ( (sum(S==Snew)<s) && (NbIter<MaxNbIter) )
	X=Xnew;
	S=Snew;
	G=Z-B*X;
    V=X+mu*G;
    Norm2RowsV=sum(V.*V,2);
    zero_idx=find(Norm2RowsV<Eps^2*max(Norm2RowsV));
    Norm2RowsV(zero_idx)=zeros(size(zero_idx));
    [~,sorted_idx]=sort(Norm2RowsV,'descend');
    Snew=sort(sorted_idx(1:s)); 
    Xnew=zeros(N,K);
    Xnew(Snew,:)=A(:,Snew)\Y;
	NbIter=NbIter+1;
end

%% outputs
NormRes=norm(Y-A*Xnew,'fro');
if strcmp(Warnings,'On')
    if sum(S==Snew)<s
        disp(strcat('Warning: SHTP_ did not converge when using a number of iterations =', num2str(MaxNbIter)));
    else 
    if NormRes>TolRes*norm(Y,'fro')
        disp(strcat('Warning: SHTP_ converged to an incorrect solution (norm of residual =', num2str(NormRes),')'));
    end
    end
end
X=diag(d)*Xnew;
S=Snew;

end
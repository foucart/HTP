% HTP_.m
% Basic implementation of the Hard Thresholding Pursuit algorithm after
% normalisation of the measurement matrix
% Find the s-sparse solution of the underdetermined mXN linear system Ax=y 
%
% Usage: [x,S,NormRes,NbIter] = HTP_(y,A,s,MaxNbIter,mu,x0,TolRes,Warnings,Eps)
%
% y: mx1 measurement vector
% A: mxN measurement matrix
% s: sparsity level
% MaxNbIter: number of iterations not to be exceeded (optional, default=500)
% mu: factor of A*(y-Ax) in the support-identification step (optional, default=1)
%     use 'NHTP' for the Normalized Hard Thresholding Pursuit algorithm 
% x0: initial vector (optional, default=zero)
% TolRes: tolerance on the Euclidean norm |y-Ax|/|y| of the relative residual (optional, default=1e-4)
% Warnings: use 'On' to display the warnings, 'No' otherwise (optional,
% default='Y')
% Eps: thresholding for sending to zero the entries of a vector with magnitude smaller than Eps times the largest entry of the vector in magnitude (optional, default=1e-8)
%
% x: an s-sparse vector which is the possible solution of Ax=y
% S: the support of x
% NormRes: the Euclidean norm |y-Ax| of the residual
% NbIter: the number of iterations performed until stationary outputs are reached
%         if stationary outputs are not reached, a warning is displayed and NbIter takes MaxNbIter 
%
% Written by Simon Foucart in August 2010, updated in February 2011
% Code proposed and used in the paper "Hard Thresholding Pursuit: an algorithm for Compressive Sensing"
% Send comments to simon.foucart@centraliens.net


function [x,S,NormRes,NbIter] = HTP_(y,A,s,MaxNbIter,mu,x0,TolRes,Warnings,Eps)


%% set the default values
N=size(A,2);
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
    x0=zeros(N,1);
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
z=A'*y;
[~,sorted_idx]=sort(abs(x0),'descend');
S0=sorted_idx(1:s);

%% initialization
x=x0;
S=S0; 
g=z-B*x;
if mu=='NHTP'
	Mu=(norm(g(S))/norm(A(:,S)*g(S)))^2;
else 
	Mu=mu;
end
v=x+Mu*g; 
absv=abs(v); 
zero_idx=find(absv<Eps*max(absv));
absv(zero_idx)=zeros(size(zero_idx));
[~,sorted_idx]=sort(absv,'descend');
Snew=sort(sorted_idx(1:s)); 
xnew=zeros(N,1);
xnew(Snew)=A(:,Snew)\y;
NbIter=1;


%% main loop
while ( (sum(S==Snew)<s) && (NbIter<MaxNbIter) )
	x=xnew;
	S=Snew;
	g=z-B*x;
	if mu=='NHTP'
		Mu=(norm(g(S))/norm(A(:,S)*g(S)))^2;
	else
		Mu=mu;
	end
	v=x+Mu*g; 
    absv=abs(v); 
    zero_idx=find(absv<Eps*max(absv));
    absv(zero_idx)=zeros(size(zero_idx));
    [~,sorted_idx]=sort(absv,'descend');
    Snew=sort(sorted_idx(1:s)); 
    xnew=zeros(N,1);
    xnew(Snew)=A(:,Snew)\y;
	NbIter=NbIter+1;
end


%% outputs
NormRes=norm(y-A*xnew);
if strcmp(Warnings,'On')
    if sum(S==Snew)<s
        disp(strcat('Warning: HTP_ did not converge when using a number of iterations =', num2str(MaxNbIter)));
    else 
    if NormRes>TolRes*norm(y)
        disp(strcat('Warning: HTP_ converged to an incorrect solution (norm of residual =', num2str(NormRes),')'));
    end
    end
end
x=diag(d)*xnew;
S=Snew;


end
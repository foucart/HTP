% FHTP.m
% Basic implementation of the Fast Hard Thresholding Pursuit algorithm
% Find the s-sparse solution of the underdetermined linear system Ax=y 
%
% Usage: [x,S,NormRes,NbIter] = FHTP(y,A,s,MaxNbIter,mu,NbDesc,t,x0,TolRes,Warnings)
%
% y: mx1 measurement vector
% A: mxN measurement matrix
% s: sparsity level
% MaxNbIter: number of iterations not to be exceeded (optional, default=500)
% mu: factor of A*(y-Ax) in the support-identification step (optional, default=1)
%     use 'NFHTP' for the Normalized Fast Hard Thresholding Pursuit algorithm
%     use 'NIHT' for the Normalized Iterative Hard Thhresholding algorithm
%     (this forces NbDesc to equal 0)
% NbDesc: number of descent iterations used to replace the projection step (optional, default=3)
% t: factor of (A*(y-A*u))_S in the descent steps (optional, default='steepest')
%    use 'steepest' for the choice dictated by a steepest descent
% x0: initial vector (optional, default=zero)
% TolRes: tolerance on the Euclidean norm |y-Ax|/|y| of the relative residual (optional, default=1e-4)
% Warnings: use 'On' to display the warnings, 'No' otherwise (optional, default='On')
%
% x: an s-sparse vector which is the possible solution of Ax=y
% S: the support of x
% NormRes: the Euclidean norm |y-Ax| of the residual
% NbIter: the number of iterations performed until |y-Ax|<=TolRes*|y|
%         if this does not occur, a warning is displayed and NbIter takes MaxNbIter 
%
% Written by Simon Foucart in August 2010, updated in February 2011
% Code proposed and used in the paper "Hard Thresholding Pursuit: an algorithm for Compressive Sensing"
% Send comments to simon.foucart@centraliens.net


function [x,S,NormRes,NbIter] = FHTP(y,A,s,MaxNbIter,mu,NbDesc,t,x0,TolRes,Warnings)

%% set the default values
N=size(A,2);
if nargin<10
    Warnings='On';
end
if nargin<9
    TolRes=1e-4;
end
if nargin<8
	x0=zeros(N,1);
	S0=1:s;
end
if nargin<7
	t='steepest';
end	
if nargin<6
	NbDesc=3;
end
if nargin<5
	mu=1;
end
if nargin<4
    MaxNbIter=500;
end
if strcmp(mu,'NIHT')
   NbDesc=0; 
end


%% define auxiliary quantities
B=A'*A;
z=A'*y;
normy=norm(y);
[~,sorted_idx]=sort(abs(x0),'descend');
S0=sorted_idx(1:s)';


%% initialization
x=x0;
S=S0;
NormRes=norm(y-A*x0);
NbIter=0;


%% main loop
while ( (NormRes>TolRes*normy) && (NbIter<MaxNbIter) )
    xold=x;
    Sold=S;
	g=z-B*x;
	if ( (strcmp(mu,'NFHTP')) || (strcmp(mu,'NIHT')) )
		Mu=(norm(g(S))/norm(A(:,S)*g(S)))^2;
	else
		Mu=mu;
	end
	v=x+Mu*g;
	[~,sorted_idx]=sort(abs(v),'descend');
	S=sort(sorted_idx(1:s))'; 
	uS=v(S);
	for l=1:NbDesc
		gS=z(S)-B(S,S)*uS;
        normAgS=norm(A(:,S)*gS);
        if normAgS==0
           break 
        end
		if t=='steepest'
			T=(norm(gS)/normAgS)^2;
		else
			T=t;
		end
		uS=uS+T*gS;
	end
	x=zeros(N,1);
	x(S)=uS;
    if strcmp(mu,'NIHT')
       if ( ( sum(S==Sold)<s ) || ( Mu*norm(A*(x-xold))^2 > 0.99*norm(x-xold)^2 ) )
           while Mu*norm(A*(x-xold))^2 > 0.99*norm(x-xold)^2
               Mu=Mu/2;
               v=xold+Mu*g;
               [~,sorted_idx]=sort(abs(v),'descend');
               S=sort(sorted_idx(1:s))'; 
               uS=v(S);
               x=zeros(N,1);
               x(S)=uS;
           end
       end
    end
    NormRes=norm(y-A*x);
	NbIter=NbIter+1;
end


%% outputs
if strcmp(Warnings,'On')
    if ( (NormRes>TolRes*normy) || isnan(NormRes) )
        disp(strcat('Warning: FHTP did not find the correct solution when using a number of iterations =', num2str(MaxNbIter), '(norm of residual =', num2str(NormRes), ' > TolRes =', num2str(TolRes), ')'));
        disp(strcat('Try using FHTP_, which normalizes the columns of A to have Euclidean norm =1'));
    end
end


end
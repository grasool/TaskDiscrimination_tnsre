% Modified from Ref 25 in the IEEE TNSRE Paper

function [S,A,loglikelihood,Sigma,chi]=fpICAMF(X,par)

% warning('off','optim:quadprog:WillRunDiffAlg');

[A,Sigma,par] = initializeMF(X,par) ;


[A,Sigma] = icaMFaem(A,Sigma,X,par) ;

% solve mean field equations and exit!
[theta,J,par] = XASigma2thetaJ(X,A,Sigma,par) ;

[S,chi,f,~,~] = ec_solver(theta,J,par) ;

loglikelihood = - f  ;

% return variables sorted according to 'energy'
[~,I]=sort((sum(A.^2,1))'.*sum(S.^2,2)) ;
I=flipud(I);
S=S(I,:); A=A(:,I);
% for i=1:par.N
%     chi(:,:,i)=squeeze(chi(I,I,i));
% end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              free energy (minus log likelihood) and derivative wrt parameters                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [A,Sigma] = icaMFaem(A,Sigma,X,par) % aem

EM_step=0;
fold = inf ;
alpha = 1.05 ; % increase factor for overrelaxed
eta = 1 / alpha ;

while EM_step < par.max_ite %& (dA_rel>tol | dSigma_rel>tol)
    
    EM_step = EM_step+1;
    
    [theta,J,par] = XASigma2thetaJ(X,A,Sigma,par) ;
    
    [S,chi,f,~,~] = ec_solver(theta,J,par) ;
    
    [Aem,Sigmaem] = dASigmaf(X,A,Sigma,S,chi,par) ;
    
    if f > fold && eta > 1
        % we took a step that decreased the likelihood
        % backtrack and take regular em step instead
        eta = 1;
        A = Aold ; Sigma = Sigmaold ; f = fold ;
        Aem = Aemold ; Sigmaem = Sigmaemold ;
    else
        % increase step size
        eta = alpha * eta ;
        Aemold = Aem;
        Sigmaemold = Sigmaem;
    end
    
    % aem update
    Aold = A;
    Sigmaold = Sigma;
    fold = f;
    A = A .* ( Aem ./ A ).^eta;
    maxratio = 2;
    minratio = 0.5; % maximim increase/decrease factors
    A = min(A,maxratio*Aold);
    A = max(A,minratio*Aold);
    Sigma = Sigma .* ( Sigmaem ./ Sigma ).^eta;
    
end % EM_step

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    expectation consistent solver                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [m,chi,G,ite,dmdm]=ec_solver(theta,J,par) % ,debug_draw)
% ec_solver formerly known as ec_fac_ep.m
% expectation consistent factorized inference
% optimized using ep-style updates

% Ole Winther, IMM, February 2005

% initialize variables
M = par.M ;
N = par.N ;

% initialize constants specific for ec_solver
minchi = 1e-7;
m=zeros(M,N);
minLam_q = 1e-5;
eta=1;
dm = zeros(M,N);
% m_r = zeros(M,N);
% v_r = zeros(M,N);

eigJ=eig(J);
mineigJ = min(eigJ);
maxeigJ = max(eigJ);
Lam_r = 1 * ( maxeigJ - mineigJ ) * ones(M,1) ;
chi = inv( diag(Lam_r) - J ) ;
Lam_r = repmat(Lam_r,[1 N]) ;

% set mean value of r-distribution to be the same as m
gam_r = zeros(M,N) ;
m_r = m;
v_r = diag(chi) ;

% make a covariance matrix for each sample
chi = repmat(chi,[1 1 N]) ; % could be made more effective with lightspeed.
%chiv = zeros(M,M,N) ; chivt = zeros(M,M,N) ;

ite=0;
dmdm=Inf;
% dmdmN=Inf*ones(N,1);
tolN = par.S_tol / N ;
I = 1:N;
while (~isempty(I) && dmdm>par.S_tol && ite<par.S_max_ite)
    
    ite=ite+1;
    indx=randperm(M);
    for sindx=1:M
        cindx=indx(sindx); par.Sindx = cindx;
        
        % find mean and variance of r
        m_r(cindx,I) = ...
            sum( shiftdim( chi(cindx,:,I) , 1 ) .* ( gam_r(:,I) + theta(:,I) ) , 1 ) ;
        v_r(cindx,I) = chi(cindx,cindx,I) ;
        
        % find Lagrange parameters of s
        Lam_s1 = 1 ./ max( minchi , v_r(cindx,I) ) ;
        gam_s1 = Lam_s1 .* m_r(cindx,I);
        
        % update lam_q
        gam_q = gam_s1 - gam_r(cindx,I) ;
        Lam_q = Lam_s1 - Lam_r(cindx,I) ;
        
        % update moments of q distribution
        [m(cindx,I),v_q] = exponential( gam_q , max( Lam_q , minLam_q ) , par ); %%% OBS hack here!!!!
        
        dm(cindx,I) = m(cindx,I) - m_r(cindx,I) ;
        
        % find Lagrange parameters of s
        Lam_s2 = 1 ./ max( minchi , v_q ) ;
        gam_s2 = Lam_s2 .* m(cindx,I) ;
        
        % update lam_r
        dgam_r = eta * ( gam_s2 - gam_s1 ) ; gam_r(cindx,I) = dgam_r + gam_r(cindx,I) ;
        dLam_r = eta * ( Lam_s2 - Lam_s1 ) ; Lam_r(cindx,I) = dLam_r + Lam_r(cindx,I) ;
        
        % update chi using Sherman-Woodbury
        switch par.ecchiupdate
            case 'parallel'
                oM = ones(M,1) ;
                kappa = dLam_r ./ ( 1 + dLam_r .* v_r(cindx,I) ) ;
                kappa=reshape(kappa,1,1,length(I));
                chiv = chi(:,cindx,I) ; chiv =  kappa(oM,:,:) .* chiv ; chiv=chiv(:,oM,:);
                chivt = chi(cindx,:,I) ; chivt=chivt(oM,:,:);
                chi(:,:,I) = chi(:,:,I) - chiv .* chivt ;
            case 'sequential'
                for j=1:length(I)
                    i = I(j) ;
                    chiv = chi(:,cindx,i) ;
                    chi(:,:,i) = chi(:,:,i) ...
                        - dLam_r(j) / ( 1 + dLam_r(j) * v_r(cindx,i) ) * (chiv * chiv');
                    % v_r(cindx,:) = chi(cindx,cindx,:)
                end
        end
        
    end % over variables - sindx
    
    dmdmN=sum(dm.*dm,1) ;
    I=find(dmdmN > tolN);
    dmdm=sum(dmdmN);
    
end

if nargout > 2 % calculate free energy per sample
    
    Lam_s = 1 ./ max( minchi , v_r ) ; gam_s = Lam_s .* m_r ;
    
    gam_q = gam_s - gam_r ; Lam_q = Lam_s - Lam_r ;
    
    par.Sindx = (1:M)';
    logZ_q = logZ0_exponential(gam_q, max( minLam_q , Lam_q ) , par ) ;
    logZ_r = 0.5 * ( N * log(2*pi) - N * par.logdet2piSigma - par.XinvSigmaX ) ;
    for i=1:N
        logZ_r = logZ_r + 0.5 * logdet( chi(:,:,i) ) + 0.5 * ( gam_r(:,i) + theta(:,i) )' * ...
            chi(:,:,i) * ( gam_r(:,i) + theta(:,i) );
    end
    logZ_s = 0.5 * ( N * log(2*pi) + sum( sum(  - log( Lam_s ) + gam_s.^2 ./ Lam_s ) ) );
    G = ( - logZ_q - logZ_r + logZ_s ) / N ;
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                        parameter conversion                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [theta,J,par] = XASigma2thetaJ(X,A,Sigma,par)

% invert Sigma
switch size(Sigma,1)*size(Sigma,2)
    case 1
        invSigma = 1 / Sigma;
        par.logdet2piSigma = par.D * log( 2 * pi * Sigma ) ;
    case par.D
        invSigma = diag(1./Sigma);
        par.logdet2piSigma = sum( log( 2 * pi * Sigma ) ) ;
    case par.D^2
        invSigma = inv(Sigma);
        par.logdet2piSigma = logdet( 2 * pi * Sigma ) ;
end

% calculate external field and coupling matrix
theta = A' * invSigma * X ;
J = - A' * invSigma * A ;

par.XinvSigmaX = sum( sum( X .* ( invSigma * X ) ) ) ;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     parameter derivatives                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [A,Sigma] = dASigmaf(X,A,Sigma,S,chi,par)

M = par.M ;

tracechi = zeros(size(chi,1), size(chi,2));
traceSS = zeros(size(tracechi));

for i=1:M  % set up matrices for finding derivatives
    for j=i:M
        tracechi(i,j)=sum(chi(i,j,:));
        tracechi(j,i)=tracechi(i,j);
        traceSS(i,j)=sum(chi(i,j,:))+sum(S(i,:)'.*S(j,:)');
        traceSS(j,i)=traceSS(i,j);
    end;
end;

invSigma = inv(Sigma) ;

A = A_positive_aem(traceSS,X * S',invSigma,A) ;

Sigma = ( sum(sum((X-A*S).^2)) + sum(sum((A*tracechi).*A)) ) / ( par.N * par.D ) ;

end


function [A] = A_positive_aem(traceSS, XSt, invSigma, A)

Atol = 1e-6;
KT_max_ite=100;
A=A+10^-3*(A<eps);
sizeA = size(A,1) * size(A,2) ;
invSigmaXSt = invSigma * XSt ;
amp=invSigmaXSt./(invSigma*A*traceSS);
f = sum( sum( traceSS .* ( A' * invSigma * A ) ) ) - 2 * sum( sum( A .* ( invSigma * XSt ) ) ) ;
Aneg=any(any(amp<0));
KT_ite=0; Aerror = inf ;
alpha = 2; %1.1 ;
eta = 1 ;
%maxratio=4; minratio=0.25; % maximum increase/decrease factors
while ~Aneg && KT_ite<KT_max_ite && Aerror > Atol
    KT_ite=KT_ite+1;
    Aold = A ;
    A=A.*amp.^eta;
    %A=min(A,maxratio*Aold); A=max(A,minratio*Aold);
    fnew = sum( sum( traceSS .* ( A' * invSigma * A ) ) ) ...
        - 2 * sum( sum( A .* ( invSigma * XSt ) ) ) ;
    if f < fnew
        eta = 1 ;
        A = Aold .* amp ;
        %A=min(A,maxratio*Aold); A=max(A,minratio*Aold);
        f = sum( sum( traceSS .* ( A' * invSigma * A ) ) ) ...
            - 2 * sum( sum( A .* ( invSigma * XSt ) ) ) ;
    else
        eta = alpha * eta ;
        f = fnew ;
    end
    amp=invSigmaXSt./(invSigma*A*traceSS);
    Aneg=any(any(amp<0));
    Aerror = sum(sum(abs(A-Aold))) / sizeA ;
end
%KT_ite
if Aneg % use quadratic programming instead - doesn't work for Sigma full
    M=size(traceSS,1);
    D=size(XSt,1);
    options=optimset('Display','off','TolX',10^5*eps,'TolFun',10^5*eps);
    for i=1:D
        B=quadprog(traceSS,-XSt(i,:)',[],[],[],[],zeros(M,1),[],A(i,:)',options);
        A(i,:)=B';
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     initialization                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,Sigma,par] = initializeMF(X,par)

% function handle to mean field solver
% par.mf_solver = str2func( sprintf('%s_solver',par.solver) ) ;
par.Sprior='exponential'; par.Aprior='positive';

% number of inputs, examples
[ par.D , par.N ] = size(X) ;
par.M = par.D;
A = par.A_init;
[ par.D , par.M ] = size( A );
par.Asize = par.D * par.M ;
S=zeros(par.M,par.N);
% par.Smeanf = str2func(par.Sprior) ;
% par.logZ0f = str2func( sprintf('logZ0_%s',par.Sprior) ) ;
par.Sigmasize = 1 ;
Sigmascale = 1 ;
Sigma = Sigmascale * sum(sum( ( X - A * S ).^2 ) ) / (par.D*par.N) ;

par.max_ite = 50;% end
par.S_tol = 1e-10; % end
par.S_max_ite = 100;% end
% chi update specific for ec_solver
try
    par.ecchiupdate ;
catch
    if par.M > 10
        par.ecchiupdate = 'sequential' ;
    else
        par.ecchiupdate = 'parallel' ;
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       mean (S) functions                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f,df] = exponential(gamma,lambda,~)
eta=1;
erfclimit=-35;
%minlambda=10^-4;
%lambda=lambda.*(lambda>minlambda)+minlambda.*(lambda<=minlambda);
xi=(gamma-eta)./sqrt(lambda);
cc=(xi>erfclimit);
%i=find(cc==0);
%if ~isempty(i)
%    cc, pause
%end
xi1=xi.*cc;
epfrac=exp(-(xi1.^2)/2)./(Phi(xi1)*sqrt(2*pi));
f=cc.*(xi1+epfrac)./sqrt(lambda);            % need to go to higher order to get asymptotics right
if nargout > 1
    df=cc.*(1./lambda+f.*(xi1./sqrt(lambda)-f)); % need to go to higher order to get asymptotics right -fix at some point!!!
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       logZ0 functions                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [logZ0] = logZ0_exponential(gamma,lambda,~)
eta=1;
erfclimit=-35;
%minlambda=10^-4;
%lambda=lambda.*(lambda>minlambda)+minlambda.*(lambda<=minlambda);
xi=(gamma-eta)./sqrt(lambda);
cc=(xi>erfclimit);
xi1=xi.*cc;
logZ0terms=cc.*(log(Phi(xi1))+0.5*log(2*pi)+0.5*xi1.^2)-(1-cc).*log(abs(xi)+cc)+log(eta)-0.5*log(lambda);
logZ0=sum(sum(logZ0terms));

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                different math help function                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = logdet(A)
% log(det(A)) where A is positive-definite.
% This is faster and more stable than using log(det(A)).

[U,p] = chol(A);
if p
    y=0; % if not positive definite return 0 , could also be a log(eps)
else
    y = 2*sum(log(diag(U)));
end
end


% Phi (error) function
function y=Phi(x)
% Phi(x) = int_-infty^x Dz
z=abs(x/sqrt(2));
t=1.0./(1.0+0.5*z);
y=0.5*t.*exp(-z.*z-1.26551223+t.*(1.00002368+...
    t.*(0.37409196+t.*(0.09678418+t.*(-0.18628806+t.*(0.27886807+...
    t.*(-1.13520398+t.*(1.48851587+t.*(-0.82215223+t.*0.17087277)))))))));
y=(x<=0.0).*y+(x>0).*(1.0-y);

end


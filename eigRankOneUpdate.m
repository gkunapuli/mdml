%  EIGRANKONEUPDATE   Compute eigendecomposition of a rank-1 perturbation
%  of a matrix with known eigendecomposition. Given a matrix with known 
%  eigendecomposition, A = V*E*V', EIGRANKONEUPDATE efficiently calculates  
%  the eigendecomposition of the perturbed matrix, Ap = (A + rho*u*u'), 
%  where rho is some scalar and u is a vector. 
%
%  NOTE: The algorithm takes t = V'*u as the input rather than u directly. 
%  i.e., we assume that Ap is expressed as Ap = V*(diag(E) + rho*t*t')*V'
%
%  [W, F] = eigRankOneUpdate(V, E, t, rho) returns the eigendecomposition 
%  of Ap = W*F*W' using eigComputeRational to compute the eigenvalues from 
%  the secular equation. If a reduced eigen-decompositon is provided, a 
%  reduced update is calculated i.e., when V is N x r and E is r x 1, with 
%  r < N. This is a rank-preserving update. 
%
%  For details on computing eigenvectors, see 
%  "Rank-One Modification of the Symmetric Eigenproblem", 
%  J. R. Bunch, C. P. Nielsen & D. C. Sorensen, 
%  Numer. Math., 31, 31-48 (1978). 
%
%  For further details on stability, see 
%  "A Stable and Efficient Algorithm for the Rank-one Modification of the 
%   Symmetric Eigenvalue Problem", M. Gu & S. C. Eisenstat, 
%   Research Report YALEU/DCS/RR-916 (1992). 
%
%  For implementation details, see 
%  "Matrix Algorithms Volume II: Eigensystems", 
%  G. W. Stewart, Chapter 3.1, SIAM (2001).
%
%  version 6.5
%  Gautam Kunapuli (gkunapuli@gmail.com)
%  February 21, 2018
%
% This program comes with ABSOLUTELY NO WARRANTY; See the GNU General Public
% License for more details. This is free software, and you are welcome to 
% modify or redistribute it.

function [W, F, stats] = eigRankOneUpdate(V, E, t, rho, acc)

% Set an internal accuracy
if nargin < 5
    acc = 1e-12;
end

% Constant to test if fractions of t can be set to zero
Gt = 10; 

% Start timer
eigTimer = tic;

% Get the eigenvalues and clean them up
[lambda, I] = sort(E, 'ascend'); % TODO: Is this necessary?
V = V(:, I);
N = length(lambda);
tNorm = norm(t);

% Round to accuracy so that we can detect unique eigen-values properly
maxLambda = max(abs(lambda)); 
tol = N * acc * sqrt(maxLambda);
lambda(abs(lambda) < tol) = 0;

% Now, get the repeated eigenvalues, and replace the corresponding
% eigenvectors V(lambda) with V(lambda)*H, where H is a Householder matrix
% designed to introduce sparsity into the linear combination space i.e.,
% make t look like [a, 0, 0, b, 0, c, 0, 0, 0, ...., 0]'
[uniqueEigs, ~, uIndex] = unique(lambda);
d = length(uniqueEigs);
% multiplicity = sum(abs(ones(N, 1)*uniqueEigs' - lambda*ones(1, d)) <= acc);

for i = 1:d
    % Determine where the array starts and ends
    inds = find(uIndex == i);
    multiplicity = length(inds);
    startEig = inds(1);
    endEig = inds(end);
      
    % If this is a multiple eigenvalue, zero out all the components of t
    % but one corresponding to the one closest to the next highest (if rho
    % > 0, lowest if rho < 0)
    if multiplicity > 1
        v = t(startEig:endEig);
        
        if rho < 0
            t(startEig:endEig) = [-norm(v); zeros(multiplicity - 1, 1)];
            v(1) = v(1) + norm(v);
        else
            t(startEig:endEig) = [zeros(multiplicity - 1, 1); -norm(v)];
            v(multiplicity) = v(multiplicity) + norm(v);
        end
        
        % Sometimes, some values slip through despite the accuracy check
        % above. Make sure that the eigenvector calulation remians 
        % numerically backward stable
        if norm(v) > Gt * (maxLambda + tNorm^2) * acc
            V(:, startEig:endEig) = V(:, startEig:endEig)...
                                - 2*V(:, startEig:endEig)*(v*v')/norm(v)^2;
        end
    end
end

% Remove very small values of t
tI = find( abs(t) <= Gt * (maxLambda/tNorm + tNorm) * acc );
t(tI) = 0;

% Get the indices of the non-zero values of t
tI = setdiff(1:N, tI);
Nt = length(tI);
Ebar = lambda(tI);
tBar = t(tI);

% Compute the all the eigenvalues, and the eigenvectors corresponding to
% the components t(i) <> 0. The components for t(i) == 0 remain unchanged
mu = zeros(Nt, 1);
avgIters = 0;
avgEigValTime = 0;

if rho > 0
    for i = 1:Nt
        [mu(i), iter, time] =...
                    eigComputeRational(i, Ebar, tBar, rho, acc);

        avgIters = avgIters + iter;
        avgEigValTime = avgEigValTime + time;
    end
else
    Ef = -flipud(Ebar);
    tf = flipud(tBar);

    for i = 1:Nt
        [mu(i), iter, time] = eigComputeRational(Nt-i+1, Ef, tf, -rho, acc);

        avgIters = avgIters + iter;
        avgEigValTime = avgEigValTime + time;
    end
end
Fbar = Ebar + rho*mu;

% Clean up Fbar
Fbar(abs(Fbar) < acc) = 0;

% Insert the new eigenvalues of the perturbed matrix 
F = lambda;
F(tI) = Fbar;

% Correct the entries of t as follows to make the eigenvector computation
% more stable:
%              ____________________________________________________________
%             / i-1                   N-1  
%            /  ___  ( F(j) - E(j) )  ___   ( F(j) - E(j) )
% t(i) =    /   | |  ---------------  | |  ---------------- ( F(N) - E(i) )
%       \  /    | |  ( E(j) - E(i) )  | |  ( E(j+1) - E(i) ) 
%        \/    j = 1                 j = i 
%
% where the symbol with the #s represents PI, the product operator.

tBarNum = repmat(Fbar', Nt, 1) - repmat(Ebar, 1, Nt);
tBarDen = repmat(Ebar', Nt, 1) - repmat(Ebar, 1, Nt) + eye(Nt);

tBar = sign(tBar) .* sqrt(sign(rho) * prod(tBarNum ./ tBarDen, 2));

% Compute the eigenvectors using the corrected entries of t
W = V;
for i = 1:Nt
    W(:, tI(i)) = V(:, tI) * (tBar ./ (Fbar(i) - Ebar));
end
try
    W = normc(W);
catch
    disp(' ');
end

I = find(sum(isnan(W)) > 0);
W(:, I) = V(:, I);

% Sort the eigen values and eigenvectors to return
[F, I] = sort(F, 'ascend');
W = W(:, I);

% Collect some stats
Nt = length(tI);
stats.totalTime = toc(eigTimer);
stats.avgEigValTime = avgEigValTime / Nt;
stats.avgIters = avgIters / Nt;
stats.numEvals = Nt;
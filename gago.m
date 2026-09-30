function [xbest, fbest, out] = gago(fun, x0, opts)
% GAGO  Improved Geometry-Aware Group Optimization (multiresolution version)
%
%   [xbest, fbest, out] = gago(fun, x0, opts)
%
% Same framework as the paper (translation group acting on R^d, symmetric
% generating set S = {+-e_i}, words, Cayley-graph geodesics, adaptive
% radius), with the following algorithmic changes that make it much more
% efficient under a fixed evaluation budget:
%
%  1. Multiresolution subgroup chain (paper, Sec. 8.3, Eq. 245):
%     G_1 = d1*Z^d  <  G_2 = (d1/2)*Z^d  <  ...   The resolution delta is
%     halved whenever the current point is L-word-stationary, so the search
%     is coarse at first and refines automatically. This removes the
%     sensitivity to a single fixed delta.
%  2. Endpoint-only word evaluation. The acceptance rule (Eq. 94-98) only
%     needs f at the endpoint, so a word of length l costs ONE evaluation,
%     not l (the v1 code evaluated every intermediate vertex).
%  3. Geodesic extension. After an improving word w, the geodesic in the
%     same direction is extended by doubling multiplicity (w^2, w^4, ...)
%     while f keeps decreasing (adaptive-radius growth along the geodesic).
%  4. Pattern memory. The accumulated successful displacement (rounded to the
%     current lattice) is proposed as a long word when generators fail.
%  5. Restart + polish. Phase 1 uses random restarts on the coarse chain
%     to escape local minima; phase 2 polishes the best point down to a
%     very fine resolution.
%
% INPUTS
%   fun  : handle, f = fun(x), x is 1-by-d
%   x0   : 1-by-d initial point
%   opts : struct
%       .lb, .ub    bounds (scalar or 1-by-d)              (REQUIRED)
%       .delta0     initial resolution, fraction of range   (0.15)
%       .dexp       exploration stop resolution (relative)  (1e-2)
%       .dfin       final polish resolution (relative)      (1e-9)
%       .L          max random word length                  (4)
%       .M          random words per stalled iteration      (2*d)
%       .split      fraction of budget for phase 1          (0.7)
%       .NFE_max    evaluation budget                       (5000)
%       .fstar,.tol stop when |f-fstar|<=tol                (-Inf, 1e-8)
%       .use_words, .use_ext, .use_memory  toggles (true)   (for ablation)
%       .refine, .restart, .polish  toggles (true) for ablation:
%                   refine  = halve the resolution at stationarity,
%                   restart = random restarts in phase 1,
%                   polish  = final fine-resolution phase 2
%       .round_memory  round the pattern word to the current lattice so
%                   that every candidate is an exact element of the
%                   generated group (false by default; slightly weaker
%                   on curved valleys, but makes Prop. 3 of the paper
%                   apply verbatim)
%
% OUTPUTS
%   xbest, fbest, out.fhist / out.nfehist (best-so-far at improvements),
%   out.NFE (evaluations used).

if nargin < 3, opts = struct(); end
opts = set_default(opts, 'delta0', 0.15);
opts = set_default(opts, 'dexp',   1e-2);
opts = set_default(opts, 'dfin',   1e-9);
opts = set_default(opts, 'L',      4);
opts = set_default(opts, 'split',  0.7);
opts = set_default(opts, 'NFE_max', 5000);
opts = set_default(opts, 'fstar',  -Inf);
opts = set_default(opts, 'tol',    1e-8);
opts = set_default(opts, 'use_words',  true);
opts = set_default(opts, 'use_ext',    true);
opts = set_default(opts, 'use_memory', true);
opts = set_default(opts, 'round_memory', true);
opts = set_default(opts, 'refine',  true);
opts = set_default(opts, 'restart', true);
opts = set_default(opts, 'polish',  true);
if ~isfield(opts,'lb') || ~isfield(opts,'ub')
    error('gago:bounds', 'opts.lb and opts.ub are required.');
end

x0 = x0(:)';
d  = numel(x0);
lb = opts.lb; ub = opts.ub;
if isscalar(lb), lb = lb*ones(1,d); end
if isscalar(ub), ub = ub*ones(1,d); end
span = ub - lb;
opts = set_default(opts, 'M', 2*d);
M = opts.M;
N = opts.NFE_max;

S = [eye(d); -eye(d)];            % symmetric generating set (2d x d)

% shared state (visible to nested functions)
NFE = 0; fbest = inf; xbest = x0;
fhist = []; nfehist = [];

% ---------------- main procedure ----------------
x  = x0; fx = evalf(x);

if ~opts.restart
    % single descent from x0 down to the final resolution
    descend(x, fx, opts.delta0, opts.dfin, N);
else
    lim1 = N;  dm = opts.dfin;
    if opts.polish
        lim1 = floor(opts.split * N);   % phase 1 budget
        dm   = opts.dexp;               % coarse exploration tolerance
    end
    while NFE < lim1 && ~reached()
        [x, fx] = descend(x, fx, opts.delta0, dm, lim1);
        if NFE >= lim1 || reached(), break; end
        x  = lb + span .* rand(1,d);    % restart on a fresh orbit
        fx = evalf(x);
    end
    if opts.polish && ~reached() && NFE < N
        % phase 2: polish the best point on finer and finer lattices
        descend(xbest, fbest, 4*max(opts.dexp, 1e-3), opts.dfin, N);
    end
end

out = struct('fhist', fhist, 'nfehist', nfehist, 'NFE', NFE);

% =====================================================================
% nested helpers
% =====================================================================
    function tf = reached()
        tf = abs(fbest - opts.fstar) <= opts.tol;
    end

    function v = evalf(y)
        y = min(max(y, lb), ub);
        v = fun(y);
        NFE = NFE + 1;
        if v < fbest
            fbest = v; xbest = y;
            fhist   = [fhist; v];     %#ok<AGROW>
            nfehist = [nfehist; NFE]; %#ok<AGROW>
        end
    end

    function y = clip(y)
        y = min(max(y, lb), ub);
    end

    function [x, fx] = descend(x, fx, rel, dmin, limit)
        acc = zeros(1,d);             % pattern memory
        while NFE < limit && rel > dmin && ~reached()
            step = []; bfv = fx; bx = [];

            % --- 1. local generator search (all 2d elementary moves)
            for j = 1:2*d
                if NFE >= limit, break; end
                w = rel * span .* S(j,:);
                y = clip(x + w);
                if isequal(y, x), continue; end
                v = evalf(y);
                if v < bfv, bfv = v; bx = y; step = w; end
            end

            % --- 2. word search (only if no generator improves)
            if isempty(step) && NFE < limit
                cands = {};
                if opts.use_memory && any(acc)
                    cands = {acc, 0.5*acc};
                    if opts.round_memory
                        % round to the current lattice so every candidate
                        % is an exact element of the generated group
                        h = rel * span;
                        cands = {round(acc ./ h) .* h, round(0.5*acc ./ h) .* h};
                    end
                end
                if opts.use_words
                    for c = 1:M
                        l = randi([2, opts.L]);
                        idx = randi(2*d, l, 1);
                        cands{end+1} = rel * span .* sum(S(idx,:), 1); %#ok<AGROW>
                    end
                end
                for c = 1:numel(cands)
                    if NFE >= limit, break; end
                    w = cands{c};
                    y = clip(x + w);
                    if isequal(y, x), continue; end
                    v = evalf(y);
                    if v < bfv, bfv = v; bx = y; step = w; end
                end
            end

            if ~isempty(step)
                % --- 3. accept, then extend along the geodesic (doubling)
                x = bx; fx = bfv; m = 1;
                if opts.use_ext
                    while NFE < limit
                        y = clip(x + step*m);
                        if isequal(y, x), break; end
                        v = evalf(y);
                        if v < fx
                            x = y; fx = v; m = 2*m;
                        else
                            break;
                        end
                    end
                end
                if opts.use_memory
                    acc = 0.5*acc + step*m;
                else
                    acc = zeros(1,d);
                end
            else
                % --- 4. L-word-stationary: refine to the next subgroup
                if ~opts.refine, break; end
                rel = rel * 0.5;
                acc = zeros(1,d);
            end
        end
    end
end

function opts = set_default(opts, field, val)
if ~isfield(opts, field) || isempty(opts.(field))
    opts.(field) = val;
end
end

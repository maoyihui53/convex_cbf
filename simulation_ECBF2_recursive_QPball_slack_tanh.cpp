#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// Simple 2D linear algebra
// ============================================================

struct Vec2 {
    double x;
    double y;
    Vec2() : x(0.0), y(0.0) {}
    Vec2(double x_, double y_) : x(x_), y(y_) {}
};

Vec2 operator+(const Vec2& a, const Vec2& b) { return Vec2(a.x + b.x, a.y + b.y); }
Vec2 operator-(const Vec2& a, const Vec2& b) { return Vec2(a.x - b.x, a.y - b.y); }
Vec2 operator*(double s, const Vec2& v)      { return Vec2(s * v.x, s * v.y); }
Vec2 operator*(const Vec2& v, double s)      { return Vec2(s * v.x, s * v.y); }
Vec2 operator/(const Vec2& v, double s)      { return Vec2(v.x / s, v.y / s); }

Vec2& operator+=(Vec2& a, const Vec2& b) { a.x += b.x; a.y += b.y; return a; }

double dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
double norm(const Vec2& v)               { return std::sqrt(dot(v, v)); }

// ============================================================
// 2x2 matrix (for Hessians)
// ============================================================

struct Mat2 {
    double a11, a12, a21, a22;
    Mat2() : a11(0), a12(0), a21(0), a22(0) {}
};

Mat2 outer(const Vec2& u, const Vec2& v) {
    Mat2 M;
    M.a11 = u.x * v.x; M.a12 = u.x * v.y;
    M.a21 = u.y * v.x; M.a22 = u.y * v.y;
    return M;
}

Mat2 operator+(const Mat2& A, const Mat2& B) {
    Mat2 C;
    C.a11 = A.a11 + B.a11; C.a12 = A.a12 + B.a12;
    C.a21 = A.a21 + B.a21; C.a22 = A.a22 + B.a22;
    return C;
}

Mat2& operator+=(Mat2& A, const Mat2& B) {
    A.a11 += B.a11; A.a12 += B.a12;
    A.a21 += B.a21; A.a22 += B.a22;
    return A;
}

Mat2 operator*(double s, const Mat2& A) {
    Mat2 C;
    C.a11 = s * A.a11; C.a12 = s * A.a12;
    C.a21 = s * A.a21; C.a22 = s * A.a22;
    return C;
}

// v^T M v
double quadform(const Mat2& M, const Vec2& v) {
    double Mv_x = M.a11 * v.x + M.a12 * v.y;
    double Mv_y = M.a21 * v.x + M.a22 * v.y;
    return v.x * Mv_x + v.y * Mv_y;
}

// ============================================================
// State: second order system (x, v)
// ============================================================

struct State {
    Vec2 x;   // position
    Vec2 v;   // velocity
};

State operator+(const State& a, const State& b) { return State{a.x + b.x, a.v + b.v}; }
State operator*(double s, const State& a)       { return State{s * a.x, s * a.v}; }

// ============================================================
// Obstacle representations
// ============================================================

// Polygon given by vertices (CCW)
struct PolygonObstacle {
    std::vector<Vec2> vertices;
};

// One edge in normal+point form
struct Edge {
    Vec2 n;  // outward unit normal (points outside obstacle)
    Vec2 p;  // a point on the edge
};

// Polygon given by list of edges (normal+point)
struct NormalPointObstacle {
    std::vector<Edge> edges;
};

// ============================================================
// Parameters
// ============================================================

struct Params {
    Vec2 w;   // agent half-size (rectangle) for Minkowski inflation

    // Obstacles (can use both)
    std::vector<PolygonObstacle>     polyObs;
    std::vector<NormalPointObstacle> npObs;

    // Goal/path
    Vec2 xgoal;
    std::vector<Vec2> refPath;
    bool   useRefPath = false;
    double wpTol      = 0.2;

    // Desired controller gains
    double Kp   = 0.5;
    double Kv   = 1.0;
    double umax = 1.0; // desired input saturation (for ud only)

    // Safety filter on/off
    bool filterOn = true;

    // Simulation step size
    double dt = 0.01;

    // Smooth CBF composition parameters (paper uses kappa,b)
    double kappa = 10.0;
    double b     = std::log(2.0);

    // Continuous-time relative-degree-2 ECBF gains:
    // Enforce: h¨ + k1 h˙ + k0 h >= 0
    double cbf_k0 = 4.0;   // >0
    double cbf_k1 = 4.0;   // >0

    // Actual actuator bound for the filtered u
    double umax_real = 3.0;

    // Numerical stabilization for exp(kappa*hi): apply monotone scaling gamma(hi)=tanh(beta*hi)
    // before exponentiation. beta>0 controls how quickly tanh saturates.
    double tanh_beta = 1.0;

    // Soft-constraint (slack) penalty used only when hard constraint is infeasible under ||u||<=umax_real.
    // Larger => try harder to satisfy safety constraint even if it requires moving away from ud.
    double slack_weight = 1000.0;

    // Whether to use the general recursive (Eq. 36) evaluation
    bool useRecursiveComposition = true;
};

// ============================================================
// Dynamics: xdot = v, vdot = u
// ============================================================

State dynamics(const State& z, const Vec2& u) {
    State dzdt;
    dzdt.x = z.v;
    dzdt.v = u;
    return dzdt;
}

// ============================================================
// Primitive CBFs for all obstacle edges: hi(x) and gradhi(x)
// plus grouping info per obstacle
// ============================================================

void CBFsAll(const Vec2& x, const Params& par,
             std::vector<double>& hi_all,
             std::vector<Vec2>&   gradhi_all,
             std::vector<int>&    obs_offset,
             std::vector<int>&    obs_count)
{
    hi_all.clear();
    gradhi_all.clear();
    obs_offset.clear();
    obs_count.clear();

    // 1) Vertex polygons
    for (const auto& poly : par.polyObs) {
        int M = static_cast<int>(poly.vertices.size());
        if (M < 2) continue;

        int start = static_cast<int>(hi_all.size());
        int edgesAdded = 0;

        for (int j = 0; j < M; ++j) {
            const Vec2& vj = poly.vertices[j];
            const Vec2& vk = poly.vertices[(j + 1) % M];

            Vec2 e(vk.x - vj.x, vk.y - vj.y);
            double elen = norm(e);
            if (elen < 1e-12) continue;

            // For CCW vertices, outward normal is (e.y, -e.x)/|e|
            Vec2 n(e.y / elen, -e.x / elen);

            // Minkowski inflation radius for axis-aligned rectangle agent half-size w
            double r = std::fabs(n.x) * par.w.x + std::fabs(n.y) * par.w.y;

            // hi = n^T (x - vj) - r
            Vec2 diff(x.x - vj.x, x.y - vj.y);
            double h = dot(n, diff) - r;

            hi_all.push_back(h);
            gradhi_all.push_back(n);
            ++edgesAdded;
        }

        if (edgesAdded > 0) {
            obs_offset.push_back(start);
            obs_count.push_back(edgesAdded);
        }
    }

    // 2) Normal+point polygons
    for (const auto& npObs : par.npObs) {
        int M = static_cast<int>(npObs.edges.size());
        if (M <= 0) continue;

        int start = static_cast<int>(hi_all.size());
        int edgesAdded = 0;

        for (const auto& e : npObs.edges) {
            Vec2 n = e.n;
            Vec2 p = e.p;

            double r = std::fabs(n.x) * par.w.x + std::fabs(n.y) * par.w.y;
            Vec2 diff(x.x - p.x, x.y - p.y);
            double h = dot(n, diff) - r;

            hi_all.push_back(h);
            gradhi_all.push_back(n);
            ++edgesAdded;
        }

        if (edgesAdded > 0) {
            obs_offset.push_back(start);
            obs_count.push_back(edgesAdded);
        }
    }
}

// ============================================================
// Nonsmooth logic CBF (for comparison): per obstacle OR=max, across obstacles AND=min
// ============================================================

double hlogic_nonsmooth_all(const std::vector<double>& hi_all,
                            const std::vector<int>& obs_offset,
                            const std::vector<int>& obs_count)
{
    int No = static_cast<int>(obs_offset.size());
    if (No == 0) return 1e6;

    std::vector<double> ho(No, 1e6);
    for (int k = 0; k < No; ++k) {
        int start = obs_offset[k];
        int cnt   = obs_count[k];
        if (cnt <= 0) continue;

        double h_or = hi_all[start];
        for (int j = 1; j < cnt; ++j) {
            h_or = std::max(h_or, hi_all[start + j]);
        }
        ho[k] = h_or;
    }

    double h = ho[0];
    for (int k = 1; k < No; ++k) {
        h = std::min(h, ho[k]);
    }
    return h;
}

// ============================================================
// General multi-level recursive composition of a single smooth CBF
// matching Eq. (36) in the paper, but extended to return gradient+Hessian.
//
// Paper recursion (in H-space) uses:
//  H_i^0(x)   = exp(kappa * h_i(x))
//  H_i^ell(x) = sum_{j in J_i^ell} H_j^{ell-1}(x)                 if union level
//  H_i^ell(x) = 1 / sum_{j in J_i^ell} (1 / H_j^{ell-1}(x))       if intersection level
//  h(x)       = (1/kappa) * ln(H_1^M(x)) - b/kappa
//
// We compute for each node also:
//  g = ∇H,  S = ∇^2 H, then obtain ∇h and ∇^2 h from log(H).
// ============================================================

// ============================================================
// Numerically stable monotone scaling for primitive constraints:
//   gamma(hi) = tanh(beta * hi)
// This is suggested in the paper (Remark on scaling hi to avoid exp overflow)
// and keeps exp(kappa*gamma(hi)) bounded in [exp(-kappa), exp(kappa)].
// ============================================================
static inline void tanh_gamma(double hi, double beta,
                              double& g0, double& g1, double& g2)
{
    // g0 = tanh(beta*hi)
    // g1 = d/dhi tanh(beta*hi) = beta * sech^2(beta*hi)
    // g2 = d^2/dhi^2 tanh(beta*hi) = -2 beta^2 tanh(beta*hi) sech^2(beta*hi)
    double y = beta * hi;
    double t = std::tanh(y);
    double sech2 = 1.0 - t * t;
    g0 = t;
    g1 = beta * sech2;
    g2 = -2.0 * beta * beta * t * sech2;
}


enum class LevelType { Union, Intersection };

struct SpecLevel {
    LevelType type;
    // For each output node i at this level: J[i] gives indices of children in previous level.
    std::vector<std::vector<int>> J;
};

struct ComposeSpec {
    std::vector<SpecLevel> levels; // size M
};

// Node value for H, its gradient, and Hessian.
struct NodeVal {
    double H;
    Vec2   g; // grad H
    Mat2   S; // Hess H
};

static inline NodeVal make_leaf(double hi,
                                const Vec2& grad_hi,
                                const Mat2& hess_hi,
                                double kappa,
                                double tanh_beta)
{
    // H = exp(kappa * gamma(hi)), gamma(hi)=tanh(beta*hi)
    // g = dH = H * (kappa*gamma'(hi)) * grad_hi
    // S = Hess(H) = H * [ ( (kappa*gamma')^2 + (kappa*gamma'') ) * (grad_hi grad_hi^T)
    //                    + (kappa*gamma') * hess_hi ]
    NodeVal out;

    // Ensure beta>0
    if (!(tanh_beta > 0.0)) tanh_beta = 1.0;

    double g0, g1, g2;
    tanh_gamma(hi, tanh_beta, g0, g1, g2);

    double phi   = kappa * g0;
    double phi1  = kappa * g1; // d phi / d hi
    double phi2  = kappa * g2; // d^2 phi / d hi^2

    out.H = std::exp(phi);

    // g = H * phi1 * grad_hi
    out.g.x = out.H * phi1 * grad_hi.x;
    out.g.y = out.H * phi1 * grad_hi.y;

    // S = H * [ (phi1^2 + phi2) * (grad_hi grad_hi^T) + phi1 * hess_hi ]
    Mat2 gg = outer(grad_hi, grad_hi);
    out.S = ((phi1 * phi1 + phi2) * out.H) * gg;

    out.S.a11 += out.H * phi1 * hess_hi.a11;
    out.S.a12 += out.H * phi1 * hess_hi.a12;
    out.S.a21 += out.H * phi1 * hess_hi.a21;
    out.S.a22 += out.H * phi1 * hess_hi.a22;

    return out;
}

static inline NodeVal combine_union(const std::vector<NodeVal>& prev,
                                    const std::vector<int>& children)
{
    // Union layer: H = sum_j Hj, g = sum_j gj, S = sum_j Sj
    // For numerical robustness we compute H using a max-rescaling.
    NodeVal out;
    out.H = 0.0;
    out.g = Vec2(0.0, 0.0);
    out.S = Mat2();

    double maxH = 0.0;
    for (int idx : children) {
        maxH = std::max(maxH, prev[idx].H);
    }

    if (maxH <= 0.0) {
        // Degenerate: all zero
        return out;
    }

    double sumScaled = 0.0;
    for (int idx : children) {
        const NodeVal& c = prev[idx];
        sumScaled += c.H / maxH;
        out.g += c.g;
        out.S += c.S;
    }
    out.H = maxH * sumScaled;
    return out;
}

static inline NodeVal combine_intersection(const std::vector<NodeVal>& prev,
                                           const std::vector<int>& children)
{
    // H = 1/Q, where Q = sum_j 1/Hj
    // gQ = sum_j d(1/Hj) = sum_j (-1/Hj^2) gj
    // SQ = sum_j Hess(1/Hj) = sum_j [ 2/Hj^3 (gj gj^T) - 1/Hj^2 Sj ]
    // then:
    // g = dH = -(1/Q^2) gQ
    // S = 2/Q^3 (gQ gQ^T) - 1/Q^2 SQ
    NodeVal out;
    out.H = 0.0;
    out.g = Vec2(0.0, 0.0);
    out.S = Mat2();

    double Q = 0.0;
    Vec2  gQ(0.0, 0.0);
    Mat2  SQ;

    const double epsH = 1e-300;

    for (int idx : children) {
        const NodeVal& c = prev[idx];
        double Hj = std::max(c.H, epsH);

        double invH  = 1.0 / Hj;
        double invH2 = invH * invH;
        double invH3 = invH2 * invH;

        Q += invH;

        // gQ += -(1/H^2) g
        gQ.x += -invH2 * c.g.x;
        gQ.y += -invH2 * c.g.y;

        // SQ += 2/H^3 (g g^T) - 1/H^2 S
        Mat2 gg = outer(c.g, c.g);
        SQ += (2.0 * invH3) * gg;
        SQ += (-invH2) * c.S;
    }

    if (Q < 1e-300) {
        // Should not happen unless everything is huge (or numeric issue)
        out.H = 1e300;
        return out;
    }

    out.H = 1.0 / Q;

    // g = -(1/Q^2) gQ
    double invQ2 = 1.0 / (Q * Q);
    out.g.x = -invQ2 * gQ.x;
    out.g.y = -invQ2 * gQ.y;

    // S = 2/Q^3 gQ gQ^T - 1/Q^2 SQ
    double invQ3 = invQ2 / Q;
    Mat2 gQgQT = outer(gQ, gQ);
    Mat2 term1 = (2.0 * invQ3) * gQgQT;
    Mat2 term2 = (-invQ2) * SQ;
    out.S = term1 + term2;

    return out;
}

static inline void build_default_spec_from_obstacles(const std::vector<int>& obs_offset,
                                                     const std::vector<int>& obs_count,
                                                     ComposeSpec& spec)
{
    spec.levels.clear();
    const int No = static_cast<int>(obs_offset.size());
    if (No == 0) return;

    // Level 1: union inside each obstacle (each obstacle corresponds to one group)
    SpecLevel L1;
    L1.type = LevelType::Union;
    L1.J.resize(No);

    for (int k = 0; k < No; ++k) {
        int start = obs_offset[k];
        int cnt   = obs_count[k];
        L1.J[k].reserve(std::max(cnt, 0));
        for (int j = 0; j < cnt; ++j) {
            L1.J[k].push_back(start + j);
        }
    }

    // Level 2: intersection across obstacles (one group containing all previous nodes)
    SpecLevel L2;
    L2.type = LevelType::Intersection;
    L2.J.resize(1);
    L2.J[0].reserve(No);
    for (int k = 0; k < No; ++k) L2.J[0].push_back(k);

    spec.levels.push_back(L1);
    spec.levels.push_back(L2);
}

void CBF_eval_recursive_withHess(const Vec2& x, const Params& par,
                                 double& h, Vec2& gradh, Mat2& hessh)
{
    std::vector<double> hi_all;
    std::vector<Vec2>   gradhi_all;
    std::vector<int>    obs_offset, obs_count;

    CBFsAll(x, par, hi_all, gradhi_all, obs_offset, obs_count);

    gradh = Vec2(0.0, 0.0);
    hessh = Mat2();

    if (hi_all.empty()) {
        h = 1.0;
        return;
    }

    if (!std::isfinite(par.kappa) || par.kappa <= 0.0) {
        std::cerr << "[CBF_eval_recursive_withHess] ERROR: kappa must be finite and >0.\n";
        h = -1.0;
        return;
    }

    const double kappa = par.kappa;

    // Build the composition spec. This default matches the Example-2 style:
    // (edges of obstacle 1 OR ...) AND (edges of obstacle 2 OR ...) AND ...
    ComposeSpec spec;
    build_default_spec_from_obstacles(obs_offset, obs_count, spec);

    // Leaf nodes: H^0_i, g^0_i, S^0_i
    std::vector<NodeVal> prev(hi_all.size());

    for (std::size_t i = 0; i < hi_all.size(); ++i) {
        Mat2 hess_hi; // primitive hi is linear => Hessian = 0
        prev[i] = make_leaf(hi_all[i], gradhi_all[i], hess_hi, kappa, par.tanh_beta);
    }

    // Recursively combine levels
    for (const auto& level : spec.levels) {
        std::vector<NodeVal> cur(level.J.size());
        for (std::size_t i = 0; i < level.J.size(); ++i) {
            const auto& children = level.J[i];
            if (children.empty()) {
                cur[i].H = 1e300;
                continue;
            }
            if (level.type == LevelType::Union) {
                cur[i] = combine_union(prev, children);
            } else {
                cur[i] = combine_intersection(prev, children);
            }
        }
        prev.swap(cur);
    }

    if (prev.empty()) {
        h = 1.0;
        return;
    }

    const NodeVal& root = prev[0];
    double H = root.H;
    if (H < 1e-300) H = 1e-300;

    // h = (log H - b)/kappa
    h = (std::log(H) - par.b) / kappa;

    // grad h = (1/kappa) * grad(log H) = (1/kappa) * (g/H)
    gradh.x = root.g.x / (kappa * H);
    gradh.y = root.g.y / (kappa * H);

    // Hess h = (1/kappa) * Hess(log H)
    // Hess(log H) = (1/H) S - (1/H^2) g g^T
    Mat2 gg = outer(root.g, root.g);
    Mat2 term1 = (1.0 / H) * root.S;
    Mat2 term2 = (1.0 / (H * H)) * gg;

    Mat2 logHess;
    logHess.a11 = term1.a11 - term2.a11;
    logHess.a12 = term1.a12 - term2.a12;
    logHess.a21 = term1.a21 - term2.a21;
    logHess.a22 = term1.a22 - term2.a22;

    hessh = (1.0 / kappa) * logHess;
}

// ============================================================
// Desired controller: PD + waypoint tracking
// ============================================================

Vec2 kd(const State& z, const Params& par, int currentWp) {
    Vec2 target;
    if (par.useRefPath && !par.refPath.empty()
        && currentWp >= 0
        && currentWp < static_cast<int>(par.refPath.size())) {
        target = par.refPath[currentWp];
    } else {
        target = par.xgoal;
    }

    Vec2 e_pos(target.x - z.x.x, target.y - z.x.y);
    Vec2 e_vel(-z.v.x,          -z.v.y);

    Vec2 u(par.Kp * e_pos.x + par.Kv * e_vel.x,
           par.Kp * e_pos.y + par.Kv * e_vel.y);

    // saturation for desired input only
    double n = norm(u);
    if (n > par.umax && n > 1e-12) {
        u = (par.umax / n) * u;
    }
    return u;
}

// ============================================================
// Solve the constrained "single halfspace + ball" QP in 2D:
//
//   minimize_u ||u - ud||^2
//   s.t.        a^T u >= b
//               ||u|| <= uMax
// ============================================================

static inline Vec2 project_to_ball(const Vec2& u, double uMax) {
    double n = norm(u);
    if (n <= uMax || n < 1e-12) return u;
    return (uMax / n) * u;
}

Vec2 solveQP_halfspace_ball_2d(const Vec2& ud,
                               const Vec2& a, double b,
                               double uMax,
                               bool& feasible,
                               bool& activeHalfspace,
                               bool& activeBall)
{
    feasible = true;
    activeHalfspace = false;
    activeBall = false;

    if (uMax <= 0.0) {
        feasible = (b <= 0.0);
        activeBall = true;
        activeHalfspace = (b > 0.0);
        return Vec2(0.0, 0.0);
    }

    double a2 = dot(a, a);
    const double eps = 1e-12;
    if (a2 < eps) {
        if (b > 0.0) feasible = false;
        Vec2 u = project_to_ball(ud, uMax);
        activeBall = (norm(u) >= uMax - 1e-10);
        return u;
    }

    if (dot(a, ud) >= b && norm(ud) <= uMax) {
        return ud;
    }

    Vec2 u_half = ud;
    double Au = dot(a, ud);
    if (Au < b) {
        activeHalfspace = true;
        double lambda = (b - Au) / a2;
        u_half = ud + lambda * a;
    }

    if (norm(u_half) <= uMax) {
        return u_half;
    }

    activeBall = true;

    double an = std::sqrt(a2);
    if (b > uMax * an + 1e-10) {
        feasible = false;
        Vec2 u_best = (uMax / an) * a;
        return u_best;
    }

    Vec2 u_ball = project_to_ball(ud, uMax);
    if (dot(a, u_ball) >= b - 1e-12) {
        return u_ball;
    }

    Vec2 e1 = (1.0 / an) * a;
    Vec2 e2(-e1.y, e1.x);
    double s = b / an;
    double t2 = uMax * uMax - s * s;
    if (t2 < 0.0) t2 = 0.0;
    double t = std::sqrt(t2);

    Vec2 u1 = s * e1 + t * e2;
    Vec2 u2 = s * e1 - t * e2;

    Vec2 d1 = u1 - ud;
    Vec2 d2 = u2 - ud;
    double n1 = dot(d1, d1);
    double n2 = dot(d2, d2);

    activeHalfspace = true;
    return (n1 <= n2) ? u1 : u2;
}


struct QPBallSlackResult {
    Vec2  u;
    double slack;          // delta >= 0 such that a^T u + delta >= b
    bool  hardFeasible;    // whether hard constraint (delta=0) is feasible under ||u||<=uMax
    bool  activeHalfspace; // meaningful only when hardFeasible=true
    bool  activeBall;      // whether ||u|| hits the bound (either hard or soft case)
};

// Soft fallback when hard halfspace+ball is infeasible:
//
//   minimize_u 0.5||u-ud||^2 + 0.5*rho*(b - a^T u)^2
//   s.t.        ||u|| <= uMax
//
// This corresponds to introducing a slack delta >= 0 in the constraint a^T u + delta >= b
// and penalizing delta in the objective (quadratic penalty). We only use this fallback when
// hard feasibility fails, so that when feasible we keep delta=0 exactly.
static inline Vec2 solve_soft_ball_qp(const Vec2& ud,
                                      const Vec2& a, double b,
                                      double uMax, double rho,
                                      bool& activeBall)
{
    activeBall = false;

    if (uMax <= 0.0) {
        activeBall = true;
        return Vec2(0.0, 0.0);
    }

    double a2 = dot(a, a);
    if (a2 < 1e-12 || rho <= 0.0) {
        // No meaningful constraint direction or no penalty -> just project ud to ball
        return project_to_ball(ud, uMax);
    }

    // Q = I + rho a a^T, c = ud + rho b a
    Vec2 c = ud + (rho * b) * a;

    // Unconstrained minimizer u0 = Q^{-1} c (Sherman-Morrison)
    // (I + rho a a^T)^{-1} = I - (rho/(1+rho||a||^2)) a a^T
    double denom0 = 1.0 + rho * a2;
    double atc = dot(a, c);
    Vec2 u0 = c - (rho / denom0) * (atc * a);

    if (norm(u0) <= uMax) {
        return u0;
    }

    // Otherwise, solve with Lagrange multiplier lambda >=0 such that ||u|| = uMax:
    // ( (1+lambda)I + rho a a^T ) u = c
    // Let alpha = 1+lambda >= 1.
    auto u_of_alpha = [&](double alpha) -> Vec2 {
        // inv( alpha I + rho a a^T ) = (1/alpha)I - (rho/(alpha*(alpha+rho||a||^2))) a a^T
        double denom = alpha + rho * a2;
        double atc_local = atc; // a^T c
        Vec2 term1 = (1.0 / alpha) * c;
        Vec2 term2 = (rho / (alpha * denom)) * (atc_local * a);
        return term1 - term2;
    };

    double alpha_lo = 1.0;
    double alpha_hi = 1.0;
    // Increase alpha until ||u(alpha_hi)|| <= uMax
    for (int it = 0; it < 80; ++it) {
        Vec2 uhi = u_of_alpha(alpha_hi);
        if (norm(uhi) <= uMax) break;
        alpha_hi *= 2.0;
        if (alpha_hi > 1e12) break;
    }

    // Bisection on alpha in [alpha_lo, alpha_hi]
    for (int it = 0; it < 80; ++it) {
        double alpha_mid = 0.5 * (alpha_lo + alpha_hi);
        Vec2 umid = u_of_alpha(alpha_mid);
        if (norm(umid) > uMax) alpha_lo = alpha_mid;
        else alpha_hi = alpha_mid;
    }

    activeBall = true;
    return u_of_alpha(alpha_hi);
}

static inline QPBallSlackResult solveQP_halfspace_ball_slack_2d(const Vec2& ud,
                                                                const Vec2& a, double b,
                                                                double uMax,
                                                                double slack_weight)
{
    QPBallSlackResult res;
    res.u = Vec2(0.0, 0.0);
    res.slack = 0.0;
    res.hardFeasible = true;
    res.activeHalfspace = false;
    res.activeBall = false;

    // If a is nearly zero, the halfspace doesn't depend on u.
    double a2 = dot(a, a);
    if (a2 < 1e-12) {
        res.u = project_to_ball(ud, uMax);
        res.activeBall = (norm(res.u) >= uMax - 1e-10);
        res.slack = std::max(0.0, b); // since a^T u = 0
        res.hardFeasible = (b <= 0.0);
        res.activeHalfspace = false;
        return res;
    }

    double an = std::sqrt(a2);

    // Hard feasibility test: exists u with ||u||<=uMax and a^T u >= b  <=>  b <= uMax ||a||
    if (b > uMax * an + 1e-12) {
        // Hard infeasible => use soft fallback
        res.hardFeasible = false;

        bool actBallSoft = false;
        Vec2 u_soft = solve_soft_ball_qp(ud, a, b, uMax, slack_weight, actBallSoft);

        res.u = u_soft;
        res.activeBall = actBallSoft || (norm(res.u) >= uMax - 1e-10);
        res.activeHalfspace = false;

        double viol = b - dot(a, res.u);
        res.slack = (viol > 0.0) ? viol : 0.0;
        return res;
    }

    // Hard feasible: solve the exact projection QP onto (halfspace ∩ ball)
    bool feasible, actHalf, actBall;
    Vec2 u_hard = solveQP_halfspace_ball_2d(ud, a, b, uMax, feasible, actHalf, actBall);

    res.u = u_hard;
    res.activeHalfspace = actHalf;
    res.activeBall = actBall || (norm(res.u) >= uMax - 1e-10);
    res.hardFeasible = feasible;
    res.slack = 0.0;
    return res;
}


// ============================================================
// Continuous-time relative-degree-2 ECBF safety filter (QP + ball bound)
//
// Enforce: hddot + k1*hdot + k0*h >= 0
// => grad h^T u >= - v^T Hess h v - k1*grad h^T v - k0*h
// ============================================================

Vec2 controller_ECBF2_QPball_recursive(const State& z, const Params& par, int currentWp,
                                       Vec2& udOut, double& hOut,
                                       double& hdotOut, double& hddotOut,
                                       double& cbfLhsOut,
                                       double& slackOut,
                                       int& qpFeasibleOut,
                                       int& activeHalfOut,
                                       int& activeBallOut,
                                       int& slackUsedOut)
{
    Vec2 ud = kd(z, par, currentWp);
    udOut = ud;
    slackOut = 0.0;
    slackUsedOut = 0;


    double h;
    Vec2 gradh;
    Mat2 hessh;

    // Use the general recursive Eq.(36) evaluation
    CBF_eval_recursive_withHess(z.x, par, h, gradh, hessh);

    hOut = h;

    double hdot = dot(gradh, z.v);
    hdotOut = hdot;

    double vHv = quadform(hessh, z.v);

    if (!par.filterOn) {
        Vec2 u_nofilter = project_to_ball(ud, par.umax_real);
        double hddot = vHv + dot(gradh, u_nofilter);
        hddotOut = hddot;
        cbfLhsOut = hddot + par.cbf_k1 * hdot + par.cbf_k0 * h;
        qpFeasibleOut = 1;
        activeHalfOut = 0;
        activeBallOut = (norm(u_nofilter) >= par.umax_real - 1e-10) ? 1 : 0;
        slackOut = 0.0;
        slackUsedOut = 0;
        return u_nofilter;
    }

    Vec2 A = gradh;
    double b = -vHv - par.cbf_k1 * hdot - par.cbf_k0 * h;

    QPBallSlackResult qpres = solveQP_halfspace_ball_slack_2d(ud, A, b, par.umax_real, par.slack_weight);

    slackOut      = qpres.slack;
    slackUsedOut  = (qpres.slack > 1e-12) ? 1 : 0;
    qpFeasibleOut = qpres.hardFeasible ? 1 : 0;
    activeHalfOut = qpres.activeHalfspace ? 1 : 0;
    activeBallOut = qpres.activeBall ? 1 : 0;

    Vec2 u = qpres.u;

    double hddot = vHv + dot(gradh, u);
    hddotOut = hddot;
    cbfLhsOut = hddot + par.cbf_k1 * hdot + par.cbf_k0 * h;

    return u;
}

// ============================================================
// Obstacle construction helpers
// ============================================================

PolygonObstacle makeRotatedRectangle(const Vec2& center,
                                     const Vec2& halfSize,
                                     double psi)
{
    PolygonObstacle obs;
    double c = std::cos(psi);
    double s = std::sin(psi);

    auto rot = [&](const Vec2& v) -> Vec2 {
        return Vec2(c * v.x - s * v.y,
                    s * v.x + c * v.y);
    };

    Vec2 p1 = rot(Vec2(-halfSize.x, -halfSize.y));
    Vec2 p2 = rot(Vec2( halfSize.x, -halfSize.y));
    Vec2 p3 = rot(Vec2( halfSize.x,  halfSize.y));
    Vec2 p4 = rot(Vec2(-halfSize.x,  halfSize.y));

    obs.vertices.push_back(center + p1);
    obs.vertices.push_back(center + p2);
    obs.vertices.push_back(center + p3);
    obs.vertices.push_back(center + p4);
    return obs;
}

NormalPointObstacle makeNormalPointObstacleFromVertices(const std::vector<Vec2>& verticesCCW)
{
    NormalPointObstacle obs;
    int M = static_cast<int>(verticesCCW.size());
    if (M < 2) return obs;

    for (int j = 0; j < M; ++j) {
        const Vec2& vj = verticesCCW[j];
        const Vec2& vk = verticesCCW[(j + 1) % M];

        Vec2 e(vk.x - vj.x, vk.y - vj.y);
        double elen = norm(e);
        if (elen < 1e-12) continue;

        Vec2 n(e.y / elen, -e.x / elen);  // outward normal for CCW polygon
        obs.edges.push_back(Edge{n, vj});
    }
    return obs;
}

// ============================================================
// Main: simulation + CSV output
// ============================================================

int main() {
    Params par;
    par.w = Vec2(0.2, 0.3);

    // Obstacles: two rectangles (vertex-based)
    par.polyObs.push_back(makeRotatedRectangle(Vec2(4.0, 6.5), Vec2(1.0, 0.2), 0.0));
    par.polyObs.push_back(makeRotatedRectangle(Vec2(2.0, 7.0), Vec2(1.0, 0.2), 0.0));

    // One hexagon (normal+point via vertices)
    std::vector<Vec2> polyVerts;
    polyVerts.push_back(Vec2(2.0, 1.0));
    polyVerts.push_back(Vec2(4.0, 1.0));
    polyVerts.push_back(Vec2(5.0, 3.0));
    polyVerts.push_back(Vec2(3.5, 4.5));
    polyVerts.push_back(Vec2(2.0, 5.5));
    polyVerts.push_back(Vec2(1.5, 3.0));
    par.npObs.push_back(makeNormalPointObstacleFromVertices(polyVerts));

    // Reference path
    par.useRefPath = true;
    par.wpTol      = 0.2;
    par.refPath.push_back(Vec2(0.0, 0.0));
    par.refPath.push_back(Vec2(2.0, 6.4));
    par.refPath.push_back(Vec2(4.5, 5.2));
    par.refPath.push_back(Vec2(7.0, 7.0));

    // Desired controller params
    par.xgoal = Vec2(7.0, 7.0);
    par.Kp    = 0.5;
    par.Kv    = 1.5;
    par.umax  = 2.0;

    par.filterOn = true;

    // Simulation step
    par.dt = 0.01;

    // CBF composition params
    par.kappa = 10.0;
    par.b     = std::log(2.0);

    // ECBF gains (rd=2): critical damping
    double omega = 2.0;
    par.cbf_k1 = 2.0 * omega;
    par.cbf_k0 = omega * omega;

    par.umax_real = 3.0;

    // Simulation horizon
    double t0   = 0.0;
    double tend = 60.0;
    double dt   = par.dt;
    int N = static_cast<int>((tend - t0) / dt) + 1;

    std::vector<double> t(N);
    std::vector<State>  z(N);
    std::vector<Vec2>   u(N), ud(N);
    std::vector<double> h(N), hc(N), hdot(N), hddot(N), cbfLhs(N), slack(N);
    std::vector<int>    qpFeas(N), activeHalf(N), activeBall(N), slackUsed(N);

    // Initial state
    z[0].x = Vec2(0.0, 0.0);
    z[0].v = Vec2(0.0, 0.0);

    int currentWp = 0;

    // Main loop (RK4)
    for (int k = 0; k < N; ++k) {
        t[k] = t0 + k * dt;

        Vec2 udk, uk;
        double hk, hdotk, hddotk, lhs, slackk;
        int feasFlag, halfFlag, ballFlag, slackFlag;

        uk = controller_ECBF2_QPball_recursive(z[k], par, currentWp,
                                               udk, hk, hdotk, hddotk, lhs, slackk,
                                               feasFlag, halfFlag, ballFlag, slackFlag);

        u[k]      = uk;
        ud[k]     = udk;
        h[k]      = hk;
        hdot[k]   = hdotk;
        hddot[k]  = hddotk;
        cbfLhs[k] = lhs;
        qpFeas[k] = feasFlag;
        activeHalf[k] = halfFlag;
        activeBall[k] = ballFlag;
        slack[k] = slackk;
        slackUsed[k] = slackFlag;

        // Nonsmooth comparison hc
        std::vector<double> hi_all;
        std::vector<Vec2>   gradhi_all;
        std::vector<int>    obs_offset, obs_count;
        CBFsAll(z[k].x, par, hi_all, gradhi_all, obs_offset, obs_count);
        if (!obs_offset.empty()) hc[k] = hlogic_nonsmooth_all(hi_all, obs_offset, obs_count);
        else hc[k] = h[k];

        // RK4 integration
        if (k + 1 < N) {
            Vec2 ud_tmp;
            double h_tmp, hdot_tmp, hddot_tmp, lhs_tmp, slack_tmp;
            int feas_tmp, half_tmp, ball_tmp, slackFlag_tmp;

            State k1 = dynamics(z[k], uk);
            State z2 = z[k] + 0.5 * dt * k1;
            Vec2 u2  = controller_ECBF2_QPball_recursive(z2, par, currentWp,
                                                      ud_tmp, h_tmp, hdot_tmp, hddot_tmp, lhs_tmp, slack_tmp,
                                                      feas_tmp, half_tmp, ball_tmp, slackFlag_tmp);
            State k2 = dynamics(z2, u2);

            State z3 = z[k] + 0.5 * dt * k2;
            Vec2 u3  = controller_ECBF2_QPball_recursive(z3, par, currentWp,
                                                      ud_tmp, h_tmp, hdot_tmp, hddot_tmp, lhs_tmp, slack_tmp,
                                                      feas_tmp, half_tmp, ball_tmp, slackFlag_tmp);
            State k3 = dynamics(z3, u3);

            State z4 = z[k] + dt * k3;
            Vec2 u4  = controller_ECBF2_QPball_recursive(z4, par, currentWp,
                                                      ud_tmp, h_tmp, hdot_tmp, hddot_tmp, lhs_tmp, slack_tmp,
                                                      feas_tmp, half_tmp, ball_tmp, slackFlag_tmp);
            State k4 = dynamics(z4, u4);

            z[k + 1] = z[k] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

            // waypoint switching
            if (par.useRefPath && !par.refPath.empty()
                && currentWp < static_cast<int>(par.refPath.size())) {

                Vec2 target = par.refPath[currentWp];
                Vec2 err(z[k + 1].x.x - target.x, z[k + 1].x.y - target.y);
                double dist = norm(err);

                if (dist < par.wpTol && currentWp + 1 < static_cast<int>(par.refPath.size())) {
                    currentWp++;
                }
            }
        }
    }

    // Output CSV
    std::ofstream ofs("simulation_result_ECBF2_recursive_QPball.csv");
    if (!ofs) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }

    ofs << "t,x1,x2,v1,v2,u1,u2,ud1,ud2,h,hc,hdot,hddot,cbf_lhs,qp_feasible,active_half,active_ball,slack,slack_used\n";
    for (int k = 0; k < N; ++k) {
        ofs << t[k]      << ","
            << z[k].x.x  << "," << z[k].x.y << ","
            << z[k].v.x  << "," << z[k].v.y << ","
            << u[k].x    << "," << u[k].y   << ","
            << ud[k].x   << "," << ud[k].y  << ","
            << h[k]      << "," << hc[k]    << ","
            << hdot[k]   << "," << hddot[k] << ","
            << cbfLhs[k] << ","
            << qpFeas[k] << ","
            << activeHalf[k] << ","
            << activeBall[k] << "\n";
    }
    ofs.close();

    std::cout << "Simulation finished. Results saved to simulation_result_ECBF2_recursive_QPball.csv\n";
    return 0;
}

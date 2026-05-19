public:

    using Scal    = Scal_;
    using Real    = typename Scalar::Real<Scal_>;
    using Int     = Int_;

    static constexpr bool ComplexQ   = Scalar::ComplexQ<Scal>;

    static constexpr Real zero        = 0;
    static constexpr Real half        = 0.5;
    static constexpr Real one         = 1;
    static constexpr Real two         = 2;
    static constexpr Real three       = 3;
    static constexpr Real four        = 4;
    static constexpr Real eps         = Scalar::eps<Scal>;
    static constexpr Real eps_squared = eps * eps;
    static constexpr Real eps_sqrt    = cSqrt(eps);
    static constexpr Real infty       = Scalar::Infty<Scal>;

    static constexpr Scalar::Flag F_Plus    = Scalar::Flag::Plus;
    static constexpr Scalar::Flag F_Minus   = Scalar::Flag::Minus;
    static constexpr Scalar::Flag F_Zero    = Scalar::Flag::Zero;
    static constexpr Scalar::Flag F_Generic = Scalar::Flag::Generic;

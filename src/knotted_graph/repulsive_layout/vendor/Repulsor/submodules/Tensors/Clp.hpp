#pragma once

#define TENSORS_CLP_H

// To use this you need to install CLP (see https://github.com/coin-or/Clp) and link against libClp and libCoinUtils.

// With homebrew ypu just do
//
//      brew install clp
//
// In an Apple Silicon Mac the headers should be located in
//
//      /opt/homebrew/opt/coinutils/include/coinutils/coin
//      /opt/homebrew/opt/clp/include/clp/coin
//
// and the libraries should be located in
//
//      /opt/homebrew/opt/coinutils/lib
//      /opt/homebrew/opt/clp/lib

// Without homwbrew you can use this workflow:
//
//  - git-clone the CoinUtils repository https://github.com/coin-or/CoinUtils
//  - cd into the new directory
//  - run
//          ./configure -C
//          make
//          make test
//          make install
//  - You can delete the cloned repository now.
//  - git-clone CoinUtils https://github.com/coin-or/Clp
//  - cd into the new directory
//  - run
//          ./configure -C
//          make
//          make test
//          make install
//  - You can delete the cloned repository now.
//  - On macos the header files should be located at /user/local/include/coin-or and the library files should be in /usr/local/lib. I guess the paths will be the same under Linux.
//
//


#include "ClpSimplex.hpp"
#include "ClpSimplexDual.hpp"
#include "ClpPlusMinusOneMatrix.hpp"
#include "ClpNetworkMatrix.hpp"
//#include "CoinHelperFunctions.hpp"


namespace Tensors
{
    template<Parallel_T parQ>
    CoinPackedMatrix MatrixCSR_transpose_to_CoinPackedMatrix(
        const Sparse::MatrixCSR<double,int,CoinBigIndex,parQ> & A
    )
    {
        // https://coin-or.github.io/Clp/Doxygen/classClpPackedMatrix.html
//        CoinPackedMatrix B(
//            false, A.RowCount(), A.ColCount(), A.NonzeroCount(),
//            A.Values().data(), A.Inner().data(), A.Outer().data(), nullptr
//        );
        
        CoinPackedMatrix B (
            true, A.ColCount(), A.RowCount(), A.NonzeroCount(),
            A.Values().data(), A.Inner().data(), A.Outer().data(), nullptr
        );
        
        return B;
    }
    
    template<Parallel_T parQ>
    CoinPackedMatrix MatrixCSR_to_CoinPackedMatrix(
        const Sparse::MatrixCSR<double,int,CoinBigIndex,parQ> & A
    )
    {
        // https://coin-or.github.io/Clp/Doxygen/classClpPackedMatrix.html
//        CoinPackedMatrix B(
//            false, A.RowCount(), A.ColCount(), A.NonzeroCount(),
//            A.Values().data(), A.Inner().data(), A.Outer().data(), nullptr
//        );
        
        CoinPackedMatrix B (
            false, A.ColCount(), A.RowCount(), A.NonzeroCount(),
            A.Values().data(), A.Inner().data(), A.Outer().data(), nullptr
        );
        
        return B;
    }

    template<FloatQ Real, IntQ Int, IntQ LInt, Parallel_T parQ>
    Sparse::MatrixCSR<Real,IntQ,IntQ,parQ> CoinPackedMatrix_to_MatrixCSR(
        const CoinPackedMatrix & A, int thread_count = 1
    )
    {
        if( A.isColOrdered() )
        {
            return Sparse::MatrixCSR<FloatQ,Int,LInt,parQ>(
                A.getVectorStarts(), A.getIndices(), A.getElements(),
                A.getNumCols(), A.getNumRows(), thread_count
            ).Transpose();
        }
        else
        {
            return Sparse::MatrixCSR<FloatQ,Int,LInt,parQ>(
                A.getVectorStarts(), A.getIndices(), A.getElements(),
                A.getNumRows(), A.getNumCols(), thread_count
            );
        }
    }
    
    
    // TODO: Check input.
    
    template<FloatQ Real_ = double, IntQ Int_ = int, IntQ LInt_ = Int_>
    class ClpWrapper final
    {
    public:

        using Real = Real_;
        using Int  = Int_;
        using LInt = LInt_;
        
        using COIN_Real = double;
        using COIN_Int  = int;
        using COIN_LInt = CoinBigIndex;
        
        struct Settings_T
        {
            bool check_inputQ    = true;
            bool minimizeQ       = true;
            bool dualQ           = false;
            bool check_integralQ = true;
        };
        
    private:
        
        Settings_T settings;
        
        ClpSimplex LP;
        
        const COIN_Real integer_tol = 0.000'1;
        
    public:
        
    /*!@brief Create an instance of ClpSimplex by loading the supplied data. This sets up the optimization problem
     *      Mininimize/maximize  `c^T.x`
     *      s.t,    `y = x.AT`,
     *             `var_lb[i] <= x[i] <= var_ub[i]` for `0 <= i < n`,
     *      and    `con_lb[j] <= y[j] <= con_ub[j]` for `0 <= j < m`,
     *
     *      where `n = AT.RowCount()`, `m = AT.ColCount()`.
     *
     * @param c The vector with the coefficients for the objective.
     * @param var_lb The lower bounds on the variables.
     * @param var_ub The upper bounds on the variables.
     * @param AT_outer The row pointers of the _transpose_ of the constraint matrix`A`.
     * @param AT_inner The column indices of the _transpose_ of the constraint matrix`A`.
     * @param AT_values The nonzero values of the _transpose_ of the constraint matrix`A`.
     * @param con_ub The lower bounds for the matrix contraints.
     * @param con_lb The upper bounds for the matrix contraints.
     * @param settings_ An instance of `ClpWrapper::Settings_T` to set up parameters.
     */
        
        template<typename R, typename I, typename J>
        ClpWrapper(
            I variable_count, I constraint_count,
            cptr<R> c, cptr<R> var_lb, cptr<R> var_ub,
            cptr<J> AT_outer, cptr<I> AT_inner, cptr<R> AT_values,
            cptr<R> con_lb, cptr<R> con_ub,
            Settings_T settings_ = ClpWrapper::Settings_T{}
        )
        :   settings( settings_ )
        {
            LP.setMaximumIterations(1000000);
            
            if( settings.minimizeQ )
            {
                LP.setOptimizationDirection( 1); // +1 -> minimize; -1 -> maximize
            }
            else
            {
                LP.setOptimizationDirection(-1);
            }
            
            LP.setLogLevel(0);
            
            using RArray_T = Tensor1<COIN_Real,COIN_Int>;
            using IArray_T = Tensor1<COIN_Int ,COIN_Int>;
            using JArray_T = Tensor1<COIN_LInt,COIN_Int>;
            
            const COIN_Int n = static_cast<COIN_Int>(variable_count);
            const COIN_Int m = static_cast<COIN_Int>(constraint_count);
            
            RArray_T AT_values_;
            RArray_T var_lb_;
            RArray_T var_ub_;
            RArray_T con_lb_;
            RArray_T con_ub_;
            RArray_T c_;
            
            IArray_T AT_inner_;
            JArray_T AT_outer_;

            if( !SameQ<R,COIN_Real> )
            {
                c_         = RArray_T( c     , n );
                var_lb_    = RArray_T( var_lb, n );
                var_ub_    = RArray_T( var_ub, n );
                AT_values_ = RArray_T( AT_values, AT_outer[n] );
                con_lb_    = RArray_T( con_lb, m );
                con_ub_    = RArray_T( con_ub, m );
                
            }
            else
            {
                (void)AT_values_;
                (void)var_lb_;
                (void)var_ub_;
                (void)con_lb_;
                (void)con_ub_;
                (void)c_;
            }
            
            if( !SameQ<I,COIN_Int> )
            {
                AT_inner_ = RArray_T( AT_inner, AT_outer[n] );
            }
            else
            {
                (void)AT_inner_;
            }
            
            if( !SameQ<J,COIN_Int> )
            {
                AT_outer_ = RArray_T( AT_outer, n + 1 );
            }
            else
            {
                (void)AT_outer_;
            }
            
            // TODO: Check bounds.
            
            LP.loadProblem(
                n, m,
                SameQ<J,COIN_LInt> ? AT_outer  : AT_outer_.data(),
                SameQ<I,COIN_Int > ? AT_inner  : AT_inner_.data(),
                SameQ<R,COIN_Real> ? AT_values : AT_values_.data(),
                SameQ<R,COIN_Real> ? var_lb    : var_lb_.data(),
                SameQ<R,COIN_Real> ? var_ub    : var_ub_.data(),
                SameQ<R,COIN_Real> ? c         : c_.data(),
                SameQ<R,COIN_Real> ? con_lb    : con_lb_.data(),
                SameQ<R,COIN_Real> ? con_ub    : con_ub_.data()
            );
            
            LP.setLogLevel(0);
            
            if( settings.dualQ )
            {
                LP.dual();
            }
            else
            {
                LP.primal();
            }
            
            if( !LP.statusOfProblem() )
            {
                wprint(ClassName() + "(): ClpSimplex::" + (settings.dualQ ? "dual" : "primal") + " reports a problem in the solve phase.  The returned solution may be incorrect.");

                TOOLS_DDUMP(variable_count);
                TOOLS_DDUMP(constraint_count);
                
                TOOLS_DDUMP(LP.statusOfProblem());
                TOOLS_DDUMP(LP.getIterationCount());
                TOOLS_DDUMP(LP.numberPrimalInfeasibilities());
                TOOLS_DDUMP(LP.largestPrimalError());
                TOOLS_DDUMP(LP.sumPrimalInfeasibilities());
                    
                TOOLS_DDUMP(LP.numberDualInfeasibilities());
                TOOLS_DDUMP(LP.largestDualError());
                TOOLS_DDUMP(LP.sumDualInfeasibilities());
                   
                TOOLS_DDUMP(LP.objectiveValue());
            }
        }
        
        
        /*!@brief Create an instance of ClpSimplex by loading the supplied data. This sets up the optimization problem
         *      Mininimize/maximize  `c^T.x`
         *      s.t,    `y = x.AT`,
         *             `var_lb[i] <= x[i] <= var_ub[i]` for `0 <= i < n`,
         *      and    `con_lb[j] <= y[j] <= con_ub[j]` for `0 <= j < m`,
         *
         *      where `n = AT.RowCount()`, `m = AT.ColCount()`.
         *
         * @param c The vector with the coefficients for the objective.
         * @param var_lb The lower bounds on the variables.
         * @param var_ub The upper bounds on the variables.
         * @param AT The _transpose_ of the constraint matrix `A`. Mind row-major/column-major conversion!
         * @param con_ub The lower bounds for the matrix contraints.
         * @param con_lb The upper bounds for the matrix contraints.
         * @param settings_ An instance of `ClpWrapper::Settings_T` to set up parameters.
         */
        
        template<typename R, IntQ I, IntQ J, Parallel_T parQ>
        ClpWrapper(
            cref<Tensor1<R,I>> c,
            cref<Tensor1<R,I>> var_lb,
            cref<Tensor1<R,I>> var_ub,
            cref<Sparse::MatrixCSR<R,I,J,parQ>> AT,
            cref<Tensor1<R,I>> con_lb,
            cref<Tensor1<R,I>> con_ub,
            cref<Settings_T> settings_ = ClpWrapper::Settings_T{}
        )
        : ClpWrapper{
            AT.RowCount(), AT.ColCount(),
            c.data(), var_lb.data(), var_ub.data(),
            AT.Outer().data(), AT.Inner().data(), AT.Values().data(),
            con_lb.data(), con_ub.data(),
            settings_
        }
        {
            const I n = AT.RowCount();
            const I m = AT.ColCount();

            if( c.Size() != n )
            {
                eprint(ClassName()+ "(): argument c has size " + ToString(c.Size()) + ", but should have size " + ToString(n) +".");
            }
            if( var_lb.Size() != n )
            {
                eprint(ClassName()+ "(): argument var_lb has size " + ToString(var_lb.Size()) + ", but should have size " + ToString(n) +".");
            }
            if( var_ub.Size() != n )
            {
                eprint(ClassName()+ "(): Argument var_ub has size " + ToString(var_ub.Size()) + ", but should have size " + ToString(n) +".");
            }
            
            if( con_lb.Size() != m )
            {
                eprint(ClassName()+ "(): Argument con_lb has size " + ToString(con_lb.Size()) + ", but should have size " + ToString(m) +".");
            }
            if( con_ub.Size() != m )
            {
                eprint(ClassName()+ "(): Argument con_ub has size " + ToString(con_ub.Size()) + ", but should have size " + ToString(m) +".");
            }
        }
        
        
        template<typename R, typename I>
        ClpWrapper(
            I vertex_count, I edge_count,
            cptr<I> tails, cptr<I> heads,
            cptr<R> edge_cost, cptr<R> edge_cap_lb, cptr<R> edge_cap_ub,
            cptr<R> vertex_supply_demand,
            cref<Settings_T> settings_ = ClpWrapper::Settings_T{}
        )
        :   settings( settings_ )
        {
            
            LP.setMaximumIterations(1000000);
            
            if( settings.minimizeQ )
            {
                LP.setOptimizationDirection( 1); // +1 -> minimize; -1 -> maximize
            }
            else
            {
                LP.setOptimizationDirection(-1);
            }
            
            LP.setLogLevel(0);
            
            using RArray_T = Tensor1<COIN_Real,COIN_Int>;
            using IArray_T = Tensor1<COIN_Int ,COIN_Int>;
            
            const COIN_Int n = static_cast<COIN_Int>(edge_count);
            const COIN_Int m = static_cast<COIN_Int>(vertex_count);
            
            RArray_T edge_cap_lb_;
            RArray_T edge_cap_ub_;
            RArray_T supply_demand_;
            RArray_T edge_cost_;
            
            IArray_T tails_;
            IArray_T heads_;

            if( !SameQ<R,COIN_Real> )
            {
                edge_cost_     = RArray_T( edge_cost           , n );
                edge_cap_lb_   = RArray_T( edge_cap_lb         , n );
                edge_cap_ub_   = RArray_T( edge_cap_ub         , n );
                supply_demand_ = RArray_T( vertex_supply_demand, m );
            }
            else
            {
                (void)edge_cost_;
                (void)edge_cap_lb_;
                (void)edge_cap_ub_;
                (void)supply_demand_;
            }
            
            if( !SameQ<I,COIN_Int> )
            {
                tails_ = RArray_T( tails, n );
                heads_ = RArray_T( heads, n );
            }
            else
            {
                (void)tails_;
                (void)heads_;
            }
            
            ClpNetworkMatrix network (
                edge_count,
                SameQ<I,COIN_Int> ? heads : heads_.data(),
                SameQ<I,COIN_Int> ? tails : tails_.data()
            );
            
            LP.loadProblem(
                network,
                SameQ<R,COIN_Real> ? edge_cap_lb  : edge_cap_lb_.data(),
                SameQ<R,COIN_Real> ? edge_cap_ub  : edge_cap_ub_.data(),
                SameQ<R,COIN_Real> ? edge_cost    : edge_cost_.data(),
                SameQ<R,COIN_Real> ? vertex_supply_demand : supply_demand_.data(),
                SameQ<R,COIN_Real> ? vertex_supply_demand : supply_demand_.data()
            );
            
            LP.setLogLevel(0);
            
            if( settings.dualQ )
            {
                LP.dual();
            }
            else
            {
                LP.primal();
            }
        }
        
        
        template<typename R, typename I>
        ClpWrapper(
            cref<Tensor1<I,I>> tails,
            cref<Tensor1<I,I>> heads,
            cref<Tensor1<R,I>> edge_cost,
            cref<Tensor1<R,I>> edge_cap_lb,
            cref<Tensor1<R,I>> edge_cap_ub,
            cref<Tensor1<R,I>> vertex_supply_demand,
            cref<Settings_T>   settings_ = ClpWrapper::Settings_T{}
        )
        : ClpWrapper{
            vertex_supply_demand.Size(), tails.Size(),
            tails.data(), heads.data(),
            edge_cost.data(), edge_cap_lb.data(), edge_cap_ub.data(),
            vertex_supply_demand.data(),
            settings_
        }
        {
            const I n = tails.Size();
//            const I m = vertex_supply_demand.Size();

            if( edge_cost.Size() != n )
            {
                eprint(ClassName()+ "(): Argument edge_cost has size " + ToString(edge_cost.Size()) + ", but should have size " + ToString(n) +".");
            }
            if( edge_cap_lb.Size() != n )
            {
                eprint(ClassName()+ "(): Argument edge_cap_lb has size " + ToString(edge_cap_lb.Size()) + ", but should have size " + ToString(n) +".");
            }
            if( edge_cap_ub.Size() != n )
            {
                eprint(ClassName()+ "(): Argument edge_cap_ub has size " + ToString(edge_cap_ub.Size()) + ", but should have size " + ToString(n) +".");
            }
        }
        
        
        mref<ClpSimplex> Problem()
        {
            return LP;
        }
        
        Int VariableCount() const
        {
            return int_cast<Int>(LP.matrix()->getNumCols());
        }
        
        Int ConstraintCount() const
        {
            return int_cast<Int>(LP.matrix()->getNumRows());
        }
        
        cptr<COIN_Real> PrimalSolutionPtr() const
        {
            return LP.primalColumnSolution();
        }
        
        Tensor1<Real,Int> PrimalSolution() const
        {
            return Tensor1<Real,Int>( PrimalSolutionPtr(), VariableCount() );
        }
        
        Tensor1<Real,Int> PrimalColumnSolution() const
        {
            return Tensor1<Real,Int>( LP.primalColumnSolution(), VariableCount() );
        }
        
        Tensor1<Real,Int> PrimalRowSolution() const
        {
            return Tensor1<Real,Int>( LP.primalRowSolution(), ConstraintCount() );
        }
        
        Tensor1<Real,Int> DualColumnSolution() const
        {
            return Tensor1<Real,Int>( LP.dualColumnSolution(), ConstraintCount() );
        }
        
        Tensor1<Real,Int> DualRowSolution() const
        {
            return Tensor1<Real,Int>( LP.dualRowSolution(), VariableCount() );
        }
        
        
        cptr<COIN_Real> VariableLowerBoundPtr() const
        {
            return LP.getColLower();
        }
        
        Tensor1<Real,Int> VariableLowerBound() const
        {
            return Tensor1<Real,Int>( VariableLowerBoundPtr(), VariableCount() );
        }

        
        cptr<COIN_Real> VariableUpperBoundPtr() const
        {
            return LP.getColUpper();
        }
        
        Tensor1<Real,Int> VariableUpperBound() const
        {
            return Tensor1<Real,Int>( VariableUpperBoundPtr(), VariableCount() );
        }
        
        
        cptr<COIN_Real> ConstraintLowerBoundPtr() const
        {
            return LP.getRowLower();
        }
                
        Tensor1<Real,Int> ConstraintLowerBound() const
        {
            return Tensor1<Real,Int>(ConstraintLowerBoundPtr(),ConstraintCount());
        }
        
        
        cptr<COIN_Real> ConstraintUpperBoundPtr() const
        {
            return LP.getRowUpper();
        }

        Tensor1<Real,Int> ConstraintUpperBound() const
        {
            return Tensor1<Real,Int>(ConstraintUpperBoundPtr(),ConstraintCount());
        }
        
        /*!@brief This does what ClpSimplex::cleanPrimalSolution is supposed to do: Round the solution to nearest integer and recheck feasibility.
         */
        
        template<typename T = Int>
        Tensor1<T,Int> IntegralPrimalSolution() const
        {
            TOOLS_MAKE_FP_STRICT();
            
            const COIN_Int n = LP.getNumCols();
            const COIN_Int m = LP.getNumRows();
            
            Tensor1<COIN_Real,COIN_Int> s ( n );
            Tensor1<COIN_Real,COIN_Int> b ( m );
            
            COIN_Real diff_max = 0;
            COIN_Real diff_sum = 0;
            
            Int var_lb_infeasible_count = 0;
            Int var_ub_infeasible_count = 0;
            
            // Rounding and check.

            cptr<COIN_Real> sol    = LP.primalColumnSolution();
            cptr<COIN_Real> var_lb = VariableLowerBoundPtr();
            cptr<COIN_Real> var_ub = VariableUpperBoundPtr();
            
            for( COIN_Int i = 0; i < n; ++i )
            {
                COIN_Real r_i  = std::round(sol[i]);
                COIN_Real diff = Abs(sol[i] - r_i);
                
                diff_max = Max( diff_max, diff );
                diff_sum += diff;
                s[i] = r_i;
                
                var_lb_infeasible_count += (r_i < var_lb[i]);
                var_ub_infeasible_count += (r_i > var_ub[i]);
            }
            
            if( diff_max > integer_tol )
            {
                eprint(MethodName("IntegralPrimalSolution") + ": CLP returned noninteger solution vector. Greatest deviation = " + ToString(value)(diff_max) + ". Sum of deviations = " + ToString(diff_sum) + "." );
            }
            
            if( var_lb_infeasible_count > Int(0) )
            {
                eprint(MethodName("IntegralPrimalSolution") + ": Violations for lower box constraints = " + ToString(var_lb_infeasible_count) + ".");
            }
            
            if( var_ub_infeasible_count > Int(0) )
            {
                eprint(MethodName("IntegralPrimalSolution") + ": Violations for upper box constraints = " + ToString(var_ub_infeasible_count) +".");
            }
            
            // Using CLP's matrix-vector product so that I do not have to transpose A.
            // Also, I do not want to pull in all that code from SparseBLAS.
            LP.matrix()->times(s.data(),b.data());
            
            Int con_lb_infeasible_count = 0;
            Int con_ub_infeasible_count = 0;
            
            cptr<COIN_Real> con_lb = ConstraintLowerBoundPtr();
            cptr<COIN_Real> con_ub = ConstraintUpperBoundPtr();
            
            for( COIN_Int j = 0; j < m; ++j )
            {
                b[j] = std::round(b[j]);
                con_lb_infeasible_count += (b[j] < con_lb[j]);
                con_ub_infeasible_count += (b[j] > con_ub[j]);
            }
            
            if( con_lb_infeasible_count > Int(0) )
            {
                eprint(MethodName("IntegralPrimalSolution") + ": Violations for lower matrix constraints = " + ToString(con_lb_infeasible_count) + ".");
            }
            
            if( con_ub_infeasible_count > Int(0) )
            {
                eprint(MethodName("IntegralPrimalSolution") + ": Violations for upper matrix constraints = " + ToString(con_ub_infeasible_count) + ".");
            }
            
            // The integer conversion is safe as we have rounded s correctly already.
            return Tensor1<T,Int>(s);
        }

        static std::string MethodName( const std::string & tag )
        {
            return ClassName() + "::" + tag;
        }
        
        static std::string ClassName()
        {
            return std::string("ClpWrapper")
                + "<" + TypeName<Real>
                + "," + TypeName<Int>
                + "," + TypeName<LInt>
                + ">";
        }
        
    }; // ClpWrapper
    
} // namespace Tensors

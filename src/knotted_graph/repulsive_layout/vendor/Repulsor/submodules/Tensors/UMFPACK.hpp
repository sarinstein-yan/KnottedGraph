#pragma once

#include <umfpack.h>

// To use this you need to install SuiteSparse and link against libamd and libumfpack.

namespace Tensors
{
    // TODO: Test whether the code works in the complex case.
    
    template<typename Scal_, IntQ Int_>
    class UMFPACK final
    {
    public:
        
        using Scal = Scal_;
        using Int  = Int_;
  
        using F_T = Scalar::Flag;
  
        using Pattern_T = Sparse::BinaryMatrixCSR<Int,Int,Sequential>;
        
//        static_assert(
//            SameQ<Scal,Real64> || SameQ<Scal,Complex64> ||
//            SameQ<Scal,Real32> || SameQ<Scal,Complex32>,
//            ""
//        );
        
        static_assert(SameQ<Scal,Real64> || SameQ<Scal,Complex64>,"");
        
        using Real = Scalar::Real<Scal>;
        
        static_assert( SameQ<Int,Int64> || SameQ<Int,Int32>, "");
        
    private:
        
        void * symbolic = nullptr;
        void * numeric  = nullptr;
        
        Real * null     = nullptr;
        
        Pattern_T pattern;
        Tensor1<Scal,Int> values;
        
        Tensor1<Scal,Int> x_buffer;
        Tensor1<Scal,Int> b_buffer;
        
        int symbolic_status = -1;
        int numeric_status  = -1;
        int solve_status    = -1;
        
        
    public:

        // No default constructor
        UMFPACK() = delete;
        // Destructor
        ~UMFPACK()
        {
            this->FreeSymbolic();
            this->FreeNumeric ();
        }
        // Copy constructor
        UMFPACK( const UMFPACK & other ) = delete;
        // Copy assignment operator
        UMFPACK & operator=( const UMFPACK & other ) = delete;
//        // Move constructor
//        UMFPACK( UMFPACK && other ) = default;
//        // Move assignment operator
//        UMFPACK & operator=( UMFPACK && other ) = default;
        
        template<IntQ ExtInt, IntQ ExtLInt>
        explicit UMFPACK(
            const ExtInt m_, const ExtInt n_, cptr<ExtInt> outer_, cptr<ExtLInt> inner_
        )
        :   pattern  { outer_, inner_, int_cast<Int>(m_), int_cast<Int>(n_), Int(1) }
        ,   values   { NonzeroCount()     }
        ,   x_buffer { pattern.ColCount() }
        ,   b_buffer { pattern.RowCount() }
        {
            this->SymbolicFactorization();
        }
        
        
        template<IntQ ExtInt, IntQ ExtLInt, typename ExtScal>
        explicit UMFPACK(
            const ExtInt m_, const ExtInt n_,
            cptr<ExtInt> outer_, cptr<ExtLInt> inner_, cptr<ExtScal> values_
        )
        :   UMFPACK ( m_, n_, outer_, inner_ )
        {
            this->SymbolicFactorization();
            this->NumericFactorization(values_);
        }
        
        
        template<typename ExtScal, IntQ ExtInt, IntQ ExtLInt, Parallel_T ExtparQ>
        explicit UMFPACK( cref<Sparse::MatrixCSR<ExtScal,ExtInt,ExtLInt,ExtparQ>> A )
        :   UMFPACK( A.RowCount(), A.ColCount(), A.Outer(), A.Inner(), A.Values() )
        {}
        
        template<typename ExtScal, IntQ ExtInt, IntQ ExtLInt, Parallel_T ExtparQ>
        explicit UMFPACK( cref<Sparse::BinaryMatrixCSR<ExtScal,ExtInt,ExtparQ>> A )
        :   UMFPACK( A.RowCount(), A.ColCount(), A.Outer(), A.Inner() )
        {}

        
        
    public:
        
        Int RowCount() const
        {
            return pattern.RowCount();
        }
        
        Int ColCount() const
        {
            return pattern.ColCount();
        }
        
        Int NonzeroCount() const
        {
            return pattern.NonzeroCount();
        }
        
        cref<Pattern_T> GetPattern() const
        {
            return pattern;
        }
        
        cref<Tensor1<Scal,Int>> Values() const
        {
            return values;
        }
        
        mref<Tensor1<Scal,Int>> Values()
        {
            return values;
        }
        
        int SymbolicStatus() const
        {
            return symbolic_status;
        }
        
        int NumericStatus() const
        {
            return numeric_status;
        }
        
        
        int SolveStatus() const
        {
            return solve_status;
        }
        
        
        void SymbolicFactorization()
        {
            if( symbolic != nullptr ) { return; }
            
            TOOLS_PTIMER(timer,ClassName()+"::SymbolicFactorization" );
            
            const Int m = pattern.RowCount();
            const Int n = pattern.ColCount();
            
            Int * Ap = pattern.Outer().data();
            Int * Ai = pattern.Inner().data();
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    symbolic_status = umfpack_dl_symbolic(m,n,Ap,Ai,null,&symbolic,null,null);
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    symbolic_status = umfpack_sl_symbolic(m,n,Ap,Ai,null,&symbolic,null,null);
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    symbolic_status = umfpack_zl_symbolic(m,n,Ap,Ai,null,null,&symbolic,null,null);
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    symbolic_status = umfpack_cl_symbolic(m,n,Ap,Ai,null,null,&symbolic,null,null);
//                }
            }
            else if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    symbolic_status = umfpack_di_symbolic(m,n,Ap,Ai,null,&symbolic,null,null);
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    symbolic_status = umfpack_si_symbolic(m,n,Ap,Ai,null,&symbolic,null,null);
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    symbolic_status = umfpack_zi_symbolic(m,n,Ap,Ai,null,null,&symbolic,null,null);
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    symbolic_status = umfpack_ci_symbolic(m,n,Ap,Ai,null,null,&symbolic,null,null);
//                }
            }
            
            if( symbolic_status != UMFPACK_OK )
            {
                eprint(ClassName()+"::SymbolicFactorization: Returned error code is " + ToString(symbolic_status) + ". Message: " + this->SymbolicErrorMessage(symbolic_status)
                );
            }
        }
        
        std::string SymbolicErrorMessage( int status )
        {
            switch(status)
            {
                case UMFPACK_OK:
                {
                    return "Symbolic factorization was successful.";
                }
                case UMFPACK_ERROR_n_nonpositive:
                {
                    return "Size of matrix is less than zero.";
                }
                case UMFPACK_ERROR_invalid_matrix:
                {
                    return "One of the following issues: number of entries in the matrix is negative, first row pointer is nonzero, a column has a negative number of entries, a row index is out of bounds, or the columns of input matrix were jumbled (unsorted columns or duplicate entries).";
                }
                case UMFPACK_ERROR_out_of_memory:
                {
                    return "Insufficient memory to perform the symbolic analysis.";
                }
                case UMFPACK_ERROR_argument_missing:
                {
                    return "One or more required arguments is missing.";
                }
                case UMFPACK_ERROR_internal_error:
                {
                    return "Something very serious went wrong. This is a bug. Please contact the author (DrTimothyAldenDavis@gmail.com).";
                }
                default:
                {
                    return "Unknown error code.";
                }
            }
        }

        
        // You can either fill this->Values().data() and call this->NumericFactorization() or you can provide a pointer to the array of nonzero values.
        template<typename ExtScal>
        void NumericFactorization( cptr<ExtScal> a_ )
        {
            values.Read(a_);
            
            this->NumericFactorization();
        }
        
        void NumericFactorization()
        {
            TOOLS_PTIMER(timer,ClassName()+"::NumericFactorization");
            
            if( symbolic_status != UMFPACK_OK )
            {
                eprint(ClassName()+"::NumericFactorization:  Called with invalid symbolic factorization. symbolic_status = " + ToString(symbolic_status) + ". Message: " + this->SymbolicErrorMessage(symbolic_status)
                );
            }
            
            Int  * Ap = pattern.Outer().data();
            Int  * Ai = pattern.Inner().data();
            Real * Ax = reinterpret_cast<Real *>(values.data());

            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    numeric_status = umfpack_dl_numeric(Ap,Ai,Ax,symbolic,&numeric,null,null);
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    numeric_status = umfpack_sl_numeric(Ap,Ai,Ax,symbolic,&numeric,null,null);
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    numeric_status = umfpack_zl_numeric(Ap,Ai,Ax,null,symbolic,&numeric,null,null);
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    numeric_status = umfpack_cl_numeric(Ap,Ai,Ax,null,symbolic,&numeric,null,null);
//                }
            }
            else //  if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    numeric_status = umfpack_di_numeric(Ap,Ai,Ax,symbolic,&numeric,null,null);
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    numeric_status = umfpack_si_numeric(Ap,Ai,Ax,symbolic,&numeric,null,null);
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    numeric_status = umfpack_zi_numeric(Ap,Ai,Ax,null,symbolic,&numeric,null,null);
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    numeric_status = umfpack_ci_numeric(Ap,Ai,Ax,null,symbolic,&numeric,null,null);
//                }
            }
            
            if( numeric_status != UMFPACK_OK )
            {
                eprint(ClassName()+"::NumericFactorization: Returned error code is " + ToString(numeric_status) + ". Message: " + this->NumericErrorMessage(numeric_status)
                );
            }
        }
        
        std::string NumericErrorMessage( int status )
        {
            switch(status)
            {
                case UMFPACK_OK:
                {
                    return "Numeric factorization was successful.";
                }
                case UMFPACK_WARNING_singular_matrix:
                {
                    return "Numeric factorization was successful, but the matrix is singular.";
                }
                case UMFPACK_ERROR_out_of_memory:
                {
                    return "Insufficient memory to complete the numeric factorization.";
                }
                case UMFPACK_ERROR_argument_missing:
                {
                    return "One or more required arguments are missing.";
                }
                case UMFPACK_ERROR_invalid_Symbolic_object:
                {
                    return "Symbolic object provided as input is invalid.";
                }
                case UMFPACK_ERROR_different_pattern:
                {
                    return "The sparsity pattern has changed since the call to `SymbolicFactorization`.";
                }
                default:
                {
                    return "Unknown error code.";
                }
            }
        }
        
    private:
        
        template<Op op>
        constexpr int SolveMode()
        {
            static_assert(
                (op == Op::Id       ) || (op == Op::Trans) ||
                (op == Op::ConjTrans) || (op == Op::Conj ), ""
            );
            
            // UMFPACK works with CSC format; we use CSR format.
            // Hence, we have to make sure that things are correctly transposed.
            
            switch( op )
            {
                case Op::Trans:     return UMFPACK_A;
                
                case Op::Id:        return UMFPACK_Aat;
                
                case Op::Conj:      return UMFPACK_At;
                    
                // Here it becomes hacky.
                // We also have to conjugate the inputs and outputs.
                case Op::ConjTrans: return UMFPACK_A;
            }
        }
        
        
        
    public:
        
        template<
            Op op = Op::Id, F_T alpha_flag, F_T beta_flag,
            typename a_T, typename B_T, typename b_T, typename X_T
        >
        void Solve( const a_T alpha, cptr<B_T> B, const b_T beta, mptr<X_T> X )
        {
            if( numeric_status != UMFPACK_OK )
            {
                eprint( ClassName()+"::Solve: Called without valid numeric factorization. numeric_status = " + ToString(numeric_status) + "." );
                return;
            }
            
            if( numeric == nullptr )
            {
                eprint( ClassName()+"::Solve: NumericFactorization has not been called, yet." );
                return;
            }
            
            
            TOOLS_PTIMER(timer,ClassName()+"::Solve"
                 + "<" + TypeName<a_T>
                 + "," + TypeName<B_T>
                 + "," + TypeName<b_T>
                 + "," + TypeName<X_T>
                 + ">"
            );
            
            // UMFPACK works with CSC format; we use CSR format.
            // Hence, we have to make sure that things are correctly transposed.

            const int mode = this->template SolveMode<op>();

            // We have to conjugate x to emulate the conjugate-transpose solve.
            combine_buffers<
                F_T::Plus, F_T::Zero, VarSize, Sequential,
                (op == Op::ConjTrans) ? Op::Conj : Op::Id, Op::Id
            >(
                Scal(1), B, Scal(0), b_buffer.data(), pattern.RowCount()
            );

            Real * x_ptr = reinterpret_cast<Real *>(x_buffer.data());
            Real * b_ptr = reinterpret_cast<Real *>(b_buffer.data());
            
            Int  * Ap = pattern.Outer().data();
            Int  * Ai = pattern.Inner().data();
            Real * Ax = reinterpret_cast<Real *>(values.data());
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    solve_status = umfpack_dl_solve(
                        mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null
                    );
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    solve_status = umfpack_sl_solve(
//                        mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null
//                    );
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    solve_status = umfpack_zl_solve(
                        mode,Ap,Ai,Ax,null,x_ptr,null,b_ptr,null,numeric,null,null
                    );
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    solve_status = umfpack_cl_solve(
//                        mode,Ap,Ai,Ax,null,x_ptr,null,b_ptr,null,numeric,null,null
//                    );
//                }
            }
            else
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    solve_status = umfpack_di_solve(
                        mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null
                    );
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    solve_status = umfpack_si_solve(
//                        mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null
//                    );
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    solve_status = umfpack_zi_solve(
                        mode,Ap,Ai,Ax,null,x_ptr,null,b_ptr,null,numeric,null,null
                    );
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    solve_status = umfpack_ci_solve(
//                        mode,Ap,Ai,Ax,null,x_ptr,null,b_ptr,null,numeric,null,null
//                    );
//                }
            }
            
            if( solve_status != 0 )
            {
                eprint(ClassName()+"::Solve: Returned error code is " + ToString(solve_status) + ".");
            }

            // We have to conjugate x to emulate the conjugate-transpose solve.
            combine_buffers<
                alpha_flag, beta_flag, VarSize, Sequential,
                (op == Op::ConjTrans) ? Op::Conj : Op::Id, Op::Id
            >(
                alpha, x_buffer.data(), beta, X, ColCount()
            );
        }
        
        template<Op op = Op::Id, typename B_T, typename X_T>
        void Solve( cptr<B_T> B, mptr<X_T> X )
        {
            this->template Solve<op,F_T::Plus,F_T::Zero>( Scal(1), B, Scal(0), X );
        }
        
        
        std::pair<Scal,Real> Determinant()
        {
            TOOLS_PTIMER(timer,ClassName()+"::Determinant");
            
            if( numeric == nullptr )
            {
                eprint( ClassName()+"::Determinant: NumericFactorization has not been called, yet." );
                return std::pair<Scal,Real>( 0, 0 );
            }
            
            if( (symbolic_status != 0) || (numeric_status != 0) )
            {
                wprint( ClassName()+"::Determinant: Called without valid numeric factorization. Numeric_status = " + ToString(numeric_status) + "." );
                return std::pair<Scal,Real>( 0, 0 );
            }
            
            Scal Mx;
            Real Ex;
            
            int status = 0;
            
            Tiny::Vector<UMFPACK_INFO,double,Int> info;
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_dl_get_determinant(
                        &Mx,&Ex,numeric,info.data()
                    );
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    status = umfpack_sl_get_determinant(
//                        &Mx,&Ex,numeric,info.data()
//                    );
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zl_get_determinant(
                        reinterpret_cast<Real *>(&Mx),null,&Ex,numeric,info.data()
                    );
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    status = umfpack_cl_get_determinant(
//                        reinterpret_cast<Real *>(&Mx),null,&Ex,numeric,info.data()
//                    );
//                }
            }
            else if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_di_get_determinant(
                        &Mx,&Ex,numeric,info.data()
                    );
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    status = umfpack_si_get_determinant(
//                        &Mx,&Ex,numeric,info.data()
//                    );
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zi_get_determinant(
                        reinterpret_cast<Real *>(&Mx),null,&Ex,numeric,info.data()
                    );
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    status = umfpack_ci_get_determinant(
//                        reinterpret_cast<Real *>(&Mx),null,&Ex,numeric,info.data()
//                    );
//                }
            }
            
            if( status != 0 )
            {
                eprint(ClassName()+"::Determinant: Returned error code is " + ToString(status) + ".");
            }
            
            return std::pair<Scal,Real>( Mx, Ex );
        }
        
        
    private:
        
        void FreeSymbolic()
        {
            if( symbolic == nullptr ) { return; }
            
            TOOLS_PTIMER(timer,ClassName()+"::FreeSymbolic");
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    umfpack_dl_free_symbolic(&symbolic);
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    umfpack_sl_free_symbolic(&symbolic);
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    umfpack_zl_free_symbolic(&symbolic);
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    umfpack_cl_free_symbolic(&symbolic);
//                }
            }
            else // if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    umfpack_di_free_symbolic(&symbolic);
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    umfpack_si_free_symbolic(&symbolic);
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    umfpack_zi_free_symbolic(&symbolic);
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    umfpack_ci_free_symbolic(&symbolic);
//                }
            }
                
            symbolic = nullptr;
        }
        
        void FreeNumeric()
        {
            if( numeric == nullptr ) { return; }
            
            TOOLS_PTIMER(timer,ClassName()+"::FreeNumeric");
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    umfpack_dl_free_numeric(&numeric);
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    umfpack_sl_free_numeric(&numeric);
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    umfpack_zl_free_numeric(&numeric);
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    umfpack_cl_free_numeric(&numeric);
//                }
            }
            else // if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    umfpack_di_free_numeric(&numeric);
                }
//                else if constexpr( SameQ<Scal,Real32> )
//                {
//                    umfpack_si_free_numeric(&numeric);
//                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    umfpack_zi_free_numeric(&numeric);
                }
//                else if constexpr( SameQ<Scal,Complex32> )
//                {
//                    umfpack_ci_free_numeric(&numeric);
//                }
            }
        }
        
    public:
        
        static std::string ClassName()
        {
            return std::string("UMFPACK") + "<" + TypeName<Scal> + ">";
        }
        
    }; // class UMFPACK
    
} // namespace Tensors

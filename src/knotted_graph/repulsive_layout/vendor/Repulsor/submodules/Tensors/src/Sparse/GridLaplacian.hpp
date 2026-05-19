#pragma once

namespace Tensors
{
    namespace Sparse
    {
        /*!@brief Finite difference Laplacian (5 stencil) on a square grid. Mostly meant as standard test case for debugging. It also tests the sequential assembly pipeline.
         *
         */
        
        template<typename Scal, IntQ Int, IntQ LInt, Parallel_T parQ = Parallel>
        MatrixCSR<Scal,Int,LInt,parQ> GridLaplacian(
            const Int grid_size, const Scal mass, const Int thread_count
        )
        {
            // Assembling graph Laplacian + mass matrix on grid of size grid_size x grid_size.
            TripleAggregator<Int,Int,Scal,LInt> triples;

            for( Int i = 0; i < grid_size - 1; ++i )
            {
                for( Int j = 0; j < grid_size - 1; ++j )
                {
                    Int V = grid_size * i + j;
                    Int L = V + grid_size;
                    Int R = V + 1;
                    
                    triples.Push( V, V,  1 );
                    triples.Push( V, R, -1 );
                    triples.Push( R, V, -1 );
                    triples.Push( R, R,  1 );
                    
                    triples.Push( V, V,  1 );
                    triples.Push( V, L, -1 );
                    triples.Push( L, V, -1 );
                    triples.Push( L, L,  1 );
                }
            }
            
            for( Int i = 0; i < grid_size - 1; ++i )
            {
                Int j = grid_size-1;
                Int V = grid_size * i + j;
                Int L = V + grid_size;
                
                triples.Push( V, V,  1 );
                triples.Push( V, L, -1 );
                triples.Push( L, V, -1 );
                triples.Push( L, L,  1 );
            }
            
            for( Int j = 0; j < grid_size - 1; ++j )
            {
                Int i = grid_size - 1;
                Int V = grid_size * i + j;
                Int R = V + 1;
                
                triples.Push( V, V,  1 );
                triples.Push( V, R, -1 );
                triples.Push( R, V, -1 );
                triples.Push( R, R,  1 );
            }
            
            for( Int i = 0; i < grid_size; ++i )
            {
                for( Int j = 0; j < grid_size; ++j )
                {
                    Int V = grid_size * i + j;
                    triples.Push( V, V, mass );
                }
            }
            
            return Sparse::MatrixCSR<Scal,Int,LInt,parQ>(
                triples,
                grid_size * grid_size, grid_size * grid_size,
                thread_count, true, false, false
            );
        }
        
        
        
        /*!@brief Finite difference Laplacian (5 stencil) on a square grid. Mostly meant as standard test case for debugging. It also tests the parallel assembly pipeline.
         *
         */
        
        template<typename Scal, IntQ Int, IntQ LInt,Parallel_T parQ = Parallel>
        MatrixCSR<Scal,Int,LInt,parQ> GridLaplacian_Parallel(
            const Int grid_size, const Scal mass, const Int thread_count
        )
        {
            // Assembling graph Laplacian + mass matrix on grid of size grid_size x grid_size.
            
            std::vector<TripleAggregator<Int,Int,Scal,LInt>> thread_triples ( Max(Size_T(1), ToSize_T(thread_count)) );

            Do<parQ>(
                [&thread_triples,thread_count,grid_size,mass]( const Int thread )
                {
                    auto & triples = thread_triples[ToSize_T(thread)];
                    
                    const Int i_begin = JobPointer( grid_size - Int(1), thread_count, thread );
                    const Int i_end   = JobPointer( grid_size - Int(1), thread_count, thread + Int(1) );
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int j = 0; j < grid_size - Int(1); ++j )
                        {
                            Int V = grid_size * i + j;
                            Int L = V + grid_size;
                            Int R = V + Int(1);
                            
                            triples.Push( V, V, Scal( 2) + mass );
                            triples.Push( V, R, Scal(-1) );
                            triples.Push( R, R, Scal( 1) );
                            triples.Push( V, L, Scal(-1) );
                            triples.Push( L, L, Scal( 1) );
                        }
                        
                        {
                            Int j = grid_size - Int(1);
                            Int V = grid_size * i + j;
                            Int L = V + grid_size;
                            
                            triples.Push( V, V, Scal( 1) + mass );
                            triples.Push( V, L, Scal(-1) );
                            triples.Push( L, L, Scal( 1) );
                        }
                    }
                    
                    if( thread + Int(1) == thread_count )
                    {
                        for( Int j = 0; j < grid_size - Int(1); ++j )
                        {
                            Int i = grid_size - Int(1);
                            Int V = grid_size * i + j;
                            Int R = V + Int(1);
                            
                            triples.Push( V, V, Scal( 1) + mass );
                            triples.Push( V, R, Scal(-1) );
                            triples.Push( R, R, Scal( 1) );
                        }
                        
                        {
                            Int i = grid_size - Int(1);
                            Int j = grid_size - Int(1);
                            
                            const Int V = grid_size * i + j;
                            triples.Push( V, V, mass );
                        }
                    }
                },
                thread_count, thread_count
            );
            
            return Sparse::MatrixCSR<Scal,Int,LInt,parQ>(
                thread_triples,
                grid_size * grid_size, grid_size * grid_size,
                thread_count, true, true
            );
        }
    }
} // namespace Tensors

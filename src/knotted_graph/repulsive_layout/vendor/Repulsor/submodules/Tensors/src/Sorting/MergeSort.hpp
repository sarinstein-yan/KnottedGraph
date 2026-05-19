#pragma once

namespace Tensors
{    
    template<Size_T vector_size = 8, bool reverseQ = false>
    class MergeSorter final
    {
    public:
        
        MergeSorter() = default;
        
    private:
        
        Size_T n;
        
        Size_T reg_chunk_count;
        Size_T     chunk_count;
        
        Size_T node_count = 0;
        Size_T internal_node_count = 0;
        
        // Leaves are precisely the last edge count nodes.
        Size_T last_row_begin = 0;
        Size_T offset         = 0;
        
        Tensor1<Size_T,Size_T> node_begin;
        Tensor1<Size_T,Size_T> node_end;
        
    public:
        
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, mptr<T> b, const Size_T n_, C comp = C() )
        {
            // a is the buffer to be sorted.
            // b is work space
            
            if( n_ == 0 )
            {
                return;
            }
            
            
            FindChunkSize<1>( a, b, n_, comp  );
        }
                
        template<Size_T chunk_size, typename T, typename C = std::less<T>>
        void MergeSort( mptr<T> a, mptr<T> b, const Size_T n_, C comp = C() )
        {
            n = n_;
            
            reg_chunk_count = n / chunk_size;
            chunk_count     = (n + chunk_size - 1 )/ chunk_size;
            
            PrepareBinaryTree( chunk_size );

            SortChunks<chunk_size>( a, b, comp );
            
            Merge_DFS( a, b, comp );
        }

        
    private:
        
        
        template<Size_T chunk_size, typename T, typename C>
        void FindChunkSize( mptr<T> a, mptr<T> b, const Size_T n_, C comp  )
        {
            if( n_ < Size_T(2) * chunk_size )
            {
                return MergeSort<chunk_size>( a, b, n_, comp );
            }
            else
            {
                if constexpr ( chunk_size < 2305843009213693952UL )
                {
                    return FindChunkSize<2*chunk_size>( a, b, n_, comp );
                }
            }
        }
        
        template<Size_T chunk_size, typename T, typename C = std::less<T>>
        void SortChunks( mptr<T> a, mptr<T> b, C comp = C() )
        {
            for( Size_T chunk = 0; chunk < reg_chunk_count; ++chunk )
            {
                BitonicSort<chunk_size,vector_size,reverseQ>( &a[chunk_size * chunk], comp );
            }
            
            if( chunk_count > reg_chunk_count )
            {
                Size_T reg_n = chunk_size * reg_chunk_count;
                T * a_ = &a[reg_n];
                T * b_ = &b[reg_n];
                
                Size_T n_ = n - reg_n;
                
                
                if constexpr( chunk_size > Size_T(16) )
                {
                    MergeSorter<vector_size,reverseQ> S ;
                    
                    S.MergeSort<chunk_size/Size_T(2)>( a_, b_, n_, comp );
                }
                else
                {
                   switch( n_ )
                    {
                        case 2:
                        {
                            SortNet<2,reverseQ>()(a_,comp);
                            break;
                        }
                        case 3:
                        {
                            SortNet<3,reverseQ>()(a_,comp);
                            break;
                        }
                        case 4:
                        {
                            SortNet<4,reverseQ>()(a_,comp);
                            break;
                        }
                        case 5:
                        {
                            SortNet<5,reverseQ>()(a_,comp);
                            break;
                        }
                        case 6:
                        {
                            SortNet<6,reverseQ>()(a_,comp);
                            break;
                        }
                        case 7:
                        {
                            SortNet<7,reverseQ>()(a_,comp);
                            break;
                        }
                        case 8:
                        {
                            SortNet<8,reverseQ>()(a_,comp);
                            break;
                        }
                        case 9:
                        {
                            SortNet<9,reverseQ>()(a_,comp);
                            break;
                        }
                        case 10:
                        {
                            SortNet<10,reverseQ>()(a_,comp);
                            break;
                        }
                        case 11:
                        {
                            SortNet<11,reverseQ>()(a_,comp);
                            break;
                        }
                        case 12:
                        {
                            SortNet<12,reverseQ>()(a_,comp);
                            break;
                        }
                        case 13:
                        {
                            SortNet<13,reverseQ>()(a_,comp);
                            break;
                        }
                        case 14:
                        {
                            SortNet<14,reverseQ>()(a_,comp);
                            break;
                        }
                        case 15:
                        {
                            SortNet<15,reverseQ>()(a_,comp);
                            break;
                        }
                        case 16:
                        {
                            SortNet<16,reverseQ>()(a_,comp);
                            break;
                        }
                    }
                }
            }
            
            // TODO: This copy can be avoided by sorting the chunks directly into a, b, depending on the parity of the the depth of their leave node.
            
            copy_buffer( a, b, n );
        }
        
        
        template<typename T, typename C = std::less<T>>
        void Merge_DFS( mptr<T> a, mptr<T> b, C comp = C() )
        {
            TOOLS_PTIMER(timer,ClassName()+"::Merge_DFS");
            
            merge_DFS( 0, b, a, comp );
        }
        
        template<typename T, typename C = std::less<T>>
        void merge_DFS( Size_T N, mptr<T> a_0, mptr<T> a_1, C comp = C() )
        {
            if( N < internal_node_count )
            {
                const Size_T L = LeftChild(N);
                const Size_T R = L+1;
                
                merge_DFS( L, a_1, a_0, comp );
                merge_DFS( R, a_1, a_0, comp );

                MergeBuffers<reverseQ>(
                    &a_0[node_begin[L]], node_end[L] - node_begin[L],
                    &a_0[node_begin[R]], node_end[R] - node_begin[R],
                    &a_1[node_begin[N]],
                    comp
                );
            }
        }
        
        
        void PrepareBinaryTree( const Size_T chunk_size )
        {
            TOOLS_PTIMER(timer,ClassName()+"::PrepareBinaryTree");
            
            node_count          = ( 2 * chunk_count - 1 );
            internal_node_count = node_count - chunk_count;
            
            // Leaves are precisely the last edge_count nodes.
            last_row_begin      = (Size_T(1) << Depth(node_count-1)) - 1;
            offset              = node_count - internal_node_count - last_row_begin;
            
            node_begin.template RequireSize<false>( node_count );
            node_end  .template RequireSize<false>( node_count );

            // TODO: Can we somehow avoid the use of node_begin and node_end?
            
            // Compute range of leave nodes in last row.
            for( Size_T N = last_row_begin; N < node_count; ++N )
            {
                node_begin[N] =       chunk_size * (N - last_row_begin    );
                node_end  [N] = Min(n,chunk_size * (N - last_row_begin + 1));
            }
            
            
            // Compute range of leave nodes in penultimate row.
            for( Size_T N = internal_node_count; N < last_row_begin; ++N )
            {
                node_begin[N] =       chunk_size * (N + offset    );
                node_end  [N] = Min(n,chunk_size * (N + offset + 1));
            }
            
            // Compute range of internal nodes.
            for( Size_T N = internal_node_count; N --> Size_T(0); )
            {
                const Size_T L = LeftChild (N);
                const Size_T R = L + Size_T(1);
                
                node_begin[N] = node_begin[L];
                node_end  [N] = node_end  [R];
            }
        }
        
        
    public:
        
        static constexpr Size_T LeftChild( const Size_T node )
        {
            return Size_T(2) * node + Size_T(1);
        }
        
        static constexpr Size_T RightChild( const Size_T node )
        {
            return Size_T(2) * node + Size_T(2);
        }
        
        static constexpr Size_T Parent( const Size_T node )
        {
            // TODO: -1 as return value is not meaningful for unsigned type Size_T!
            return node > Size_T(0) ? (node - Size_T(1)) / Size_T(2) : -1;
        }
        
        static constexpr Size_T Depth( const Size_T node )
        {
            // Depth equals the position of the most significant bit of node+1.
            constexpr Size_T one = 1;
            
            return static_cast<Size_T>( MSB( node + one ) - one );
        }
        
        
        static constexpr Size_T Column( const Size_T i )
        {
            // The start of each column is the number with all bits < Depth() being set.
            
            constexpr Size_T one = 1;
            
            Size_T k = i + 1;

            return i - (PrevPow(k) - one);
        }
        
        Size_T Begin( const Size_T node ) const
        {
            return node_begin[node];
        }
        
        Size_T End( const Size_T node ) const
        {
            return node_end[node];
        }
        
        static std::string ClassName()
        {
            return std::string("MergeSorter")+"<"+ToString(vector_size)+","+ToString(reverseQ)+">";
        }
    };
    
} // namespace Tensors


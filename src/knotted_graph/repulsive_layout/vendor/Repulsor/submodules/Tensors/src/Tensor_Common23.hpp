private:

std::array<Int,rank> dims = {};     // dimensions visible to user

public:



// Copy constructor
TENSOR_T( const TENSOR_T & other )
:   n    ( other.n    )
,   dims ( other.dims )
{
#ifdef TENSORS_ALLOCATION_LOGS
    logprint(ClassName() + " copy-constructor (size = " + ToString(other.Size()) + ")");
#endif
    allocate();
    Read(other.a);
}

// Copy-cast constructor
template<typename S, IntQ J, Size_T alignment_>
explicit TENSOR_T( const TENSOR_T<S,J,alignment_> & other )
:   n    ( other.Size()    )
{
    copy_buffer<rank>(other.Dims(),other.Dims());
    
#ifdef TENSORS_ALLOCATION_LOGS
    logprint(ClassName() + " copy-cast constuctor (size = " + ToString(other.Size()) + ")");
#endif
    allocate();
    Read(other.data());
}

inline friend void swap( TENSOR_T & A, TENSOR_T & B) noexcept
{
#ifdef TENSORS_ALLOCATION_LOGS
    logprint(ClassName() + " swap (sizes = {" + ToString(A.Size()) + "," + ToString(B.Size()) + "})");
#endif
    // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
    using std::swap;
    
    if( &A == &B )
    {
        wprint( std::string("An object of type ") + ClassName() + " has been swapped to itself.");
    }
    else
    {
        swap( A.dims, B.dims    );
        swap( A.n   , B.n       );
        swap( A.a   , B.a       );
    }
}

// Copy assignment operator
// We ship our own because we can do a few optimizations here.
mref<TENSOR_T> operator=( const TENSOR_T & other )
{
#ifdef TENSORS_ALLOCATION_LOGS
        logprint(ClassName() + " copy-assignment (size = " + ToString(other.Size()) + ")");
#endif
    if( this != &other )
    {
        if( dims != other.dims )
        {
            n    = other.n;
            dims = other.dims;
            
#ifdef TENSORS_ALLOCATION_LOGS
            logprint( ClassName() + " reallocation (size = " + ToString(other.Size()) + ")");
#endif
            
            safe_free(a);
            allocate();
        }
        Read( other.a );
    }
    return *this;
}


// Copy-cast-assignment operator
template<
    typename S, typename J, Size_T alignment_,
    class = std::enable_if_t<(!SameQ<S,Scal>) || (!SameQ<J,Int>) || ( alignment_ != Alignment)>
>
mref<TENSOR_T> operator=( const TENSOR_T<S,J,alignment_> & other )
{

#ifdef TENSORS_ALLOCATION_LOGS
    logprint(ClassName() + " copy-cast-assignment (size = " + ToString(other.Size()) + ")");
#endif
    bool different_dimsQ = false;
    
    constexpr Size_T i_count = ToSize_T(Rank());
    
    for( Size_T i = 0; i < i_count; ++i )
    {
        different_dimsQ = different_dimsQ || std::cmp_not_equal( Dim(i), other.Dim(i) );
    }
    
    if( different_dimsQ )
    {
#ifdef TENSORS_ALLOCATION_LOGS
        logprint(ClassName() + " reallocation (size = " + ToString(other.Size()) + ")");
#endif
        n = other.Size();
        
        for( Size_T i = 0; i < i_count; ++i )
        {
            dims[i] = int_cast<Int>(other.Dim(i));
        }
        
        safe_free(a);
        allocate();
    }
    Read( other.data() );
    
    return *this;
}


TOOLS_FORCE_INLINE cptr<Int> Dims() const
{
    return &dims[0];
}

TOOLS_FORCE_INLINE Int Dim( const Int i ) const
{
    return ( i < Rank() ) ? dims[ToSize_T(i)] : Scalar::Zero<Int>;
}

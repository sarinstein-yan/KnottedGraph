#pragma once

namespace Tensors
{
    // Some extra care was taken so that Int_ can be an unsigned integer.
    template<typename Entry_T_, IntQ Int_>
    class Stack final
    {
    public:
        
        using Entry_T     = Entry_T_;
        using Int         = Int_;

        Stack( Int max_size )
        :   a ( max_size + Int(1) )
        {
            a[0] = Entry_T();
        }
        
        // For debugging purposed it might be nice to have a controlled initialization.
        Stack( Int max_size, Entry_T init )
        :   a ( max_size + Int(1), init )
        {
//            a[0] = Entry_T();
        }
        
        // Default constructor
        Stack()
        :   a ( Int(1) )
        {
            a[0] = Entry_T();
        }
        
//        // Destructor
//        ~Stack() = default;
//        // Copy constructor
//        Stack( const Stack & other ) = default;
//        // Copy assignment operator
//        Stack & operator=( const Stack & other ) = default;
//        // Move constructor
//        Stack( Stack && other ) = default;
//        // Move assignment operator
//        Stack & operator=( Stack && other ) = default;
        
        Int Capacity() const
        {
            return a.Size()-Int(1);
        }
        
        Int Size() const
        {
            return ptr;
        }
        
        mptr<Entry_T> begin()
        {
            return &a[Int(1)];
        }
        
        cptr<Entry_T> begin() const
        {
            return &a[Int(1)];
        }
        
        mptr<Entry_T> end()
        {
            return &a[ptr+Int(1)];
        }
        
        cptr<Entry_T> end() const
        {
            return &a[ptr+Int(1)];
        }
        
        mref<Entry_T> operator[]( const Int i )
        {
            return a[i+Int(1)];
        }
        
        cref<Entry_T> operator[]( const Int i ) const
        {
            return a[i+Int(1)];
        }
        
        void Reset()
        {
            ptr = 0;
        }
        
        void Reset( const Int minimum_size)
        {
            a.template RequireSize<false>(minimum_size);
            ptr = 0;
        }
        
        void Push( cref<Entry_T> value )
        {
#ifdef TENSORS_BOUND_CHECKS
            if( (ptr + 1) == a.Size() )
            {
                eprint(ClassName()+"::Push: Stack overflow");
            }
#endif
            
            a[++ptr] = value;
        }
        
        void Push( Entry_T && value )
        {
#ifdef TENSORS_BOUND_CHECKS
            if( (ptr + 1) == a.Size() )
            {
                eprint(ClassName()+"::Push: Stack overflow");
            }
#endif
            a[++ptr] = std::move(value);
        }
        
        mref<Entry_T> Top()
        {
            return a[ptr];
        }
        
        cref<Entry_T> Top() const
        {
            return a[ptr];
        }
        
        Entry_T Pop()
        {
#ifdef TENSORS_BOUND_CHECKS
            if( ptr <= Int(0) )
            {
                eprint(ClassName()+"::Push: Stack underflow");
                return Entry_T();
            }
#endif
            Entry_T r ( std::move(a[ptr--]) );
            
//            ptr = (ptr > Int(0)) ? (ptr - Int(1)) : Int(0);
            
            return r;
        }
        
        bool EmptyQ() const
        {
            return (ptr <= Int(0));
        }
        
        Int ElementCount() const
        {
            return ptr;
        }
        
        bool HasRoom( const Int element_count ) const
        {
            return ( (ptr + element_count) < Capacity() );
        }
        
        std::string String() const
        {
            return OutString::FromVector( &a[1], Size() );
        }
        
        friend std::string ToString( const Stack & stack )
        {
            return stack.String();
        }
        
        Tensor1<Entry_T,Int> & GetTensor()
        {
            return a;
        }
        
    private:
        
        Tensor1<Entry_T,Int> a;
        
        Int ptr = 0;
        
    public:
        
        static std::string ClassName()
        {
            return std::string("Stack")
            + "<" + TypeName<Entry_T>
            + "," + TypeName<Int>
            + ">";
        }
        
        
    }; // class Stack
    
    
//    template<Size_T max_element_count_, typename Entry_T_, IntQ Int_>
//    class FixedSizePairStack
//    {
//
//    public:
//        
//        using Entry_T = Entry_T_;
//        using Int     = Int_;
//        
//        static constexpr Int max_element_count = static_cast<Int>(max_element_count_);
//        static constexpr Int actual_element_count = max_element_count + 1;
//        
////        using Pair_T      = std::array<Entry_T,2>;
//        using Pair_T      = std::pair<Entry_T,Entry_T>;
////        using Container_T = std::array<Pair_T,actual_size>;
//        
//        using Container_T = Tiny::Matrix<actual_element_count,2,Entry_T,Int>;
//        
//        
//        FixedSizePairStack()
//        {
//            a[0][0] = Entry_T();
//            a[0][1] = Entry_T();
//             
////            a[0] = { Entry_T(), Entry_T() };
//        }
//        
//        ~FixedSizePairStack() = default;
//        
//        // Copy constructor
//        FixedSizePairStack( const FixedSizePairStack & other )
////        :   a   ( other.a   )
//        :   ptr ( other.ptr )
//        {
//            std::copy<actual_element_count * 2>( &other.a[0][0], &a[0][0] );
//        }
//        
//        inline friend void swap( FixedSizePairStack & A, FixedSizePairStack & B) noexcept
//        {
//            using std::swap;
//            
//            if( &A == &B )
//            {
//                wprint( std::string("An object of type ") + ClassName() + " has been swapped to itself.");
//            }
//            else
//            {
////                swap( A.a  , B.a   );
//                
//                std::swap_ranges(
//                    &A.a[0][0],
//                    &A.a[0][0] + 2 * actual_element_count,
//                    &B.a[0][0]
//                );
//                swap( A.ptr, B.ptr );
//            }
//        }
//        
//        // Copy-assignment operator
//        FixedSizePairStack & operator=( FixedSizePairStack other ) noexcept
//        {
//            swap(*this, other);
//            
//            return *this;
//        }
//        
//        
//        // Move constructor
//        FixedSizePairStack( FixedSizePairStack && other ) noexcept
//        :   FixedSizePairStack()
//        {
//            swap(*this, other);
//        }
//        
//        
//        static constexpr Int Size()
//        {
//            return max_element_count;
//        }
//        
//        void Reset()
//        {
//            ptr = 0;
//        }
//        
//        inline void Push( cref<Entry_T> val_0, cref<Entry_T> val_1 )
//        {
//            ++ptr;
//            a[ptr][0] = val_0;
//            a[ptr][1] = val_1;
//        }
//
//        Pair_T Top() const
//        {
//            return { a[ptr][0], a[ptr][1] };
//        }
//        
//        inline Pair_T Pop()
//        {
//            Pair_T r { a[ptr][0], a[ptr][1] };
//            --ptr;
//            return r;
//        }
//        
//        bool EmptyQ() const
//        {
//            return ptr <= 0;
//        }
//        
//        Int ElementCount() const
//        {
//            return ptr;
//        }
//         
//        template<Int element_count>
//        bool HasRoom() const
//        {
//            static_assert( element_count <= actual_element_count, "" );
//            
//            return ( ptr < actual_element_count - element_count );
//        }
//        
//        bool HasRoom( const Int element_count ) const
//        {
//            return ( (ptr + element_count) < actual_element_count );
//        }
//        
//        std::string String() const
//        {
//            return OutString::FromMatrix( &a[1][0], ptr, 2 );
//        }
//        
//        friend std::string ToString( const FixedSizePairStack & stack )
//        {
//            return stack.String();
//        }
//        
////        cref<Container_T> GetCcontainer() const
////        {
////            return a;
////        }
//        
//    private:
//        
//        
//        Entry_T a [actual_element_count][2];
//        
//        Int ptr = 0;
//        
//    public:
//        
//        static std::string ClassName()
//        {
//            return std::string("FixedSizePairStack")
//            + "<" + TypeName<Entry_T>
//            + "," + TypeName<Int>
//            + ">";
//        }
//        
//        
//    }; // class FixedSizePairStack

} // namespace Tensors

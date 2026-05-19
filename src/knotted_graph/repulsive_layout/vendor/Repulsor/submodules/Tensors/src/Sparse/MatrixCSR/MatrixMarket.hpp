public: 

void LoadFromMatrixMarket( cref<std::filesystem::path> file, Int thread_count_ )
{
    std::string tag = ClassName()+"::LoadFromMatrixMarket";
    
    TOOLS_PTIMER(timer,tag);
    
    std::ifstream  s ( file );

    if( !s.good() )
    {
        eprint(tag + ": File " + file.string() + " not found. Aborting.");
        return;
    }
    
    logprint("Loading from file " + file.string() );
    
    std::string token;
    
    s >> token;
    
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
    if( token != "%%matrixmarket")
    {
        eprint( tag + ": Not a MatrixMarket file. Doing nothing.");
        TOOLS_DDUMP( token );
        return;
    }
    
    s >> token;
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
    
    if( token != "matrix")
    {
        eprint( tag + ": Second word in file is not \"matrix\". Doing nothing.");
        TOOLS_DDUMP( token );
        return;
    }
    
    s >> token;
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
    if( token != "coordinate")
    {
        eprint( tag + ": Third word in file is not \"coordinate\". Stored matrix is a dense matrix and shall better not be loaded. Doing nothing.");
        TOOLS_DDUMP( token );
        return;
    }
    
    std::string scalar_type;
    s >> scalar_type;
    std::transform(scalar_type.begin(), scalar_type.end(), scalar_type.begin(), ::tolower);
    if constexpr ( Scalar::RealQ<Scal> )
    {
        if( scalar_type == "complex")
        {
            eprint( tag + ": Scalar type requested is " + TypeName<Scal> + ", but type in file is \"complex\". Doing nothing.");
            return;
        }
    }
    
    if constexpr ( IntQ<Scal> )
    {
        if( (scalar_type != "integer") && (scalar_type != "pattern") )
        {
            eprint( tag + ": Scalar type requested is " + TypeName<Scal> + ", but type in file is \"" + scalar_type + "\". Doing nothin.");
            return;
        }
    }
    
    std::string symmetry;
    s >> symmetry;
    std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::tolower);
    
    bool symmetrizeQ = false;
    
    if( symmetry == "skew-symmetric")
    {
        eprint( tag + ": Matrix symmetry is \"" + symmetry + "\". The current implementation cannot handle this. Doing nothing.");
        return;
    }
    else if ( symmetry == "hermitian")
    {
        eprint( tag + ": Matrix symmetry is \"" + symmetry + "\". The current implementation cannot handle this. Doing nothing.");
        return;
    }
    else if ( symmetry == "symmetric")
    {
        symmetrizeQ = true;
    }
    else if ( symmetry == "general")
    {
        symmetrizeQ = false;
    }
    else
    {
        eprint( tag + ": Matrix symmetry is \"" + symmetry + "\". This is invalid for the MatrixMarket format. Doing nothing.");
        TOOLS_DDUMP( token );
        return;
    }
        

    s.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    while( std::getline( s, token ) && (token[0] == '%') )
    {}
    
    std::stringstream line (token);
    
    Int row_count = 0;
    line >> row_count;
    TOOLS_PDUMP(row_count);
    
    Int col_count = 0;
    line >> col_count;
    TOOLS_PDUMP(col_count);
    
    LInt nonzero_count = 0;
    line >> nonzero_count;
    TOOLS_PDUMP(nonzero_count);

    if(row_count < Int(0))
    {
        eprint( tag + ": Invalid row_count." );
        return;
    }
    
    if(col_count < Int(0))
    {
        eprint( tag + ": Invalid col_count." );
        return;
    }
    
    if(nonzero_count < LInt(0))
    {
        eprint( tag + ": Invalid nonzero_count." );
        return;
    }
    
    Tensor1<Int, LInt> i_list ( nonzero_count );
    Tensor1<Int, LInt> j_list ( nonzero_count );
    Tensor1<Scal,LInt> a_list ( nonzero_count );
    
    // TODO: We could also use std::from_chars -- once that is broadly available with floating point support an all compilers.
    
    if ( scalar_type == "pattern" )
    {
        for( LInt k = 0; k < nonzero_count; ++k )
        {
            Int i;
            Int j;
            s >> i;
            i_list[k] = i - Int(1);
            s >> j;
            j_list[k] = j - Int(1);
            
            a_list[k] = Scal(1);
        }
    }
    else if ( scalar_type != "complex" )
    {
        for( LInt k = 0; k < nonzero_count; ++k )
        {
            Int i;
            Int j;
            
            s >> i;
            i_list[k] = i - Int(1);
            s >> j;
            j_list[k] = j - Int(1);
            
            s >> a_list[k];
        }
    }
    else if constexpr (Scalar::ComplexQ<Scal>)
    {
        for( LInt k = 0; k < nonzero_count; ++k )
        {
            Int i;
            Int j;
            s >> i;
            i_list[k] = i - Int(1);
            s >> j;
            j_list[k] = j - Int(1);
            
            Scalar::Real<Scal> re;
            Scalar::Real<Scal> im;
            s >> re;
            s >> im;
            
            a_list[k] = Scal(re,im);
        }
    }
    
    MatrixCSR A (
        nonzero_count,
        i_list.data(), j_list.data(), a_list.data(),
        row_count, col_count, thread_count_, true, symmetrizeQ
    );
    
    swap( *this, A );
}


void WriteToMatrixMarket( cref<std::filesystem::path> file )
{
    std::string tag = ClassName()+"::WriteToMatrixMarket";
    
    TOOLS_PTIMER(timer,tag);
    
    if( !WellFormedQ() )
    {
        eprint( tag + ": Matrix is not well-formed. Doing nothing." );
        return;
    }
    
    std::ofstream  s ( file );
    
    s << "%%MatrixMarket" << " " << "matrix" << " " << "coordinate" << " ";
    
    if constexpr ( Scalar::ComplexQ<Scal> )
    {
        s << "complex";
    }
    else if constexpr ( IntQ<Scal> )
    {
        s << "integer";
    }
    else /*if constexpr ( Scalar::RealQ<Scal> )*/
    {
        s << "real";
    }
    
    s << " " << "general" << "\n";
    s << RowCount() << " " << ColCount() << " " << NonzeroCount() << "\n";
    
    const Int s_thread_count = 4;
    
    std::vector<std::string> thread_strings ( s_thread_count );
    
    auto s_job_ptr = JobPointers<Int>( m, outer.data(), s_thread_count, false );
    
    Do<parQ>(
        [&,this]( const Int thread )
        {
            const Int i_begin = s_job_ptr[thread+0];
            const Int i_end   = s_job_ptr[thread+1];

            char line[128];
            
            std::string s_loc;
            
            // TODO: Use std::to_chars .
            
            for( Int i = i_begin; i < i_end; ++i )
            {
                const LInt k_begin = outer[i    ];
                const LInt k_end   = outer[i + 1];

                for( LInt k = k_begin; k < k_end; ++k )
                {
                    const Int  j = inner[k];
                    const Scal a = values[k];
                    
                    // TODO: Maybe use std::format instead?
                    if constexpr ( IntQ<Scal> )
                    {
                        std::snprintf(line, 128, "%d %d %d\n", i+1,j+1,a);
                    }
                    else if constexpr ( std::is_same_v<Scal,Complex64> )
                    {
                        std::snprintf(line, 128, "%d %d %.17E %.17E\n", i+1,j+1,Re(a),Im(a));
                    }
                    else if constexpr ( std::is_same_v<Scal,Complex32> )
                    {
                        std::snprintf(line, 128, "%d %d %.7E %.7E\n", i+1,j+1,Re(a),Im(a));
                    }
                    else if constexpr ( std::is_same_v<Scal,Real64> )
                    {
                        std::snprintf(line, 128, "%d %d %.17E\n", i+1,j+1,a);
                    }
                    else if constexpr ( std::is_same_v<Scal,Real32> )
                    {
                        std::snprintf(line, 128, "%d %d %.7E\n", i+1,j+1,a);
                    }
                    
                    s_loc += line;
                }
            }

            thread_strings[thread] = std::move(s_loc);
        },
        s_thread_count, s_thread_count
    );
    
    Do<parQ>(
        [&,this]( const Int thread )
        {
            thread_strings[2 * thread] += thread_strings[2 * thread + 2];
        },
        2, 2
    );
    
    s << thread_strings[0];
    s << thread_strings[2];
}

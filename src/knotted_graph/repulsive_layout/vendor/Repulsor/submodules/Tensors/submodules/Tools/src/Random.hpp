#pragma once

namespace Tools
{
    template<typename PRNG_T>
    PRNG_T InitializedRandomEngine()
    {
        using RNG_T = std::random_device;
        using RNG_result_type_T  = typename RNG_T::result_type;
        
        using PRNG_result_type_T = typename PRNG_T::result_type;
        
        // TODO: This cannot be universally true.
        static constexpr Size_T PRNG_T_state_size = 4;
        
        static constexpr Size_T seed_size  = (PRNG_T_state_size * sizeof(PRNG_result_type_T)) / sizeof(RNG_result_type_T);
        
        std::array<RNG_result_type_T,seed_size> seed;
        
        std::generate( seed.begin(), seed.end(), RNG_T() );
        
        std::seed_seq seed_sequence ( seed.begin(), seed.end() );
        
        return PRNG_T( seed_sequence );
    }
}
